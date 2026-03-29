//! Tool execution pipeline for one iteration of the agent loop.
//!
//! Runs before/after hooks, approval gates, dedup checks, parallel or
//! sequential tool execution, progress reporting, and loop detection.

use super::{
    DraftEvent, ParsedToolCall, ToolExecutionOutcome, canonicalize_json_for_tool_signature,
    execute_tools_parallel, execute_tools_sequential, maybe_inject_channel_delivery_defaults,
    scrub_credentials, should_execute_tools_in_parallel, truncate_tool_result,
};
use crate::approval::{ApprovalManager, ApprovalRequest, ApprovalResponse};
use crate::observability::{Observer, runtime_trace};
use crate::providers::ChatMessage;
use crate::tools::Tool;
use crate::util::truncate_with_ellipsis;
use anyhow::Result;
use std::collections::HashSet;
use std::fmt::Write;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Aggregated results from executing one iteration's tool calls.
pub(super) struct ToolPipelineResult {
    pub(super) tool_results: String,
    pub(super) individual_results: Vec<(Option<String>, String)>,
    pub(super) detection_relevant_output: String,
}

/// Execute all tool calls for one iteration: run hooks, approval, dedup,
/// then execute tools (parallel or sequential), collect results, and feed
/// the loop detector.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_tool_call_pipeline(
    tool_calls: &[ParsedToolCall],
    hooks: Option<&crate::hooks::HookRunner>,
    channel_name: &str,
    provider_name: &str,
    model: &str,
    turn_id: &str,
    on_delta: Option<&tokio::sync::mpsc::Sender<DraftEvent>>,
    channel_reply_target: Option<&str>,
    approval: Option<&ApprovalManager>,
    dedup_exempt_tools: &[String],
    tools_registry: &[Box<dyn Tool>],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    observer: &dyn Observer,
    cancellation_token: Option<&CancellationToken>,
    iteration: usize,
    max_tool_result_chars: usize,
    loop_ignore_tools: &HashSet<&str>,
    loop_detector: &mut crate::agent::loop_detector::LoopDetector,
    history: &mut Vec<ChatMessage>,
) -> Result<ToolPipelineResult> {
    let mut seen_tool_signatures: HashSet<(String, String)> = HashSet::new();
    let mut tool_results = String::new();
    let mut individual_results: Vec<(Option<String>, String)> = Vec::new();
    let mut ordered_results: Vec<Option<(String, Option<String>, ToolExecutionOutcome)>> =
        (0..tool_calls.len()).map(|_| None).collect();
    let allow_parallel_execution = should_execute_tools_in_parallel(tool_calls, approval);
    let mut executable_indices: Vec<usize> = Vec::new();
    let mut executable_calls: Vec<ParsedToolCall> = Vec::new();

    for (idx, call) in tool_calls.iter().enumerate() {
        // ── Hook: before_tool_call (modifying) ──────────
        let mut tool_name = call.name.clone();
        let mut tool_args = call.arguments.clone();
        if let Some(hooks) = hooks {
            match hooks
                .run_before_tool_call(tool_name.clone(), tool_args.clone())
                .await
            {
                crate::hooks::HookResult::Cancel(reason) => {
                    tracing::info!(tool = %call.name, %reason, "tool call cancelled by hook");
                    let cancelled = format!("Cancelled by hook: {reason}");
                    runtime_trace::record_event(
                        "tool_call_result",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(turn_id),
                        Some(false),
                        Some(&cancelled),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "tool": call.name,
                            "arguments": scrub_credentials(&tool_args.to_string()),
                        }),
                    );
                    if let Some(tx) = on_delta {
                        let _ = tx
                            .send(DraftEvent::Progress(format!(
                                "\u{274c} {}: {}\n",
                                call.name,
                                truncate_with_ellipsis(&scrub_credentials(&cancelled), 200)
                            )))
                            .await;
                    }
                    ordered_results[idx] = Some((
                        call.name.clone(),
                        call.tool_call_id.clone(),
                        ToolExecutionOutcome {
                            output: cancelled,
                            success: false,
                            error_reason: Some(scrub_credentials(&reason)),
                            duration: Duration::ZERO,
                        },
                    ));
                    continue;
                }
                crate::hooks::HookResult::Continue((name, args)) => {
                    tool_name = name;
                    tool_args = args;
                }
            }
        }

        maybe_inject_channel_delivery_defaults(
            &tool_name,
            &mut tool_args,
            channel_name,
            channel_reply_target,
        );

        // ── Approval hook ────────────────────────────────
        if let Some(mgr) = approval {
            if mgr.needs_approval(&tool_name) {
                let request = ApprovalRequest {
                    tool_name: tool_name.clone(),
                    arguments: tool_args.clone(),
                };

                // Interactive CLI: prompt the operator.
                // Non-interactive (channels): auto-deny since no operator
                // is present to approve.
                let decision = if mgr.is_non_interactive() {
                    ApprovalResponse::No
                } else {
                    mgr.prompt_cli(&request)
                };

                mgr.record_decision(&tool_name, &tool_args, decision, channel_name);

                if decision == ApprovalResponse::No {
                    let denied = "Denied by user.".to_string();
                    runtime_trace::record_event(
                        "tool_call_result",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(turn_id),
                        Some(false),
                        Some(&denied),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "tool": tool_name.clone(),
                            "arguments": scrub_credentials(&tool_args.to_string()),
                        }),
                    );
                    if let Some(tx) = on_delta {
                        let _ = tx
                            .send(DraftEvent::Progress(format!(
                                "\u{274c} {}: {}\n",
                                tool_name, denied
                            )))
                            .await;
                    }
                    ordered_results[idx] = Some((
                        tool_name.clone(),
                        call.tool_call_id.clone(),
                        ToolExecutionOutcome {
                            output: denied.clone(),
                            success: false,
                            error_reason: Some(denied),
                            duration: Duration::ZERO,
                        },
                    ));
                    continue;
                }
            }
        }

        let signature = {
            let canonical_args = canonicalize_json_for_tool_signature(&tool_args);
            let args_json =
                serde_json::to_string(&canonical_args).unwrap_or_else(|_| "{}".to_string());
            (tool_name.trim().to_ascii_lowercase(), args_json)
        };
        let dedup_exempt = dedup_exempt_tools.iter().any(|e| e == &tool_name);
        if !dedup_exempt && !seen_tool_signatures.insert(signature) {
            let duplicate = format!(
                "Skipped duplicate tool call '{tool_name}' with identical arguments in this turn."
            );
            runtime_trace::record_event(
                "tool_call_result",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(turn_id),
                Some(false),
                Some(&duplicate),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "tool": tool_name.clone(),
                    "arguments": scrub_credentials(&tool_args.to_string()),
                    "deduplicated": true,
                }),
            );
            if let Some(tx) = on_delta {
                let _ = tx
                    .send(DraftEvent::Progress(format!(
                        "\u{274c} {}: {}\n",
                        tool_name, duplicate
                    )))
                    .await;
            }
            ordered_results[idx] = Some((
                tool_name.clone(),
                call.tool_call_id.clone(),
                ToolExecutionOutcome {
                    output: duplicate.clone(),
                    success: false,
                    error_reason: Some(duplicate),
                    duration: Duration::ZERO,
                },
            ));
            continue;
        }

        runtime_trace::record_event(
            "tool_call_start",
            Some(channel_name),
            Some(provider_name),
            Some(model),
            Some(turn_id),
            None,
            None,
            serde_json::json!({
                "iteration": iteration + 1,
                "tool": tool_name.clone(),
                "arguments": scrub_credentials(&tool_args.to_string()),
            }),
        );

        // ── Progress: tool start ────────────────────────────
        if let Some(tx) = on_delta {
            let hint = {
                let raw = match tool_name.as_str() {
                    "shell" => tool_args.get("command").and_then(|v| v.as_str()),
                    "file_read" | "file_write" => tool_args.get("path").and_then(|v| v.as_str()),
                    _ => tool_args
                        .get("action")
                        .and_then(|v| v.as_str())
                        .or_else(|| tool_args.get("query").and_then(|v| v.as_str())),
                };
                match raw {
                    Some(s) => truncate_with_ellipsis(s, 60),
                    None => String::new(),
                }
            };
            let progress = if hint.is_empty() {
                format!("\u{23f3} {}\n", tool_name)
            } else {
                format!("\u{23f3} {}: {hint}\n", tool_name)
            };
            tracing::debug!(tool = %tool_name, "Sending progress start to draft");
            let _ = tx.send(DraftEvent::Progress(progress)).await;
        }

        executable_indices.push(idx);
        executable_calls.push(ParsedToolCall {
            name: tool_name,
            arguments: tool_args,
            tool_call_id: call.tool_call_id.clone(),
        });
    }

    let executed_outcomes = if allow_parallel_execution && executable_calls.len() > 1 {
        execute_tools_parallel(
            &executable_calls,
            tools_registry,
            activated_tools,
            observer,
            cancellation_token,
        )
        .await?
    } else {
        execute_tools_sequential(
            &executable_calls,
            tools_registry,
            activated_tools,
            observer,
            cancellation_token,
        )
        .await?
    };

    for ((idx, call), outcome) in executable_indices
        .iter()
        .zip(executable_calls.iter())
        .zip(executed_outcomes.into_iter())
    {
        runtime_trace::record_event(
            "tool_call_result",
            Some(channel_name),
            Some(provider_name),
            Some(model),
            Some(turn_id),
            Some(outcome.success),
            outcome.error_reason.as_deref(),
            serde_json::json!({
                "iteration": iteration + 1,
                "tool": call.name.clone(),
                "duration_ms": outcome.duration.as_millis(),
                "output": scrub_credentials(&outcome.output),
            }),
        );

        // ── Hook: after_tool_call (void) ─────────────────
        if let Some(hooks) = hooks {
            let tool_result_obj = crate::tools::ToolResult {
                success: outcome.success,
                output: outcome.output.clone(),
                error: None,
            };
            hooks
                .fire_after_tool_call(&call.name, &tool_result_obj, outcome.duration)
                .await;
        }

        // ── Progress: tool completion ───────────────────────
        if let Some(tx) = on_delta {
            let secs = outcome.duration.as_secs();
            let progress_msg = if outcome.success {
                format!("\u{2705} {} ({secs}s)\n", call.name)
            } else if let Some(ref reason) = outcome.error_reason {
                format!(
                    "\u{274c} {} ({secs}s): {}\n",
                    call.name,
                    truncate_with_ellipsis(reason, 200)
                )
            } else {
                format!("\u{274c} {} ({secs}s)\n", call.name)
            };
            tracing::debug!(tool = %call.name, secs, "Sending progress complete to draft");
            let _ = tx.send(DraftEvent::Progress(progress_msg)).await;
        }

        ordered_results[*idx] = Some((call.name.clone(), call.tool_call_id.clone(), outcome));
    }

    // Collect tool results and build per-tool output for loop detection.
    // Only non-ignored tool outputs contribute to the identical-output hash.
    let mut detection_relevant_output = String::new();
    for (result_index, (tool_name, tool_call_id, outcome)) in ordered_results
        .into_iter()
        .enumerate()
        .filter_map(|(i, opt)| opt.map(|v| (i, v)))
    {
        if !loop_ignore_tools.contains(tool_name.as_str()) {
            detection_relevant_output.push_str(&outcome.output);

            // Feed the pattern-based loop detector with name + args + result.
            let args = tool_calls
                .get(result_index)
                .map(|c| &c.arguments)
                .unwrap_or(&serde_json::Value::Null);
            let det_result = loop_detector.record(&tool_name, args, &outcome.output);
            match det_result {
                crate::agent::loop_detector::LoopDetectionResult::Ok => {}
                crate::agent::loop_detector::LoopDetectionResult::Warning(ref msg) => {
                    tracing::warn!(tool = %tool_name, %msg, "loop detector warning");
                    // Inject a system nudge so the LLM adjusts strategy.
                    history.push(ChatMessage::system(format!("[Loop Detection] {msg}")));
                }
                crate::agent::loop_detector::LoopDetectionResult::Block(ref msg) => {
                    tracing::warn!(tool = %tool_name, %msg, "loop detector blocked tool call");
                    // Replace the tool output with the block message.
                    // We still continue the loop so the LLM sees the block feedback.
                    history.push(ChatMessage::system(format!(
                        "[Loop Detection — BLOCKED] {msg}"
                    )));
                }
                crate::agent::loop_detector::LoopDetectionResult::Break(msg) => {
                    runtime_trace::record_event(
                        "loop_detector_circuit_breaker",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(turn_id),
                        Some(false),
                        Some(&msg),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "tool": tool_name,
                        }),
                    );
                    anyhow::bail!("Agent loop aborted by loop detector: {msg}");
                }
            }
        }
        let result_output = truncate_tool_result(&outcome.output, max_tool_result_chars);
        individual_results.push((tool_call_id, result_output.clone()));
        let _ = writeln!(
            tool_results,
            "<tool_result name=\"{}\">\n{}\n</tool_result>",
            tool_name, result_output
        );
    }

    Ok(ToolPipelineResult {
        tool_results,
        individual_results,
        detection_relevant_output,
    })
}
