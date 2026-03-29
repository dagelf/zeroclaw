//! LLM call + response parsing for the agent tool-call loop.
//!
//! Handles streaming/non-streaming provider calls, native and text-based
//! tool-call parsing, cost tracking, and context-overflow recovery.

use super::{
    DraftEvent, ParsedToolCall, ToolLoopCancelled, build_native_assistant_history,
    build_native_assistant_history_from_parsed_calls, consume_provider_streaming_response,
    detect_tool_call_parse_issue, emergency_history_trim, fast_trim_tool_results, parse_tool_calls,
    record_tool_loop_cost_usage, scrub_credentials,
};
use crate::observability::{Observer, ObserverEvent, runtime_trace};
use crate::providers::{ChatMessage, ChatRequest, Provider, ToolCall};
use crate::util::truncate_with_ellipsis;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Outcome of calling the LLM and parsing its response.
pub(super) enum LlmCallOutcome {
    /// Successfully parsed the response into (possibly empty) tool calls.
    Parsed {
        response_text: String,
        parsed_text: String,
        tool_calls: Vec<ParsedToolCall>,
        assistant_history_content: String,
        native_tool_calls: Vec<ToolCall>,
        parse_issue_detected: bool,
        response_streamed_live: bool,
    },
    /// Context window exceeded; caller should trim history and `continue`.
    ContextRecovery,
    /// Unrecoverable error; caller should `return Err(...)`.
    Fatal(anyhow::Error),
}

/// Call the LLM (streaming or non-streaming) and parse tool calls from the response.
///
/// Returns [`LlmCallOutcome::ContextRecovery`] when a context-window-exceeded error
/// is recovered by trimming history, signalling the caller to `continue` the loop.
#[allow(clippy::too_many_arguments)]
pub(super) async fn call_llm_and_parse_response(
    active_provider: &dyn Provider,
    prepared_messages: &[ChatMessage],
    request_tools: Option<&[crate::tools::ToolSpec]>,
    use_native_tools: bool,
    active_model: &str,
    temperature: f64,
    cancellation_token: Option<&CancellationToken>,
    on_delta: Option<&tokio::sync::mpsc::Sender<DraftEvent>>,
    pacing: &crate::config::PacingConfig,
    active_provider_name: &str,
    iteration: usize,
    channel_name: &str,
    provider_name: &str,
    model: &str,
    turn_id: &str,
    observer: &dyn Observer,
    history: &mut Vec<ChatMessage>,
    llm_started_at: Instant,
) -> LlmCallOutcome {
    let should_consume_provider_stream = on_delta.is_some()
        && active_provider.supports_streaming()
        && (request_tools.is_none() || active_provider.supports_streaming_tool_events());
    tracing::debug!(
        has_on_delta = on_delta.is_some(),
        supports_streaming = active_provider.supports_streaming(),
        should_consume_provider_stream,
        "Streaming decision for iteration {}",
        iteration + 1,
    );
    let mut streamed_live_deltas = false;

    let chat_result = if should_consume_provider_stream {
        match consume_provider_streaming_response(
            active_provider,
            prepared_messages,
            request_tools,
            active_model,
            temperature,
            cancellation_token,
            on_delta,
        )
        .await
        {
            Ok(streamed) => {
                streamed_live_deltas = streamed.forwarded_live_deltas;
                Ok(crate::providers::ChatResponse {
                    text: Some(streamed.response_text),
                    tool_calls: streamed.tool_calls,
                    usage: None,
                    reasoning_content: None,
                })
            }
            Err(stream_err) => {
                tracing::warn!(
                    provider = active_provider_name,
                    model = active_model,
                    iteration = iteration + 1,
                    "provider streaming failed, falling back to non-streaming chat: {stream_err}"
                );
                runtime_trace::record_event(
                    "llm_stream_fallback",
                    Some(channel_name),
                    Some(active_provider_name),
                    Some(active_model),
                    Some(turn_id),
                    Some(false),
                    Some("provider stream failed; fallback to non-streaming chat"),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "error": scrub_credentials(&stream_err.to_string()),
                    }),
                );
                if let Some(tx) = on_delta {
                    let _ = tx.send(DraftEvent::Clear).await;
                }
                {
                    let chat_future = active_provider.chat(
                        ChatRequest {
                            messages: prepared_messages,
                            tools: request_tools,
                        },
                        active_model,
                        temperature,
                    );
                    if let Some(token) = cancellation_token {
                        tokio::select! {
                            () = token.cancelled() => Err(ToolLoopCancelled.into()),
                            result = chat_future => result,
                        }
                    } else {
                        chat_future.await
                    }
                }
            }
        }
    } else {
        // Non-streaming path: wrap with optional per-step timeout from
        // pacing config to catch hung model responses.
        let chat_future = active_provider.chat(
            ChatRequest {
                messages: prepared_messages,
                tools: request_tools,
            },
            active_model,
            temperature,
        );

        match pacing.step_timeout_secs {
            Some(step_secs) if step_secs > 0 => {
                let step_timeout = Duration::from_secs(step_secs);
                if let Some(token) = cancellation_token {
                    tokio::select! {
                        () = token.cancelled() => return LlmCallOutcome::Fatal(ToolLoopCancelled.into()),
                        result = tokio::time::timeout(step_timeout, chat_future) => {
                            match result {
                                Ok(inner) => inner,
                                Err(_) => Err(anyhow::anyhow!(
                                    "LLM inference step timed out after {step_secs}s (step_timeout_secs)"
                                )),
                            }
                        },
                    }
                } else {
                    match tokio::time::timeout(step_timeout, chat_future).await {
                        Ok(inner) => inner,
                        Err(_) => Err(anyhow::anyhow!(
                            "LLM inference step timed out after {step_secs}s (step_timeout_secs)"
                        )),
                    }
                }
            }
            _ => {
                if let Some(token) = cancellation_token {
                    tokio::select! {
                        () = token.cancelled() => return LlmCallOutcome::Fatal(ToolLoopCancelled.into()),
                        result = chat_future => result,
                    }
                } else {
                    chat_future.await
                }
            }
        }
    };

    match chat_result {
        Ok(resp) => {
            let (resp_input_tokens, resp_output_tokens) = resp
                .usage
                .as_ref()
                .map(|u| (u.input_tokens, u.output_tokens))
                .unwrap_or((None, None));

            observer.record_event(&ObserverEvent::LlmResponse {
                provider: provider_name.to_string(),
                model: model.to_string(),
                duration: llm_started_at.elapsed(),
                success: true,
                error_message: None,
                input_tokens: resp_input_tokens,
                output_tokens: resp_output_tokens,
            });

            // Record cost via task-local tracker (no-op when not scoped)
            let _ = resp
                .usage
                .as_ref()
                .and_then(|usage| record_tool_loop_cost_usage(provider_name, model, usage));

            let response_text = resp.text_or_empty().to_string();
            // First try native structured tool calls (OpenAI-format).
            // Fall back to text-based parsing (XML tags, markdown blocks,
            // GLM format) only if the provider returned no native calls —
            // this ensures we support both native and prompt-guided models.
            let mut calls: Vec<ParsedToolCall> = resp
                .tool_calls
                .iter()
                .map(|call| ParsedToolCall {
                    name: call.name.clone(),
                    arguments: serde_json::from_str::<serde_json::Value>(&call.arguments)
                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                    tool_call_id: Some(call.id.clone()),
                })
                .collect();
            let mut parsed_text = String::new();

            if calls.is_empty() {
                let (fallback_text, fallback_calls) = parse_tool_calls(&response_text);
                if !fallback_text.is_empty() {
                    parsed_text = fallback_text;
                }
                calls = fallback_calls;
            }

            let parse_issue = detect_tool_call_parse_issue(&response_text, &calls);
            if let Some(ref issue) = parse_issue {
                runtime_trace::record_event(
                    "tool_call_parse_issue",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(turn_id),
                    Some(false),
                    Some(issue.as_str()),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "response_excerpt": truncate_with_ellipsis(
                            &scrub_credentials(&response_text),
                            600
                        ),
                    }),
                );
            }

            runtime_trace::record_event(
                "llm_response",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(turn_id),
                Some(true),
                None,
                serde_json::json!({
                    "iteration": iteration + 1,
                    "duration_ms": llm_started_at.elapsed().as_millis(),
                    "input_tokens": resp_input_tokens,
                    "output_tokens": resp_output_tokens,
                    "raw_response": scrub_credentials(&response_text),
                    "native_tool_calls": resp.tool_calls.len(),
                    "parsed_tool_calls": calls.len(),
                }),
            );

            // Preserve native tool call IDs in assistant history so role=tool
            // follow-up messages can reference the exact call id.
            let reasoning_content = resp.reasoning_content.clone();
            let assistant_history_content = if resp.tool_calls.is_empty() {
                if use_native_tools {
                    build_native_assistant_history_from_parsed_calls(
                        &response_text,
                        &calls,
                        reasoning_content.as_deref(),
                    )
                    .unwrap_or_else(|| response_text.clone())
                } else {
                    response_text.clone()
                }
            } else {
                build_native_assistant_history(
                    &response_text,
                    &resp.tool_calls,
                    reasoning_content.as_deref(),
                )
            };

            let native_calls = resp.tool_calls;
            LlmCallOutcome::Parsed {
                response_text,
                parsed_text,
                tool_calls: calls,
                assistant_history_content,
                native_tool_calls: native_calls,
                parse_issue_detected: parse_issue.is_some(),
                response_streamed_live: streamed_live_deltas,
            }
        }
        Err(e) => {
            let safe_error = crate::providers::sanitize_api_error(&e.to_string());
            observer.record_event(&ObserverEvent::LlmResponse {
                provider: provider_name.to_string(),
                model: model.to_string(),
                duration: llm_started_at.elapsed(),
                success: false,
                error_message: Some(safe_error.clone()),
                input_tokens: None,
                output_tokens: None,
            });
            runtime_trace::record_event(
                "llm_response",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(turn_id),
                Some(false),
                Some(&safe_error),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "duration_ms": llm_started_at.elapsed().as_millis(),
                }),
            );

            // Context overflow recovery: trim history and retry
            if crate::providers::reliable::is_context_window_exceeded(&e) {
                tracing::warn!(
                    iteration = iteration + 1,
                    "Context window exceeded, attempting in-loop recovery"
                );

                // Step 1: fast-trim old tool results (cheap)
                let chars_saved = fast_trim_tool_results(history, 4);
                if chars_saved > 0 {
                    tracing::info!(
                        chars_saved,
                        "Context recovery: trimmed old tool results, retrying"
                    );
                    return LlmCallOutcome::ContextRecovery;
                }

                // Step 2: emergency drop oldest non-system messages
                let dropped = emergency_history_trim(history, 4);
                if dropped > 0 {
                    tracing::info!(dropped, "Context recovery: dropped old messages, retrying");
                    return LlmCallOutcome::ContextRecovery;
                }

                // Nothing left to trim — truly unrecoverable
                tracing::error!("Context overflow unrecoverable: no trimmable messages");
            }

            LlmCallOutcome::Fatal(e)
        }
    }
}
