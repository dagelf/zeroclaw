//! Report template engine for project delivery intelligence.
//!
//! Loads templates from TOML locale files in `report_templates/` with English
//! fallback, matching the `tool_descriptions/` i18n pattern from `src/i18n.rs`.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use tracing::debug;

/// Supported report output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Markdown,
    Html,
}

/// A named section within a report template.
#[derive(Debug, Clone)]
pub struct TemplateSection {
    pub heading: String,
    pub body: String,
}

/// A report template with named sections and variable placeholders.
#[derive(Debug, Clone)]
pub struct ReportTemplate {
    pub name: String,
    pub sections: Vec<TemplateSection>,
    pub format: ReportFormat,
}

/// Escape a string for safe inclusion in HTML output.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

impl ReportTemplate {
    /// Render the template by substituting `{{key}}` placeholders with values.
    pub fn render(&self, vars: &HashMap<String, String>) -> String {
        let mut out = String::new();
        for section in &self.sections {
            let heading = substitute(&section.heading, vars);
            let body = substitute(&section.body, vars);
            match self.format {
                ReportFormat::Markdown => {
                    let _ = write!(out, "## {heading}\n\n{body}\n\n");
                }
                ReportFormat::Html => {
                    let heading = escape_html(&heading);
                    let body = escape_html(&body);
                    let _ = write!(out, "<h2>{heading}</h2>\n<p>{body}</p>\n");
                }
            }
        }
        out.trim_end().to_string()
    }
}

/// Single-pass placeholder substitution.
///
/// Scans `template` left-to-right for `{{key}}` tokens and replaces them with
/// the corresponding value from `vars`.  Because the scan is single-pass,
/// values that themselves contain `{{...}}` sequences are emitted literally
/// and never re-expanded, preventing injection of new placeholders.
fn substitute(template: &str, vars: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if i + 1 < len && bytes[i] == b'{' && bytes[i + 1] == b'{' {
            // Find the closing `}}`.
            if let Some(close) = template[i + 2..].find("}}") {
                let key = &template[i + 2..i + 2 + close];
                if let Some(value) = vars.get(key) {
                    result.push_str(value);
                } else {
                    // Unknown placeholder: emit as-is.
                    result.push_str(&template[i..i + 2 + close + 2]);
                }
                i += 2 + close + 2;
                continue;
            }
        }
        result.push(template.as_bytes()[i] as char);
        i += 1;
    }

    result
}

// ── TOML deserialization structures ─────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct TomlSection {
    heading: String,
    body: String,
}

#[derive(Debug, serde::Deserialize)]
struct TomlTemplate {
    name: String,
    #[serde(default)]
    sections: Vec<TomlSection>,
}

#[derive(Debug, serde::Deserialize)]
struct TomlTemplateFile {
    weekly_status: Option<TomlTemplate>,
    sprint_review: Option<TomlTemplate>,
    risk_register: Option<TomlTemplate>,
    milestone_report: Option<TomlTemplate>,
}

/// Try to load and parse a report template TOML file from the first matching search dir.
fn load_template_file(locale: &str, search_dirs: &[PathBuf]) -> Option<TomlTemplateFile> {
    let filename = format!("report_templates/{locale}.toml");

    for dir in search_dirs {
        let path = dir.join(&filename);
        match std::fs::read_to_string(&path) {
            Ok(contents) => match toml::from_str::<TomlTemplateFile>(&contents) {
                Ok(parsed) => {
                    debug!(path = %path.display(), locale = locale, "loaded report template file");
                    return Some(parsed);
                }
                Err(e) => {
                    debug!(path = %path.display(), error = %e, "failed to parse report template file");
                }
            },
            Err(_) => {}
        }
    }
    None
}

/// Extract a specific template from a parsed TOML file.
fn extract_template(file: &TomlTemplateFile, template_name: &str) -> Option<ReportTemplate> {
    let toml_tpl = match template_name {
        "weekly_status" => file.weekly_status.as_ref(),
        "sprint_review" => file.sprint_review.as_ref(),
        "risk_register" => file.risk_register.as_ref(),
        "milestone_report" => file.milestone_report.as_ref(),
        _ => None,
    }?;

    Some(ReportTemplate {
        name: toml_tpl.name.clone(),
        sections: toml_tpl
            .sections
            .iter()
            .map(|s| TemplateSection {
                heading: s.heading.clone(),
                body: s.body.clone(),
            })
            .collect(),
        format: ReportFormat::Markdown,
    })
}

/// Build the default set of search directories for report template files.
fn default_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.to_path_buf());
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !dirs.contains(&manifest_dir) {
        dirs.push(manifest_dir);
    }

    dirs
}

/// Load a named template for the given language from TOML files.
///
/// Resolution order:
/// 1. `report_templates/{lang}.toml`
/// 2. `report_templates/en.toml` (English fallback)
/// 3. Hardcoded English default (always available)
fn load_template(template_name: &str, lang: &str) -> ReportTemplate {
    let search_dirs = default_search_dirs();

    // Try the requested locale first.
    if let Some(file) = load_template_file(lang, &search_dirs) {
        if let Some(tpl) = extract_template(&file, template_name) {
            return tpl;
        }
    }

    // Fallback to English TOML file.
    if lang != "en" {
        if let Some(file) = load_template_file("en", &search_dirs) {
            if let Some(tpl) = extract_template(&file, template_name) {
                return tpl;
            }
        }
    }

    // Hardcoded English fallback (always available, no file dependency).
    hardcoded_english(template_name)
}

/// Hardcoded English templates as a final fallback when no TOML files are found.
fn hardcoded_english(template_name: &str) -> ReportTemplate {
    match template_name {
        "weekly_status" => ReportTemplate {
            name: "Weekly Status".into(),
            sections: vec![
                TemplateSection { heading: "Summary".into(), body: "Project: {{project_name}} | Period: {{period}}".into() },
                TemplateSection { heading: "Completed".into(), body: "{{completed}}".into() },
                TemplateSection { heading: "In Progress".into(), body: "{{in_progress}}".into() },
                TemplateSection { heading: "Blocked".into(), body: "{{blocked}}".into() },
                TemplateSection { heading: "Next Steps".into(), body: "{{next_steps}}".into() },
            ],
            format: ReportFormat::Markdown,
        },
        "sprint_review" => ReportTemplate {
            name: "Sprint Review".into(),
            sections: vec![
                TemplateSection { heading: "Sprint".into(), body: "{{sprint_dates}}".into() },
                TemplateSection { heading: "Completed".into(), body: "{{completed}}".into() },
                TemplateSection { heading: "In Progress".into(), body: "{{in_progress}}".into() },
                TemplateSection { heading: "Blocked".into(), body: "{{blocked}}".into() },
                TemplateSection { heading: "Velocity".into(), body: "{{velocity}}".into() },
            ],
            format: ReportFormat::Markdown,
        },
        "risk_register" => ReportTemplate {
            name: "Risk Register".into(),
            sections: vec![
                TemplateSection { heading: "Project".into(), body: "{{project_name}}".into() },
                TemplateSection { heading: "Risks".into(), body: "{{risks}}".into() },
                TemplateSection { heading: "Mitigations".into(), body: "{{mitigations}}".into() },
            ],
            format: ReportFormat::Markdown,
        },
        _ => ReportTemplate {
            name: "Milestone Report".into(),
            sections: vec![
                TemplateSection { heading: "Project".into(), body: "{{project_name}}".into() },
                TemplateSection { heading: "Milestones".into(), body: "{{milestones}}".into() },
                TemplateSection { heading: "Status".into(), body: "{{status}}".into() },
            ],
            format: ReportFormat::Markdown,
        },
    }
}

// ── Public API (preserves existing signatures) ──────────────────────

/// Return the built-in weekly status template for the given language.
pub fn weekly_status_template(lang: &str) -> ReportTemplate {
    load_template("weekly_status", lang)
}

/// Return the built-in sprint review template for the given language.
pub fn sprint_review_template(lang: &str) -> ReportTemplate {
    load_template("sprint_review", lang)
}

/// Return the built-in risk register template for the given language.
pub fn risk_register_template(lang: &str) -> ReportTemplate {
    load_template("risk_register", lang)
}

/// Return the built-in milestone report template for the given language.
pub fn milestone_report_template(lang: &str) -> ReportTemplate {
    load_template("milestone_report", lang)
}

/// High-level template rendering function.
///
/// Returns the rendered template as a string or an error if the template
/// is not supported.
#[allow(clippy::implicit_hasher)]
pub fn render_template(
    template_name: &str,
    language: &str,
    vars: &HashMap<String, String>,
) -> anyhow::Result<String> {
    let tpl = match template_name {
        "weekly_status" | "sprint_review" | "risk_register" | "milestone_report" => {
            load_template(template_name, language)
        }
        _ => anyhow::bail!("unsupported template: {}", template_name),
    };
    Ok(tpl.render(vars))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weekly_status_renders_with_variables() {
        let tpl = weekly_status_template("en");
        let mut vars = HashMap::new();
        vars.insert("project_name".into(), "ZeroClaw".into());
        vars.insert("period".into(), "2026-W10".into());
        vars.insert("completed".into(), "- Task A\n- Task B".into());
        vars.insert("in_progress".into(), "- Task C".into());
        vars.insert("blocked".into(), "None".into());
        vars.insert("next_steps".into(), "- Task D".into());

        let rendered = tpl.render(&vars);
        assert!(rendered.contains("Project: ZeroClaw"));
        assert!(rendered.contains("Period: 2026-W10"));
        assert!(rendered.contains("- Task A"));
        assert!(rendered.contains("## Completed"));
    }

    #[test]
    fn weekly_status_de_renders_german_headings() {
        let tpl = weekly_status_template("de");
        let vars = HashMap::new();
        let rendered = tpl.render(&vars);
        assert!(rendered.contains("## Zusammenfassung"));
        assert!(rendered.contains("## Erledigt"));
    }

    #[test]
    fn weekly_status_fr_renders_french_headings() {
        let tpl = weekly_status_template("fr");
        let vars = HashMap::new();
        let rendered = tpl.render(&vars);
        assert!(rendered.contains("## Resume"));
        assert!(rendered.contains("## Termine"));
    }

    #[test]
    fn weekly_status_it_renders_italian_headings() {
        let tpl = weekly_status_template("it");
        let vars = HashMap::new();
        let rendered = tpl.render(&vars);
        assert!(rendered.contains("## Riepilogo"));
        assert!(rendered.contains("## Completato"));
    }

    #[test]
    fn html_format_renders_tags() {
        let mut tpl = weekly_status_template("en");
        tpl.format = ReportFormat::Html;
        let mut vars = HashMap::new();
        vars.insert("project_name".into(), "Test".into());
        vars.insert("period".into(), "W1".into());
        vars.insert("completed".into(), "Done".into());
        vars.insert("in_progress".into(), "WIP".into());
        vars.insert("blocked".into(), "None".into());
        vars.insert("next_steps".into(), "Next".into());

        let rendered = tpl.render(&vars);
        assert!(rendered.contains("<h2>Summary</h2>"));
        assert!(rendered.contains("<p>Project: Test | Period: W1</p>"));
    }

    #[test]
    fn sprint_review_template_has_velocity_section() {
        let tpl = sprint_review_template("en");
        let section_headings: Vec<&str> = tpl.sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(section_headings.contains(&"Velocity"));
    }

    #[test]
    fn risk_register_template_has_risk_sections() {
        let tpl = risk_register_template("en");
        let section_headings: Vec<&str> = tpl.sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(section_headings.contains(&"Risks"));
        assert!(section_headings.contains(&"Mitigations"));
    }

    #[test]
    fn milestone_template_all_languages() {
        for lang in &["en", "de", "fr", "it"] {
            let tpl = milestone_report_template(lang);
            assert!(!tpl.name.is_empty());
            assert_eq!(tpl.sections.len(), 3);
        }
    }

    #[test]
    fn substitute_leaves_unknown_placeholders() {
        let vars = HashMap::new();
        let result = substitute("Hello {{name}}", &vars);
        assert_eq!(result, "Hello {{name}}");
    }

    #[test]
    fn substitute_replaces_all_occurrences() {
        let mut vars = HashMap::new();
        vars.insert("x".into(), "1".into());
        let result = substitute("{{x}} and {{x}}", &vars);
        assert_eq!(result, "1 and 1");
    }

    #[test]
    fn unsupported_locale_falls_back_to_english() {
        let tpl = weekly_status_template("xx");
        let vars = HashMap::new();
        let rendered = tpl.render(&vars);
        assert!(rendered.contains("## Summary"));
        assert!(rendered.contains("## Completed"));
    }
}
