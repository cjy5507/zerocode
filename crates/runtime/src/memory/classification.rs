use decision_core::dreamer::LessonKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySource {
    Unknown,
    HandWritten,
    Dreamer,
}

impl MemorySource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::HandWritten => "hand_written",
            Self::Dreamer => "dreamer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    Unknown,
    Preference,
    Gotcha,
    Workflow,
    Constraint,
    TaskLog,
}

impl MemoryKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Preference => "preference",
            Self::Gotcha => "gotcha",
            Self::Workflow => "workflow",
            Self::Constraint => "constraint",
            Self::TaskLog => "task_log",
        }
    }

    #[must_use]
    pub const fn from_lesson(kind: LessonKind) -> Self {
        match kind {
            LessonKind::Preference => Self::Preference,
            LessonKind::Gotcha => Self::Gotcha,
            LessonKind::Workflow => Self::Workflow,
            LessonKind::Constraint => Self::Constraint,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryClassification {
    pub source: MemorySource,
    pub kind: MemoryKind,
    pub protected: bool,
    pub resolved_task_log: bool,
    pub written_at: Option<u64>,
}

impl Default for MemoryClassification {
    fn default() -> Self {
        Self {
            source: MemorySource::Unknown,
            kind: MemoryKind::Unknown,
            protected: true,
            resolved_task_log: false,
            written_at: None,
        }
    }
}

#[must_use]
pub fn dreamer_memory_metadata_line(
    kind: MemoryKind,
    resolved_task_log: bool,
    written_at: Option<u64>,
) -> String {
    memory_metadata_line(MemorySource::Dreamer, kind, false, resolved_task_log, written_at)
}

#[must_use]
pub fn hand_written_memory_metadata_line(kind: MemoryKind, written_at: Option<u64>) -> String {
    memory_metadata_line(MemorySource::HandWritten, kind, true, false, written_at)
}

#[must_use]
pub fn memory_metadata_line(
    source: MemorySource,
    kind: MemoryKind,
    protected: bool,
    resolved_task_log: bool,
    written_at: Option<u64>,
) -> String {
    let written_at = written_at.map_or_else(|| "unknown".to_string(), |secs| secs.to_string());
    format!(
        "- memory_metadata: v=1;source={};kind={};protected={};resolved_task_log={};written_at={}",
        source.as_str(),
        kind.as_str(),
        protected,
        resolved_task_log,
        written_at
    )
}

#[must_use]
pub fn memory_body_has_classification_metadata(body: &str) -> bool {
    body.lines()
        .any(|line| line.trim().starts_with("- memory_metadata:"))
}

#[must_use]
pub fn classify_memory_body(body: &str) -> MemoryClassification {
    let Some(line) = body
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- memory_metadata:"))
    else {
        return MemoryClassification::default();
    };
    parse_memory_metadata_line(line).unwrap_or_default()
}

fn parse_memory_metadata_line(line: &str) -> Option<MemoryClassification> {
    let metadata = line.strip_prefix("- memory_metadata:")?.trim();
    let mut version_ok = false;
    let mut source = None;
    let mut kind = None;
    let mut protected = None;
    let mut resolved_task_log = None;
    let mut resolved_task_log_seen = false;
    let mut written_at = None;
    let mut written_at_seen = false;

    for part in metadata.split(';') {
        let (key, value) = part.trim().split_once('=')?;
        match key.trim() {
            "v" => version_ok = value.trim() == "1",
            "source" => source = parse_source(value.trim()),
            "kind" => kind = parse_kind(value.trim()),
            "protected" => protected = parse_bool(value.trim()),
            "resolved_task_log" => {
                resolved_task_log = parse_bool(value.trim());
                resolved_task_log_seen = true;
            }
            "written_at" => {
                written_at_seen = true;
                let value = value.trim();
                written_at = if value == "unknown" {
                    None
                } else {
                    Some(value.parse::<u64>().ok()?)
                };
            }
            _ => {}
        }
    }

    (version_ok && resolved_task_log_seen && written_at_seen).then_some(MemoryClassification {
        source: source?,
        kind: kind?,
        protected: protected?,
        resolved_task_log: resolved_task_log?,
        written_at,
    })
}

fn parse_source(value: &str) -> Option<MemorySource> {
    match value {
        "unknown" => Some(MemorySource::Unknown),
        "hand_written" => Some(MemorySource::HandWritten),
        "dreamer" => Some(MemorySource::Dreamer),
        _ => None,
    }
}

fn parse_kind(value: &str) -> Option<MemoryKind> {
    match value {
        "unknown" => Some(MemoryKind::Unknown),
        "preference" => Some(MemoryKind::Preference),
        "gotcha" => Some(MemoryKind::Gotcha),
        "workflow" => Some(MemoryKind::Workflow),
        "constraint" => Some(MemoryKind::Constraint),
        "task_log" => Some(MemoryKind::TaskLog),
        _ => None,
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_malformed_metadata_is_protected_unknown() {
        assert_eq!(classify_memory_body("plain note"), MemoryClassification::default());
        assert_eq!(
            classify_memory_body(
                "---\n- memory_metadata: v=1;source=dreamer;kind=gotcha;protected=false;resolved_task_log=false;written_at=not-a-number"
            ),
            MemoryClassification::default()
        );
    }

    #[test]
    fn parses_dreamer_metadata_line() {
        let body = format!(
            "body\n\n---\n{}\n",
            dreamer_memory_metadata_line(MemoryKind::Gotcha, true, Some(42))
        );
        let classification = classify_memory_body(&body);
        assert_eq!(classification.source, MemorySource::Dreamer);
        assert_eq!(classification.kind, MemoryKind::Gotcha);
        assert!(!classification.protected);
        assert!(classification.resolved_task_log);
        assert_eq!(classification.written_at, Some(42));
    }
}
