use std::io;

/// Generic numbered picker used by `/model`, `/permissions`, etc.
pub(super) fn between_turn_choice_picker(
    title: &str,
    options: &[(&str, &str)],
    current: &str,
) -> io::Result<Option<String>> {
    use std::io::Write;

    let mut stdout = io::stdout();
    writeln!(stdout, "\n{title} (Enter to cancel)")?;
    for (idx, (label, value)) in options.iter().enumerate() {
        let marker = if current == *value { "*" } else { " " };
        writeln!(stdout, "  {marker} {n}) {label}  [{value}]", n = idx + 1)?;
    }
    write!(stdout, "> ")?;
    stdout.flush()?;

    let mut line = String::new();
    let bytes = io::stdin().read_line(&mut line)?;
    if bytes == 0 {
        return Ok(None);
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if let Ok(n) = trimmed.parse::<usize>() {
        if let Some((_, value)) = options.get(n.wrapping_sub(1)) {
            return Ok(Some((*value).to_string()));
        }
    }
    Ok(Some(trimmed.to_string()))
}

pub(super) fn prompt_model_picker(current: &str) -> io::Result<Option<String>> {
    between_turn_choice_picker(
        "Select a model",
        &[
            ("Opus 4.8", "claude-opus-4-8"),
            ("Sonnet 4.6", "claude-sonnet-4-6"),
            ("Haiku 4.5", "claude-haiku-4-5-20251001"),
        ],
        &crate::cli_args::resolve_model_alias(current),
    )
}

pub(super) fn prompt_permissions_picker(current: &str) -> io::Result<Option<String>> {
    between_turn_choice_picker(
        "Select a permission mode",
        &[
            ("Read-only - no writes, no shell", "read-only"),
            (
                "Workspace write - edits inside the workspace",
                "workspace-write",
            ),
            ("Danger full access - no guardrails", "danger-full-access"),
        ],
        current,
    )
}
