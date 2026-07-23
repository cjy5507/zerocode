use std::env;
use std::path::PathBuf;

use compat_harness::{extract_manifest, UpstreamPaths};
use runtime::{
    load_system_prompt, resolve_sandbox_status, ConfigLoader, PermissionMode, TokenUsage,
};
use tools::GlobalToolRegistry;

use crate::resume::StatusUsage;
use crate::session::build_runtime_plugin_state_with_loader;
use crate::{format_status_report, render_version_report, status_context};

pub(crate) fn current_tool_registry() -> Result<GlobalToolRegistry, String> {
    let cwd = crate::current_cli_cwd().map_err(|error| error.to_string())?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader.load().map_err(|error| error.to_string())?;
    let state = build_runtime_plugin_state_with_loader(&cwd, &loader, &runtime_config, None)
        .map_err(|error| error.to_string())?;
    let registry = state.tool_registry.clone();
    if let Some(mcp_state) = state.mcp_state {
        mcp_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .shutdown()
            .map_err(|error| error.to_string())?;
    }
    Ok(registry)
}

pub(crate) fn dump_manifests() {
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = UpstreamPaths::from_workspace_dir(&workspace_dir);
    match extract_manifest(&paths) {
        Ok(manifest) => {
            println!("commands: {}", manifest.commands.entries().len());
            println!("tools: {}", manifest.tools.entries().len());
            println!("bootstrap phases: {}", manifest.bootstrap.phases().len());
        }
        Err(error) => {
            eprintln!("failed to extract manifests: {error}");
            std::process::exit(1);
        }
    }
}

pub(crate) fn print_bootstrap_plan() {
    for phase in runtime::BootstrapPlan::claude_code_default().phases() {
        println!("- {phase:?}");
    }
}

pub(crate) fn print_system_prompt(cwd: PathBuf, date: String) {
    match load_system_prompt(cwd, date, env::consts::OS, "unknown") {
        Ok(sections) => println!("{}", sections.join("\n\n")),
        Err(error) => {
            eprintln!("failed to build system prompt: {error}");
            std::process::exit(1);
        }
    }
}

pub(crate) fn print_version() {
    println!("{}", render_version_report());
}

pub(crate) fn print_status_snapshot(
    model: &str,
    permission_mode: PermissionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        format_status_report(
            model,
            StatusUsage {
                message_count: 0,
                turns: 0,
                latest: TokenUsage::default(),
                cumulative: TokenUsage::default(),
                estimated_tokens: 0,
            },
            permission_mode.as_str(),
            &status_context(None)?,
        )
    );
    Ok(())
}

pub(crate) fn print_sandbox_status_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader
        .load()
        .unwrap_or_else(|_| runtime::RuntimeConfig::empty());
    println!(
        "{}",
        crate::format_sandbox_report(&resolve_sandbox_status(runtime_config.sandbox(), &cwd))
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::print_version;
    use crate::render_version_report;

    #[test]
    fn print_version_uses_shared_renderer_text() {
        assert_eq!(render_version_report(), crate::render_version_report());
        let _ = print_version as fn();
    }
}
