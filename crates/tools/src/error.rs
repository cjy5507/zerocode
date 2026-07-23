use thiserror::Error;

/// Structured error type for tool execution failures.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool name not found in any dispatcher.
    #[error("unsupported tool: {0}")]
    NotFound(String),

    /// Permission denied by the enforcement layer.
    #[error("permission denied for `{tool}`: {reason}")]
    PermissionDenied { tool: String, reason: String },

    /// Invalid input (deserialization or validation failure).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Tool execution failed.
    #[error("execution error: {0}")]
    Execution(String),

    /// I/O error during tool execution.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Plugin tool name conflict.
    #[error("plugin tool `{0}` conflicts with a built-in tool name")]
    PluginConflict(String),

    /// Duplicate tool name.
    #[error("duplicate tool name: `{0}`")]
    DuplicateName(String),
}

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        Self::Execution(s)
    }
}
