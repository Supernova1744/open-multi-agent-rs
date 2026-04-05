use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("LLM API error: {0}")]
    LlmError(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Dependency cycle detected in tasks")]
    DependencyCycle,

    #[error("Max turns ({0}) exceeded")]
    MaxTurnsExceeded(usize),

    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;
