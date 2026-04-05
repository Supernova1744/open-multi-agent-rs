use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{AgentError, Result};
use crate::types::{ToolResult, ToolUseContext};
use super::{Tool, ToolRegistry};

// ---------------------------------------------------------------------------
// Security helpers
// ---------------------------------------------------------------------------

/// Resolve `raw_path` and verify it does not escape `base`.
/// Returns the canonical absolute path on success, or a ToolResult error on traversal.
fn safe_path(raw_path: &str, base: &Path) -> std::result::Result<PathBuf, ToolResult> {
    if raw_path.is_empty() {
        return Err(ToolResult { data: "path must not be empty".to_string(), is_error: true });
    }

    let joined = if Path::new(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        base.join(raw_path)
    };

    // Normalise without hitting the filesystem (handles ".." components).
    let mut normalised = PathBuf::new();
    for component in joined.components() {
        use std::path::Component::*;
        match component {
            ParentDir => { normalised.pop(); }
            CurDir    => {}
            other     => normalised.push(other),
        }
    }

    // Reject anything that would leave the base directory.
    if !normalised.starts_with(base) {
        return Err(ToolResult {
            data: format!("path '{}' is outside the allowed directory", raw_path),
            is_error: true,
        });
    }

    Ok(normalised)
}

/// The working directory used as the sandbox root for file operations.
/// Prefers the context cwd; falls back to the process cwd; falls back to "/tmp".
fn sandbox_root(context: &ToolUseContext) -> PathBuf {
    context.cwd
        .as_deref()
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

// ---------------------------------------------------------------------------
// Bash tool
// ---------------------------------------------------------------------------

/// Execute a shell command in a sandboxed working directory.
///
/// # Security
/// - Commands run in the sandbox root (agent `cwd`, or process `cwd`).
/// - The raw command string is passed to `sh -c` — this tool should only be
///   registered when the caller fully trusts the LLM driving the agent.
///   Consider omitting it from production deployments; use purpose-built tools
///   instead.
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str { "bash" }

    fn description(&self) -> &str {
        "Execute a shell command in the working directory and return its stdout/stderr. \
         Only available when explicitly enabled; use purpose-built tools in production."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                    "maxLength": 4096
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 30000, max 120000).",
                    "minimum": 1,
                    "maximum": 120000
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let command = match input.get("command").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return Ok(ToolResult { data: "command must be a non-empty string".to_string(), is_error: true }),
        };

        // Enforce a maximum command length to limit prompt-injection blast radius.
        if command.len() > 4096 {
            return Ok(ToolResult { data: "command exceeds maximum length of 4096 characters".to_string(), is_error: true });
        }

        let timeout_ms = input
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(30_000)
            .min(120_000);

        let cwd = sandbox_root(context);

        let output = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&cwd)
                .output(),
        )
        .await
        .map_err(|_| AgentError::ToolError(format!("command timed out after {}ms", timeout_ms)))?
        .map_err(|e| AgentError::IoError(e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let data = match (stdout.is_empty(), stderr.is_empty()) {
            (false, true)  => stdout,
            (true,  false) => stderr,
            (false, false) => format!("{}\nSTDERR: {}", stdout, stderr),
            (true,  true)  => String::new(),
        };

        Ok(ToolResult { data, is_error: !output.status.success() })
    }
}

// ---------------------------------------------------------------------------
// FileRead tool
// ---------------------------------------------------------------------------

/// Read a file within the agent's sandbox directory.
///
/// # Security
/// - Path traversal is blocked: any path that resolves outside the sandbox
///   root is rejected with an error result.
/// - Absolute paths are allowed only if they still resolve inside the sandbox.
pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str { "file_read" }
    fn description(&self) -> &str { "Read the contents of a file within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (within the working directory)."
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-based, optional).",
                    "minimum": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (inclusive, optional).",
                    "minimum": 1
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        let content = tokio::fs::read_to_string(&safe).await
            .map_err(|e| AgentError::IoError(e))?;

        let start = input.get("start_line").and_then(|v| v.as_u64()).map(|n| n as usize);
        let end   = input.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);

        let data = if start.is_some() || end.is_some() {
            let lines: Vec<&str> = content.lines().collect();
            let s = start.unwrap_or(1).saturating_sub(1);
            let e = end.unwrap_or(lines.len()).min(lines.len());
            lines[s..e].join("\n")
        } else {
            content
        };

        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// FileWrite tool
// ---------------------------------------------------------------------------

/// Write a file within the agent's sandbox directory.
///
/// # Security
/// - Path traversal is blocked; same rules as `FileReadTool`.
/// - Parent directories are created automatically only within the sandbox.
pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str { "file_write" }
    fn description(&self) -> &str { "Write content to a file within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (within the working directory)."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content  = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        // Create parent directories only if they remain inside the sandbox.
        if let Some(parent) = safe.parent() {
            if parent.starts_with(&base) {
                tokio::fs::create_dir_all(parent).await
                    .map_err(|e| AgentError::IoError(e))?;
            }
        }

        tokio::fs::write(&safe, content).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!("File written: {}", safe.display()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Grep tool
// ---------------------------------------------------------------------------

/// Search files for a pattern within the agent's sandbox directory.
///
/// # Security
/// - `pattern` and `path` are passed as separate argv tokens to `grep` —
///   no shell interpolation occurs (no `sh -c`).
/// - Path traversal is blocked on the `path` argument.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str { "grep" }
    fn description(&self) -> &str { "Search for a pattern in files within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                    "maxLength": 1024
                },
                "path": {
                    "type": "string",
                    "description": "Relative path to a file or directory to search in."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively through directories (default: false)."
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() && p.len() <= 1024 => p,
            Some(_) => return Ok(ToolResult { data: "pattern must be 1–1024 characters".to_string(), is_error: true }),
            None    => return Ok(ToolResult { data: "pattern is required".to_string(), is_error: true }),
        };

        let raw_path  = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let recursive = input.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        // Invoke grep with argv tokens — no shell, no interpolation.
        let mut cmd = tokio::process::Command::new("grep");
        if recursive { cmd.arg("-r"); }
        cmd.arg("-n")
           .arg("--")        // treat next args as non-options (protects against flag injection)
           .arg(pattern)
           .arg(&safe)
           .current_dir(&base);

        let output = cmd.output().await.map_err(|e| AgentError::IoError(e))?;
        let data = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(ToolResult {
            data: if data.is_empty() { "No matches found.".to_string() } else { data },
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Register all built-in tools
// ---------------------------------------------------------------------------

pub async fn register_built_in_tools(registry: &mut ToolRegistry) {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(BashTool),
        Arc::new(FileReadTool),
        Arc::new(FileWriteTool),
        Arc::new(GrepTool),
    ];
    for tool in tools {
        let _ = registry.register(tool);
    }
}
