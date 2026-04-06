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
// FileDelete tool
// ---------------------------------------------------------------------------

/// Delete a file within the agent's sandbox directory.
pub struct FileDeleteTool;

#[async_trait]
impl Tool for FileDeleteTool {
    fn name(&self) -> &str { "file_delete" }
    fn description(&self) -> &str { "Delete a file within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to delete (within the working directory)."
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

        if safe.is_dir() {
            return Ok(ToolResult {
                data: format!("'{}' is a directory; use dir_delete to remove directories", raw_path),
                is_error: true,
            });
        }

        tokio::fs::remove_file(&safe).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!("Deleted: {}", safe.display()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// FileUpdate tool
// ---------------------------------------------------------------------------

/// Patch a file by replacing a literal string with new text, or overwriting a
/// line range — both operations stay within the sandbox.
pub struct FileUpdateTool;

#[async_trait]
impl Tool for FileUpdateTool {
    fn name(&self) -> &str { "file_update" }
    fn description(&self) -> &str {
        "Update a file within the working directory. \
         Supports two modes: \
         (1) replace the first occurrence of `old_text` with `new_text`; \
         (2) replace a range of lines (start_line..end_line, 1-based inclusive) \
             with `new_text`."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (within the working directory)."
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact string to find and replace (mode 1)."
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text (used in both modes)."
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to replace, 1-based (mode 2).",
                    "minimum": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to replace, 1-based inclusive (mode 2).",
                    "minimum": 1
                }
            },
            "required": ["path", "new_text"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let new_text = input.get("new_text").and_then(|v| v.as_str()).unwrap_or("");
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        let original = tokio::fs::read_to_string(&safe).await
            .map_err(|e| AgentError::IoError(e))?;

        let updated = if let Some(old_text) = input.get("old_text").and_then(|v| v.as_str()) {
            // Mode 1: literal string replacement (first occurrence).
            if !original.contains(old_text) {
                return Ok(ToolResult {
                    data: format!("old_text not found in '{}'", raw_path),
                    is_error: true,
                });
            }
            original.replacen(old_text, new_text, 1)
        } else if let Some(start) = input.get("start_line").and_then(|v| v.as_u64()) {
            // Mode 2: replace a line range.
            let mut lines: Vec<&str> = original.lines().collect();
            let end = input.get("end_line").and_then(|v| v.as_u64()).unwrap_or(start);
            let s = (start as usize).saturating_sub(1);
            let e = (end as usize).min(lines.len());
            if s >= lines.len() {
                return Ok(ToolResult {
                    data: format!("start_line {} is beyond the end of the file ({} lines)", start, lines.len()),
                    is_error: true,
                });
            }
            let replacement: Vec<&str> = new_text.lines().collect();
            lines.splice(s..e, replacement);
            lines.join("\n")
        } else {
            return Ok(ToolResult {
                data: "provide either old_text or start_line".to_string(),
                is_error: true,
            });
        };

        tokio::fs::write(&safe, &updated).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!("Updated: {}", safe.display()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// FileList tool
// ---------------------------------------------------------------------------

/// List the contents of a directory within the agent's sandbox.
pub struct FileListTool;

#[async_trait]
impl Tool for FileListTool {
    fn name(&self) -> &str { "file_list" }
    fn description(&self) -> &str {
        "List files and directories within the working directory. \
         Returns a newline-separated list of entries with type (file/dir) and size."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a directory (defaults to '.' — the working directory)."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Recursively list subdirectories (default: false)."
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path  = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let recursive = input.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        if !safe.is_dir() {
            return Ok(ToolResult {
                data: format!("'{}' is not a directory", raw_path),
                is_error: true,
            });
        }

        let mut lines: Vec<String> = Vec::new();
        list_dir(&safe, &base, recursive, &mut lines).await
            .map_err(|e| AgentError::IoError(e))?;

        let data = if lines.is_empty() {
            "(empty directory)".to_string()
        } else {
            lines.join("\n")
        };

        Ok(ToolResult { data, is_error: false })
    }
}

/// Recursive directory walker for FileListTool.
async fn list_dir(
    dir: &Path,
    base: &Path,
    recursive: bool,
    out: &mut Vec<String>,
) -> std::io::Result<()> {
    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let meta = entry.metadata().await?;
        let rel  = path.strip_prefix(base).unwrap_or(&path).display().to_string();
        if meta.is_dir() {
            out.push(format!("[dir]  {}", rel));
            if recursive {
                Box::pin(list_dir(&path, base, true, out)).await?;
            }
        } else {
            out.push(format!("[file] {}  ({} bytes)", rel, meta.len()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DirCreate tool
// ---------------------------------------------------------------------------

/// Create a directory (and all parents) within the agent's sandbox.
pub struct DirCreateTool;

#[async_trait]
impl Tool for DirCreateTool {
    fn name(&self) -> &str { "dir_create" }
    fn description(&self) -> &str { "Create a directory (and any missing parent directories) within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the directory to create."
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

        tokio::fs::create_dir_all(&safe).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!("Created directory: {}", safe.display()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// DirDelete tool
// ---------------------------------------------------------------------------

/// Recursively delete a directory within the agent's sandbox.
pub struct DirDeleteTool;

#[async_trait]
impl Tool for DirDeleteTool {
    fn name(&self) -> &str { "dir_delete" }
    fn description(&self) -> &str { "Recursively delete a directory and all its contents within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the directory to delete."
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

        // Never allow deleting the sandbox root itself.
        if safe == base {
            return Ok(ToolResult {
                data: "cannot delete the working directory root".to_string(),
                is_error: true,
            });
        }

        if !safe.is_dir() {
            return Ok(ToolResult {
                data: format!("'{}' is not a directory; use file_delete for files", raw_path),
                is_error: true,
            });
        }

        tokio::fs::remove_dir_all(&safe).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!("Deleted directory: {}", safe.display()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// FileMove tool
// ---------------------------------------------------------------------------

/// Move or rename a file or directory within the agent's sandbox.
pub struct FileMoveTool;

#[async_trait]
impl Tool for FileMoveTool {
    fn name(&self) -> &str { "file_move" }
    fn description(&self) -> &str { "Move or rename a file or directory within the working directory." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Relative source path."
                },
                "destination": {
                    "type": "string",
                    "description": "Relative destination path."
                }
            },
            "required": ["source", "destination"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_src  = input.get("source").and_then(|v| v.as_str()).unwrap_or("");
        let raw_dst  = input.get("destination").and_then(|v| v.as_str()).unwrap_or("");
        let base = sandbox_root(context);

        let src = match safe_path(raw_src, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };
        let dst = match safe_path(raw_dst, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        // Create destination parent directory if needed.
        if let Some(parent) = dst.parent() {
            if parent.starts_with(&base) {
                tokio::fs::create_dir_all(parent).await
                    .map_err(|e| AgentError::IoError(e))?;
            }
        }

        tokio::fs::rename(&src, &dst).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!("Moved {} → {}", src.display(), dst.display()),
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
// PythonRun tool
// ---------------------------------------------------------------------------

/// Resolve the Python interpreter: prefer `python3`, fall back to `python`.
fn python_bin() -> &'static str {
    // On Windows the interpreter is usually `python`; on Linux/macOS `python3`.
    // We probe once and cache via a simple static.
    use std::sync::OnceLock;
    static BIN: OnceLock<&'static str> = OnceLock::new();
    *BIN.get_or_init(|| {
        // Quick synchronous probe — only runs once per process.
        if std::process::Command::new("python3").arg("--version").output().is_ok() {
            "python3"
        } else {
            "python"
        }
    })
}

/// Run a Python script or an inline code snippet and return its output.
///
/// Accepts either:
/// - `file` — path to an existing `.py` file in the sandbox, or
/// - `code` — an inline Python snippet (written to a temp file and executed).
///
/// Both stdout and stderr are captured and returned.
pub struct PythonRunTool;

#[async_trait]
impl Tool for PythonRunTool {
    fn name(&self) -> &str { "python_run" }
    fn description(&self) -> &str {
        "Execute a Python script and return its stdout/stderr output. \
         Provide either `file` (path to a .py file in the working directory) \
         or `code` (an inline Python snippet to run directly)."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Relative path to a .py file to run (within the working directory)."
                },
                "code": {
                    "type": "string",
                    "description": "Inline Python code to execute."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Command-line arguments to pass to the script (only used with `file`)."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Execution timeout in milliseconds (default 30000, max 120000).",
                    "minimum": 1,
                    "maximum": 120000
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let timeout_ms = input
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(30_000)
            .min(120_000);

        let base = sandbox_root(context);
        let py   = python_bin();

        // Determine the script to run.
        let (script_path, _temp_file) = if let Some(raw) = input.get("file").and_then(|v| v.as_str()) {
            let safe = match safe_path(raw, &base) {
                Ok(p)  => p,
                Err(e) => return Ok(e),
            };
            if !safe.exists() {
                return Ok(ToolResult {
                    data: format!("file '{}' not found", raw),
                    is_error: true,
                });
            }
            (safe, None::<tempfile_guard::TempFileGuard>)
        } else if let Some(code) = input.get("code").and_then(|v| v.as_str()) {
            // Write inline code to a temp file inside the sandbox.
            let tmp_path = base.join(format!("_agent_run_{}.py", uuid::Uuid::new_v4().as_simple()));
            tokio::fs::write(&tmp_path, code).await
                .map_err(|e| AgentError::IoError(e))?;
            (tmp_path.clone(), Some(tempfile_guard::TempFileGuard(tmp_path)))
        } else {
            return Ok(ToolResult {
                data: "provide either `file` or `code`".to_string(),
                is_error: true,
            });
        };

        // Build the command.
        let mut cmd = tokio::process::Command::new(py);
        cmd.arg(&script_path).current_dir(&base);

        // Append extra args (only meaningful for file mode).
        if let Some(serde_json::Value::Array(args)) = input.get("args") {
            for a in args {
                if let Some(s) = a.as_str() {
                    cmd.arg(s);
                }
            }
        }

        let output = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            cmd.output(),
        )
        .await
        .map_err(|_| AgentError::ToolError(format!("python timed out after {}ms", timeout_ms)))?
        .map_err(|e| AgentError::IoError(e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let data = match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
            (false, true)  => stdout,
            (true,  false) => stderr,
            (false, false) => format!("{}\nSTDERR:\n{}", stdout.trim_end(), stderr.trim_end()),
            (true,  true)  => "(no output)".to_string(),
        };

        Ok(ToolResult { data, is_error: !output.status.success() })
    }
}

/// RAII guard that deletes a temp `.py` file when dropped.
mod tempfile_guard {
    pub struct TempFileGuard(pub std::path::PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

// ---------------------------------------------------------------------------
// PythonTest tool
// ---------------------------------------------------------------------------

/// Run `pytest` (or `python -m pytest`) on a file or directory and return the
/// test results.
pub struct PythonTestTool;

#[async_trait]
impl Tool for PythonTestTool {
    fn name(&self) -> &str { "python_test" }
    fn description(&self) -> &str {
        "Run pytest on a Python test file or directory and return the results. \
         Uses `pytest` if available, otherwise falls back to `python -m pytest`."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a test file or directory (defaults to '.' to discover all tests)."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Extra pytest arguments (e.g. [\"-v\", \"-k\", \"test_add\"])."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Execution timeout in milliseconds (default 60000, max 300000).",
                    "minimum": 1,
                    "maximum": 300000
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let timeout_ms = input
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(60_000)
            .min(300_000);

        let base = sandbox_root(context);
        let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        // Prefer the standalone `pytest` binary; fall back to `python -m pytest`.
        let mut cmd = if std::process::Command::new("pytest").arg("--version").output().is_ok() {
            let mut c = tokio::process::Command::new("pytest");
            c.arg(&safe);
            c
        } else {
            let mut c = tokio::process::Command::new(python_bin());
            c.arg("-m").arg("pytest").arg(&safe);
            c
        };

        // Always emit colours-off for clean text output.
        cmd.arg("--tb=short").arg("--no-header");
        cmd.current_dir(&base);

        // Append caller-supplied pytest arguments.
        if let Some(serde_json::Value::Array(args)) = input.get("args") {
            for a in args {
                if let Some(s) = a.as_str() {
                    cmd.arg(s);
                }
            }
        }

        let output = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            cmd.output(),
        )
        .await
        .map_err(|_| AgentError::ToolError(format!("pytest timed out after {}ms", timeout_ms)))?
        .map_err(|e| AgentError::IoError(e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let data = match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
            (false, _)    => format!("{}{}", stdout, if stderr.trim().is_empty() { String::new() } else { format!("\nSTDERR:\n{}", stderr.trim_end()) }),
            (true, false) => stderr,
            (true, true)  => "(no output)".to_string(),
        };

        // pytest exits with 0 = all passed, 1 = some failed, 2+ = error.
        let passed = output.status.code().map(|c| c == 0).unwrap_or(false);
        Ok(ToolResult { data, is_error: !passed })
    }
}

// ---------------------------------------------------------------------------
// PythonWrite tool
// ---------------------------------------------------------------------------

/// Write a Python source file and optionally validate its syntax immediately.
pub struct PythonWriteTool;

#[async_trait]
impl Tool for PythonWriteTool {
    fn name(&self) -> &str { "python_write" }
    fn description(&self) -> &str {
        "Write Python source code to a .py file within the working directory \
         and check it for syntax errors. Creates parent directories as needed."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the .py file to write (within the working directory)."
                },
                "code": {
                    "type": "string",
                    "description": "Python source code to write."
                },
                "check_syntax": {
                    "type": "boolean",
                    "description": "If true (default), run `python -m py_compile` on the file after writing to catch syntax errors."
                }
            },
            "required": ["path", "code"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path     = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let code         = input.get("code").and_then(|v| v.as_str()).unwrap_or("");
        let check_syntax = input.get("check_syntax").and_then(|v| v.as_bool()).unwrap_or(true);
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        // Create parent directories if needed.
        if let Some(parent) = safe.parent() {
            if parent.starts_with(&base) {
                tokio::fs::create_dir_all(parent).await
                    .map_err(|e| AgentError::IoError(e))?;
            }
        }

        tokio::fs::write(&safe, code).await
            .map_err(|e| AgentError::IoError(e))?;

        if check_syntax {
            let output = tokio::process::Command::new(python_bin())
                .arg("-m").arg("py_compile").arg(&safe)
                .current_dir(&base)
                .output()
                .await
                .map_err(|e| AgentError::IoError(e))?;

            if !output.status.success() {
                let err = String::from_utf8_lossy(&output.stderr).to_string();
                return Ok(ToolResult {
                    data: format!("File written but has syntax errors:\n{}", err.trim()),
                    is_error: true,
                });
            }
        }

        Ok(ToolResult {
            data: format!("Written: {} ({} bytes, syntax OK)", safe.display(), code.len()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// RepoIngest tool
// ---------------------------------------------------------------------------

/// Default directories to skip when walking a repo.
const SKIP_DIRS: &[&str] = &[
    ".git", "target", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "out", "coverage", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "vendor", ".idea", ".vscode",
];

/// Extensions treated as human-readable source/config files.
const TEXT_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp",
    "cs", "rb", "php", "swift", "kt", "scala", "hs", "ml", "ex", "exs", "clj",
    "toml", "yaml", "yml", "json", "xml", "html", "htm", "css", "scss", "md",
    "txt", "sh", "bash", "zsh", "fish", "ps1", "bat", "makefile", "dockerfile",
    "gitignore", "env", "cfg", "ini", "conf",
];

/// Priority files to always read first (basename match).
const PRIORITY_FILES: &[&str] = &[
    "readme.md", "readme.txt", "readme",
    "cargo.toml", "package.json", "go.mod", "pyproject.toml", "setup.py",
    "setup.cfg", "requirements.txt", "pom.xml", "build.gradle",
    "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "main.rs", "lib.rs", "main.py", "index.js", "index.ts", "app.py",
    "app.js", "app.ts", "mod.rs",
];

fn is_text_file(path: &Path) -> bool {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    TEXT_EXTENSIONS.contains(&ext.as_str())
        || TEXT_EXTENSIONS.contains(&name.as_str())
        || name == "makefile"
        || name == "dockerfile"
}

fn priority_score(path: &Path) -> u8 {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    if PRIORITY_FILES.contains(&name.as_str()) { 2 }
    else if path.extension().and_then(|e| e.to_str()).unwrap_or("") == "md" { 1 }
    else { 0 }
}

/// Detect programming language from extension.
fn detect_lang(ext: &str) -> &'static str {
    match ext {
        "rs" => "Rust", "py" => "Python", "js" => "JavaScript",
        "ts" => "TypeScript", "jsx" | "tsx" => "React/JSX",
        "go" => "Go", "java" => "Java", "c" | "h" => "C",
        "cpp" | "hpp" | "cc" => "C++", "cs" => "C#",
        "rb" => "Ruby", "php" => "PHP", "swift" => "Swift",
        "kt" => "Kotlin", "scala" => "Scala", "hs" => "Haskell",
        "ex" | "exs" => "Elixir", "clj" => "Clojure",
        "sh" | "bash" => "Shell", "ps1" => "PowerShell",
        _ => "",
    }
}

/// Extract top-level declarations from source code (language-aware, regex-free).
fn extract_outline(content: &str, ext: &str) -> Vec<String> {
    let mut items = Vec::new();
    for line in content.lines().take(500) {
        let t = line.trim();
        let is_decl = match ext {
            "rs" => t.starts_with("pub fn ") || t.starts_with("fn ") ||
                    t.starts_with("pub struct ") || t.starts_with("struct ") ||
                    t.starts_with("pub enum ") || t.starts_with("enum ") ||
                    t.starts_with("pub trait ") || t.starts_with("trait ") ||
                    t.starts_with("pub mod ") || t.starts_with("mod ") ||
                    t.starts_with("impl ") || t.starts_with("pub impl ") ||
                    t.starts_with("pub type ") || t.starts_with("type "),
            "py" => t.starts_with("def ") || t.starts_with("class ") ||
                    t.starts_with("async def "),
            "js" | "ts" | "jsx" | "tsx" =>
                    t.starts_with("function ") || t.starts_with("class ") ||
                    t.starts_with("export function ") || t.starts_with("export class ") ||
                    t.starts_with("export default ") || t.starts_with("export const ") ||
                    t.starts_with("const ") && t.contains("=>"),
            "go" => t.starts_with("func ") || t.starts_with("type ") ||
                    t.starts_with("var ") || t.starts_with("const "),
            "java" | "cs" | "kt" | "scala" =>
                    (t.contains(" class ") || t.contains(" interface ") ||
                     t.contains(" enum ") || t.contains("fun ") ||
                     t.contains("void ") || t.contains("public ") ||
                     t.contains("private ")) && !t.starts_with("//") && !t.starts_with("*"),
            _ => false,
        };
        if is_decl {
            // Trim to a single line (no body braces).
            let decl = t.trim_end_matches('{').trim().to_string();
            if decl.len() <= 120 {
                items.push(decl);
            }
        }
    }
    items.truncate(30);
    items
}

/// Walk `dir` recursively, collecting (path, priority_score, size) for text files.
fn collect_files(dir: &Path, base: &Path, out: &mut Vec<(PathBuf, u8, u64)>) -> std::io::Result<()> {
    let rd = std::fs::read_dir(dir)?;
    for entry in rd.flatten() {
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !SKIP_DIRS.contains(&name) {
                collect_files(&path, base, out)?;
            }
        } else if meta.is_file() && is_text_file(&path) {
            let score = priority_score(&path);
            out.push((path, score, meta.len()));
        }
    }
    Ok(())
}

/// Build an indented directory tree string (max depth 5, skips known noise dirs).
fn build_tree(dir: &Path, base: &Path, depth: usize, out: &mut String) -> std::io::Result<()> {
    if depth > 5 { return Ok(()); }
    let indent = "  ".repeat(depth);
    let rd = std::fs::read_dir(dir)?;
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let meta = entry.metadata()?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if meta.is_dir() {
            if SKIP_DIRS.contains(&name_str.as_ref()) { continue; }
            out.push_str(&format!("{}📁 {}/\n", indent, name_str));
            build_tree(&path, base, depth + 1, out)?;
        } else {
            out.push_str(&format!("{}📄 {}\n", indent, name_str));
        }
    }
    Ok(())
}

/// Ingest a code repository: scan structure, read key files, extract outlines.
///
/// Returns a rich markdown document containing:
/// - Project overview and detected languages
/// - Directory tree
/// - Contents and code outlines of the most important files
/// - Suggested mindmap skeleton
///
/// The returned text is designed to be fed directly into an LLM prompt so the
/// agent can produce a summary or generate a Mermaid mindmap `.md` file.
pub struct RepoIngestTool;

#[async_trait]
impl Tool for RepoIngestTool {
    fn name(&self) -> &str { "repo_ingest" }

    fn description(&self) -> &str {
        "Analyze a code repository or directory. Reads its file structure, detects \
         programming languages, extracts code outlines (functions, classes, modules), \
         and returns a rich markdown document summarizing the entire codebase. \
         Use this before writing a summary or generating a mindmap. \
         Provide `output_file` to also save the analysis to a .md file."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to analyze (defaults to the working directory)."
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to read content from (default 50, max 200).",
                    "minimum": 1,
                    "maximum": 200
                },
                "max_file_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes to read per file (default 8000, max 40000).",
                    "minimum": 200,
                    "maximum": 40000
                },
                "output_file": {
                    "type": "string",
                    "description": "Optional: write the analysis report to this .md file path (within working directory)."
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let base   = sandbox_root(context);
        let raw    = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let max_files      = input.get("max_files").and_then(|v| v.as_u64()).unwrap_or(50).min(200) as usize;
        let max_file_bytes = input.get("max_file_bytes").and_then(|v| v.as_u64()).unwrap_or(8000).min(40_000) as usize;

        let root = match safe_path(raw, &base) {
            Ok(p)  => p,
            Err(e) => return Ok(e),
        };

        if !root.is_dir() {
            return Ok(ToolResult {
                data: format!("'{}' is not a directory", raw),
                is_error: true,
            });
        }

        // ── Phase 1: collect all text files ──────────────────────────────────
        let mut all_files: Vec<(PathBuf, u8, u64)> = Vec::new();
        collect_files(&root, &root, &mut all_files)
            .map_err(|e| AgentError::IoError(e))?;

        // Sort: priority desc, then by size asc (smaller files first for variety).
        all_files.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

        let total_files = all_files.len();

        // ── Phase 2: language statistics ─────────────────────────────────────
        let mut lang_counts: HashMap<&'static str, usize> = HashMap::new();
        for (path, _, _) in &all_files {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            let lang = detect_lang(&ext);
            if !lang.is_empty() {
                *lang_counts.entry(lang).or_insert(0) += 1;
            }
        }
        let mut langs: Vec<(&str, usize)> = lang_counts.into_iter().collect();
        langs.sort_by(|a, b| b.1.cmp(&a.1));

        // ── Phase 3: directory tree (sync, brief) ─────────────────────────────
        let mut tree = String::new();
        build_tree(&root, &root, 0, &mut tree)
            .map_err(|e| AgentError::IoError(e))?;

        // ── Phase 4: read top N files and extract outlines ────────────────────
        let files_to_read: Vec<_> = all_files.iter().take(max_files).collect();
        let mut file_sections: Vec<String> = Vec::new();

        for (path, _, size) in &files_to_read {
            let rel = path.strip_prefix(&root).unwrap_or(path).display().to_string();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

            // Read up to max_file_bytes.
            let raw_bytes = tokio::fs::read(path).await
                .unwrap_or_default();
            let content = if raw_bytes.len() > max_file_bytes {
                let truncated = &raw_bytes[..max_file_bytes];
                // Try to cut at a newline boundary.
                let cut = truncated.iter().rposition(|&b| b == b'\n').unwrap_or(max_file_bytes);
                format!("{}\n… [truncated, {} bytes total]",
                    String::from_utf8_lossy(&raw_bytes[..cut]),
                    size)
            } else {
                String::from_utf8_lossy(&raw_bytes).to_string()
            };

            let outline = extract_outline(&content, &ext);

            let mut section = format!("\n### `{}`\n", rel);
            if !outline.is_empty() {
                section.push_str("**Declarations:**\n");
                for item in &outline {
                    section.push_str(&format!("- `{}`\n", item));
                }
                section.push('\n');
            }
            section.push_str(&format!("```{}\n{}\n```\n", ext, content.trim()));
            file_sections.push(section);
        }

        // ── Phase 5: compose the report ───────────────────────────────────────
        let dir_name = root.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("project");

        let lang_summary = if langs.is_empty() {
            "No source files detected.".to_string()
        } else {
            langs.iter().map(|(l, n)| format!("{} ({})", l, n)).collect::<Vec<_>>().join(", ")
        };

        let report = format!(
r#"# Repository Analysis: `{dir_name}`

## Overview
- **Path:** `{path}`
- **Total text files found:** {total_files}
- **Files analyzed (content read):** {analyzed}
- **Languages:** {lang_summary}

## Directory Tree
```
{tree}```

## Key Files & Code Outlines
{files}

---
*Generated by repo_ingest. Use this analysis to produce a summary or Mermaid mindmap.*
"#,
            dir_name   = dir_name,
            path       = root.display(),
            total_files = total_files,
            analyzed   = files_to_read.len(),
            lang_summary = lang_summary,
            tree       = tree.trim_end(),
            files      = file_sections.join(""),
        );

        // ── Phase 6: optionally write to file ─────────────────────────────────
        if let Some(out_raw) = input.get("output_file").and_then(|v| v.as_str()) {
            if !out_raw.is_empty() {
                match safe_path(out_raw, &base) {
                    Ok(out_path) => {
                        if let Some(parent) = out_path.parent() {
                            if parent.starts_with(&base) {
                                let _ = tokio::fs::create_dir_all(parent).await;
                            }
                        }
                        let _ = tokio::fs::write(&out_path, &report).await;
                    }
                    Err(_) => {}
                }
            }
        }

        Ok(ToolResult { data: report, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// Register all built-in tools
// ---------------------------------------------------------------------------

pub async fn register_built_in_tools(registry: &mut ToolRegistry) {
    let tools: Vec<Arc<dyn Tool>> = vec![
        // Shell
        Arc::new(BashTool),
        // File operations
        Arc::new(FileReadTool),
        Arc::new(FileWriteTool),
        Arc::new(FileUpdateTool),
        Arc::new(FileDeleteTool),
        Arc::new(FileListTool),
        Arc::new(FileMoveTool),
        // Directory operations
        Arc::new(DirCreateTool),
        Arc::new(DirDeleteTool),
        // Search
        Arc::new(GrepTool),
        // Python coding
        Arc::new(PythonWriteTool),
        Arc::new(PythonRunTool),
        Arc::new(PythonTestTool),
        // Codebase analysis
        Arc::new(RepoIngestTool),
    ];
    for tool in tools {
        let _ = registry.register(tool);
    }
}
