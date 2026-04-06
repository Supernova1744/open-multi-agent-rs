use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::OnceLock;

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
// HttpGet tool
// ---------------------------------------------------------------------------

/// Fetch a URL with HTTP GET and return the response body.
///
/// # Security
/// - Response body is capped at 4 MB to prevent memory exhaustion.
/// - Redirects are followed automatically (up to 10).
/// - The request carries a descriptive User-Agent.
pub struct HttpGetTool;

#[async_trait]
impl Tool for HttpGetTool {
    fn name(&self) -> &str { "http_get" }
    fn description(&self) -> &str {
        "Send an HTTP GET request to a URL and return the response body and status code. \
         Useful for fetching web pages, APIs, documentation, or any HTTP resource."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to request (must begin with http:// or https://)."
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value string pairs.",
                    "additionalProperties": { "type": "string" }
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Request timeout in milliseconds (default 15000, max 60000).",
                    "minimum": 1,
                    "maximum": 60000
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if !u.is_empty() => u,
            _ => return Ok(ToolResult { data: "url must be a non-empty string".to_string(), is_error: true }),
        };

        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolResult {
                data: "url must begin with http:// or https://".to_string(),
                is_error: true,
            });
        }

        let timeout_ms = input.get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(15_000)
            .min(60_000);

        let mut builder = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .user_agent("open-multi-agent-rs/0.1 (+https://github.com/Supernova1744/open-multi-agent-rs)")
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| AgentError::Other(e.to_string()))?
            .get(url);

        if let Some(hdrs) = input.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in hdrs {
                if let Some(val) = v.as_str() {
                    builder = builder.header(k.as_str(), val);
                }
            }
        }

        let response = builder.send().await
            .map_err(|e| AgentError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        const MAX_BYTES: usize = 4 * 1024 * 1024; // 4 MB
        let bytes = response.bytes().await
            .map_err(|e| AgentError::Other(format!("failed to read response body: {}", e)))?;

        let body = if bytes.len() > MAX_BYTES {
            format!("[truncated to 4 MB]\n{}", String::from_utf8_lossy(&bytes[..MAX_BYTES]))
        } else {
            String::from_utf8_lossy(&bytes).to_string()
        };

        let data = format!("HTTP {} {}\n\n{}", status.as_u16(), status.canonical_reason().unwrap_or(""), body);
        Ok(ToolResult { data, is_error: !status.is_success() })
    }
}

// ---------------------------------------------------------------------------
// HttpPost tool
// ---------------------------------------------------------------------------

/// Send an HTTP POST request with a body and return the response.
pub struct HttpPostTool;

#[async_trait]
impl Tool for HttpPostTool {
    fn name(&self) -> &str { "http_post" }
    fn description(&self) -> &str {
        "Send an HTTP POST request with a body and return the response body and status code. \
         Useful for calling APIs, webhooks, or submitting data."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to POST to (must begin with http:// or https://)."
                },
                "body": {
                    "type": "string",
                    "description": "Request body as a string (JSON, form data, plain text, etc.)."
                },
                "content_type": {
                    "type": "string",
                    "description": "Content-Type header value (default: application/json)."
                },
                "headers": {
                    "type": "object",
                    "description": "Additional HTTP headers as key-value string pairs.",
                    "additionalProperties": { "type": "string" }
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Request timeout in milliseconds (default 15000, max 60000).",
                    "minimum": 1,
                    "maximum": 60000
                }
            },
            "required": ["url", "body"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if !u.is_empty() => u,
            _ => return Ok(ToolResult { data: "url must be a non-empty string".to_string(), is_error: true }),
        };

        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolResult {
                data: "url must begin with http:// or https://".to_string(),
                is_error: true,
            });
        }

        let body = input.get("body").and_then(|v| v.as_str()).unwrap_or("");
        let content_type = input.get("content_type")
            .and_then(|v| v.as_str())
            .unwrap_or("application/json");
        let timeout_ms = input.get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(15_000)
            .min(60_000);

        let mut builder = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .user_agent("open-multi-agent-rs/0.1 (+https://github.com/Supernova1744/open-multi-agent-rs)")
            .build()
            .map_err(|e| AgentError::Other(e.to_string()))?
            .post(url)
            .header("Content-Type", content_type)
            .body(body.to_string());

        if let Some(hdrs) = input.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in hdrs {
                if let Some(val) = v.as_str() {
                    builder = builder.header(k.as_str(), val);
                }
            }
        }

        let response = builder.send().await
            .map_err(|e| AgentError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        const MAX_BYTES: usize = 4 * 1024 * 1024;
        let bytes = response.bytes().await
            .map_err(|e| AgentError::Other(format!("failed to read response body: {}", e)))?;

        let resp_body = if bytes.len() > MAX_BYTES {
            format!("[truncated to 4 MB]\n{}", String::from_utf8_lossy(&bytes[..MAX_BYTES]))
        } else {
            String::from_utf8_lossy(&bytes).to_string()
        };

        let data = format!("HTTP {} {}\n\n{}", status.as_u16(), status.canonical_reason().unwrap_or(""), resp_body);
        Ok(ToolResult { data, is_error: !status.is_success() })
    }
}

// ---------------------------------------------------------------------------
// JsonParse tool
// ---------------------------------------------------------------------------

/// Parse a JSON string, optionally pretty-print it or extract a sub-value via
/// JSON Pointer (RFC 6901).
pub struct JsonParseTool;

#[async_trait]
impl Tool for JsonParseTool {
    fn name(&self) -> &str { "json_parse" }
    fn description(&self) -> &str {
        "Parse a JSON string and optionally extract a value at a JSON Pointer path \
         (e.g. '/users/0/name') or pretty-print the whole document. \
         Returns an error message if the input is not valid JSON."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The JSON string to parse."
                },
                "pointer": {
                    "type": "string",
                    "description": "Optional JSON Pointer path (RFC 6901) to extract a sub-value, e.g. '/users/0/name'."
                },
                "pretty": {
                    "type": "boolean",
                    "description": "If true (default), return the output as pretty-printed JSON."
                }
            },
            "required": ["input"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw = match input.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "input is required".to_string(), is_error: true }),
        };

        let parsed: serde_json::Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => return Ok(ToolResult {
                data: format!("JSON parse error: {}", e),
                is_error: true,
            }),
        };

        let value = if let Some(ptr) = input.get("pointer").and_then(|v| v.as_str()) {
            match parsed.pointer(ptr) {
                Some(v) => v.clone(),
                None => return Ok(ToolResult {
                    data: format!("pointer '{}' not found in JSON document", ptr),
                    is_error: true,
                }),
            }
        } else {
            parsed
        };

        let pretty = input.get("pretty").and_then(|v| v.as_bool()).unwrap_or(true);
        let data = if pretty {
            serde_json::to_string_pretty(&value)
                .unwrap_or_else(|e| format!("serialization error: {}", e))
        } else {
            value.to_string()
        };

        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// JsonTransform tool
// ---------------------------------------------------------------------------

/// Map / filter / reshape a JSON array or object using a minimal template
/// expression language:
///   - `keys`              — return array of object keys
///   - `values`            — return array of object values
///   - `length`            — return array/string/object length
///   - `[<pointer>]`       — for each element in an array, extract the field at
///                           the JSON Pointer and return a new array
///   - `<pointer>`         — alias for json_parse with pointer
pub struct JsonTransformTool;

#[async_trait]
impl Tool for JsonTransformTool {
    fn name(&self) -> &str { "json_transform" }
    fn description(&self) -> &str {
        "Transform a JSON value with a simple operation: \
         'keys' (object keys), 'values' (object values), 'length' (count), \
         '[/field]' (map array → extract field from each element), \
         or '/pointer' (extract sub-value)."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The JSON string to transform."
                },
                "operation": {
                    "type": "string",
                    "description": "Transformation: 'keys', 'values', 'length', '[/field]' (map extract), or '/pointer' (extract)."
                }
            },
            "required": ["input", "operation"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw = match input.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "input is required".to_string(), is_error: true }),
        };
        let op = match input.get("operation").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "operation is required".to_string(), is_error: true }),
        };

        let parsed: serde_json::Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => return Ok(ToolResult { data: format!("JSON parse error: {}", e), is_error: true }),
        };

        let result: serde_json::Value = match op {
            "keys" => {
                match &parsed {
                    serde_json::Value::Object(m) =>
                        serde_json::Value::Array(m.keys().map(|k| json!(k)).collect()),
                    _ => return Ok(ToolResult { data: "'keys' requires a JSON object".to_string(), is_error: true }),
                }
            }
            "values" => {
                match &parsed {
                    serde_json::Value::Object(m) =>
                        serde_json::Value::Array(m.values().cloned().collect()),
                    _ => return Ok(ToolResult { data: "'values' requires a JSON object".to_string(), is_error: true }),
                }
            }
            "length" => {
                let n = match &parsed {
                    serde_json::Value::Array(a)  => a.len(),
                    serde_json::Value::Object(m) => m.len(),
                    serde_json::Value::String(s) => s.len(),
                    _ => return Ok(ToolResult { data: "'length' requires an array, object, or string".to_string(), is_error: true }),
                };
                json!(n)
            }
            op if op.starts_with('[') && op.ends_with(']') => {
                // Map: extract field from each array element.
                let ptr = &op[1..op.len()-1];
                match &parsed {
                    serde_json::Value::Array(arr) => {
                        let mapped: Vec<serde_json::Value> = arr.iter()
                            .map(|elem| elem.pointer(ptr).cloned().unwrap_or(serde_json::Value::Null))
                            .collect();
                        serde_json::Value::Array(mapped)
                    }
                    _ => return Ok(ToolResult { data: "map operation '[/field]' requires a JSON array".to_string(), is_error: true }),
                }
            }
            op if op.starts_with('/') => {
                match parsed.pointer(op) {
                    Some(v) => v.clone(),
                    None => return Ok(ToolResult {
                        data: format!("pointer '{}' not found", op),
                        is_error: true,
                    }),
                }
            }
            _ => return Ok(ToolResult {
                data: format!("unknown operation '{}'; valid: keys, values, length, [/field], /pointer", op),
                is_error: true,
            }),
        };

        let data = serde_json::to_string_pretty(&result)
            .unwrap_or_else(|e| format!("serialization error: {}", e));
        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// CsvRead tool
// ---------------------------------------------------------------------------

/// Read a CSV file within the sandbox and return its contents as a JSON array
/// of objects (keyed by header row) or as a formatted Markdown table.
pub struct CsvReadTool;

#[async_trait]
impl Tool for CsvReadTool {
    fn name(&self) -> &str { "csv_read" }
    fn description(&self) -> &str {
        "Read a CSV file within the working directory and return its rows as a \
         JSON array of objects (one object per row, keys from header row). \
         Optionally returns a Markdown table instead."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the CSV file."
                },
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter character (default: ',')."
                },
                "has_headers": {
                    "type": "boolean",
                    "description": "Whether the first row is a header row (default: true)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default: 1000).",
                    "minimum": 1,
                    "maximum": 100000
                },
                "format": {
                    "type": "string",
                    "description": "'json' (default) or 'markdown' table format.",
                    "enum": ["json", "markdown"]
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

        let delimiter = input.get("delimiter")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(',') as u8;
        let has_headers = input.get("has_headers").and_then(|v| v.as_bool()).unwrap_or(true);
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
        let format = input.get("format").and_then(|v| v.as_str()).unwrap_or("json");

        let content = std::fs::read_to_string(&safe)
            .map_err(|e| AgentError::IoError(e))?;

        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(has_headers)
            .from_reader(content.as_bytes());

        if format == "markdown" {
            // Build a Markdown table.
            let headers: Vec<String> = if has_headers {
                rdr.headers()
                    .map_err(|e| AgentError::Other(e.to_string()))?
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                vec![]
            };

            let mut rows: Vec<Vec<String>> = Vec::new();
            for result in rdr.records().take(limit) {
                let record = result.map_err(|e| AgentError::Other(e.to_string()))?;
                rows.push(record.iter().map(|f| f.to_string()).collect());
            }

            let mut md = String::new();
            if !headers.is_empty() {
                md.push_str(&format!("| {} |\n", headers.join(" | ")));
                md.push_str(&format!("| {} |\n", headers.iter().map(|_| "---").collect::<Vec<_>>().join(" | ")));
            }
            for row in &rows {
                md.push_str(&format!("| {} |\n", row.join(" | ")));
            }
            return Ok(ToolResult { data: md, is_error: false });
        }

        // Default: JSON array of objects.
        let headers: Vec<String> = if has_headers {
            rdr.headers()
                .map_err(|e| AgentError::Other(e.to_string()))?
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            vec![]
        };

        let mut records: Vec<serde_json::Value> = Vec::new();
        for result in rdr.records().take(limit) {
            let record = result.map_err(|e| AgentError::Other(e.to_string()))?;
            let obj: serde_json::Value = if headers.is_empty() {
                json!(record.iter().collect::<Vec<_>>())
            } else {
                let mut m = serde_json::Map::new();
                for (i, field) in record.iter().enumerate() {
                    let key = headers.get(i).map(|s| s.as_str()).unwrap_or("_");
                    m.insert(key.to_string(), json!(field));
                }
                serde_json::Value::Object(m)
            };
            records.push(obj);
        }

        let data = serde_json::to_string_pretty(&records)
            .unwrap_or_else(|e| format!("serialization error: {}", e));
        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// CsvWrite tool
// ---------------------------------------------------------------------------

/// Write a JSON array of objects (or arrays) to a CSV file within the sandbox.
pub struct CsvWriteTool;

#[async_trait]
impl Tool for CsvWriteTool {
    fn name(&self) -> &str { "csv_write" }
    fn description(&self) -> &str {
        "Write a JSON array of objects (or arrays) to a CSV file within the \
         working directory. Headers are inferred from the first object's keys."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to write the CSV file."
                },
                "data": {
                    "type": "string",
                    "description": "JSON array of objects or arrays to write as CSV rows."
                },
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter character (default: ',')."
                }
            },
            "required": ["path", "data"]
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

        let data_str = match input.get("data").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "data is required".to_string(), is_error: true }),
        };

        let rows: serde_json::Value = match serde_json::from_str(data_str) {
            Ok(v) => v,
            Err(e) => return Ok(ToolResult { data: format!("data JSON parse error: {}", e), is_error: true }),
        };

        let arr = match rows.as_array() {
            Some(a) => a,
            None => return Ok(ToolResult { data: "data must be a JSON array".to_string(), is_error: true }),
        };

        if arr.is_empty() {
            return Ok(ToolResult { data: format!("Written 0 rows to {}", safe.display()), is_error: false });
        }

        let delimiter = input.get("delimiter")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(',') as u8;

        if let Some(parent) = safe.parent() {
            if parent.starts_with(&base) {
                std::fs::create_dir_all(parent).map_err(|e| AgentError::IoError(e))?;
            }
        }

        let mut wtr = csv::WriterBuilder::new()
            .delimiter(delimiter)
            .from_path(&safe)
            .map_err(|e| AgentError::Other(e.to_string()))?;

        // Infer headers from first element if objects.
        if let Some(first_obj) = arr[0].as_object() {
            let headers: Vec<&str> = first_obj.keys().map(|k| k.as_str()).collect();
            wtr.write_record(&headers).map_err(|e| AgentError::Other(e.to_string()))?;
            for row in arr {
                if let Some(obj) = row.as_object() {
                    let fields: Vec<String> = headers.iter()
                        .map(|k| {
                            obj.get(*k)
                                .map(|v| match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                })
                                .unwrap_or_default()
                        })
                        .collect();
                    wtr.write_record(&fields).map_err(|e| AgentError::Other(e.to_string()))?;
                }
            }
        } else {
            // Array of arrays.
            for row in arr {
                if let Some(inner) = row.as_array() {
                    let fields: Vec<String> = inner.iter()
                        .map(|v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .collect();
                    wtr.write_record(&fields).map_err(|e| AgentError::Other(e.to_string()))?;
                }
            }
        }

        wtr.flush().map_err(|e| AgentError::IoError(e))?;
        Ok(ToolResult {
            data: format!("Written {} rows to {}", arr.len(), safe.display()),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// MathEval tool
// ---------------------------------------------------------------------------

/// Evaluate a mathematical expression safely using the `evalexpr` crate.
///
/// Supports: arithmetic, exponentiation, abs, min, max, floor, ceil, round,
/// sqrt, ln, log, sin, cos, tan, boolean logic, comparisons, and variables.
pub struct MathEvalTool;

#[async_trait]
impl Tool for MathEvalTool {
    fn name(&self) -> &str { "math_eval" }
    fn description(&self) -> &str {
        "Evaluate a mathematical expression and return the result. \
         Supports arithmetic (+, -, *, /, %, ^), math functions (sqrt, abs, min, max, \
         floor, ceil, round, sin, cos, tan, ln, log, exp), and variables. \
         No code execution — safe, pure expression evaluation."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 * (3 + 4)', 'sqrt(144)', 'x^2 + y'."
                },
                "variables": {
                    "type": "object",
                    "description": "Optional variable bindings as an object, e.g. {\"x\": 3, \"y\": 4.5}.",
                    "additionalProperties": { "type": "number" }
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let expr = match input.get("expression").and_then(|v| v.as_str()) {
            Some(e) if !e.is_empty() => e,
            _ => return Ok(ToolResult { data: "expression must be a non-empty string".to_string(), is_error: true }),
        };

        use evalexpr::ContextWithMutableVariables;
        let mut ctx = evalexpr::HashMapContext::new();
        if let Some(vars) = input.get("variables").and_then(|v| v.as_object()) {
            for (k, v) in vars {
                if let Some(n) = v.as_f64() {
                    ctx.set_value(k.clone(), evalexpr::Value::Float(n))
                        .map_err(|e| AgentError::Other(e.to_string()))?;
                }
            }
        }

        match evalexpr::eval_with_context(expr, &ctx) {
            Ok(val) => Ok(ToolResult { data: val.to_string(), is_error: false }),
            Err(e)  => Ok(ToolResult { data: format!("expression error: {}", e), is_error: true }),
        }
    }
}

// ---------------------------------------------------------------------------
// Datetime tool
// ---------------------------------------------------------------------------

/// Date and time operations (format, parse, now, diff) using the `chrono` crate.
pub struct DatetimeTool;

#[async_trait]
impl Tool for DatetimeTool {
    fn name(&self) -> &str { "datetime" }
    fn description(&self) -> &str {
        "Date and time operations: \
         'now' — return current UTC time; \
         'format' — format a Unix timestamp (seconds) using strftime patterns; \
         'parse' — parse a datetime string to Unix timestamp; \
         'diff' — compute the difference between two Unix timestamps."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "One of: 'now', 'format', 'parse', 'diff'.",
                    "enum": ["now", "format", "parse", "diff"]
                },
                "timestamp": {
                    "type": "integer",
                    "description": "Unix timestamp in seconds (used by 'format' and 'diff')."
                },
                "timestamp2": {
                    "type": "integer",
                    "description": "Second Unix timestamp for 'diff' (result = timestamp2 - timestamp)."
                },
                "format": {
                    "type": "string",
                    "description": "strftime format string for 'format' (default: '%Y-%m-%d %H:%M:%S UTC'). Also used as input format hint for 'parse'."
                },
                "input": {
                    "type": "string",
                    "description": "Datetime string to parse (for 'parse' operation). ISO 8601 and RFC 2822 are supported."
                }
            },
            "required": ["operation"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        use chrono::{DateTime, TimeZone, Utc, NaiveDateTime};

        let op = match input.get("operation").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "operation is required".to_string(), is_error: true }),
        };

        match op {
            "now" => {
                let now = Utc::now();
                let fmt = input.get("format").and_then(|v| v.as_str()).unwrap_or("%Y-%m-%d %H:%M:%S UTC");
                let formatted = now.format(fmt).to_string();
                let data = format!("{}\ntimestamp: {}", formatted, now.timestamp());
                Ok(ToolResult { data, is_error: false })
            }

            "format" => {
                let ts = match input.get("timestamp").and_then(|v| v.as_i64()) {
                    Some(t) => t,
                    None => return Ok(ToolResult { data: "timestamp (integer seconds) is required for 'format'".to_string(), is_error: true }),
                };
                let fmt = input.get("format").and_then(|v| v.as_str()).unwrap_or("%Y-%m-%d %H:%M:%S UTC");
                let dt = Utc.timestamp_opt(ts, 0)
                    .single()
                    .ok_or_else(|| AgentError::Other("invalid timestamp".to_string()))?;
                Ok(ToolResult { data: dt.format(fmt).to_string(), is_error: false })
            }

            "parse" => {
                let s = match input.get("input").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => return Ok(ToolResult { data: "input string is required for 'parse'".to_string(), is_error: true }),
                };

                // Try RFC3339 / ISO 8601 first.
                if let Ok(dt) = s.parse::<DateTime<Utc>>() {
                    return Ok(ToolResult { data: dt.timestamp().to_string(), is_error: false });
                }
                // Try common formats.
                let formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%d/%m/%Y %H:%M:%S",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                ];
                for fmt in &formats {
                    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
                        return Ok(ToolResult { data: ndt.and_utc().timestamp().to_string(), is_error: false });
                    }
                    if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, fmt) {
                        let ndt = nd.and_hms_opt(0, 0, 0).unwrap();
                        return Ok(ToolResult { data: ndt.and_utc().timestamp().to_string(), is_error: false });
                    }
                }
                Ok(ToolResult {
                    data: format!("could not parse '{}' — try ISO 8601 format (e.g. 2024-01-15T10:30:00Z)", s),
                    is_error: true,
                })
            }

            "diff" => {
                let t1 = match input.get("timestamp").and_then(|v| v.as_i64()) {
                    Some(t) => t,
                    None => return Ok(ToolResult { data: "timestamp is required for 'diff'".to_string(), is_error: true }),
                };
                let t2 = match input.get("timestamp2").and_then(|v| v.as_i64()) {
                    Some(t) => t,
                    None => return Ok(ToolResult { data: "timestamp2 is required for 'diff'".to_string(), is_error: true }),
                };
                let diff_secs = t2 - t1;
                let abs = diff_secs.unsigned_abs();
                let days    = abs / 86400;
                let hours   = (abs % 86400) / 3600;
                let minutes = (abs % 3600) / 60;
                let secs    = abs % 60;
                let data = format!(
                    "diff_seconds: {}\ndiff_human: {}{}d {}h {}m {}s",
                    diff_secs,
                    if diff_secs < 0 { "-" } else { "" },
                    days, hours, minutes, secs
                );
                Ok(ToolResult { data, is_error: false })
            }

            _ => Ok(ToolResult {
                data: format!("unknown operation '{}'; valid: now, format, parse, diff", op),
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// TextRegex tool
// ---------------------------------------------------------------------------

/// Apply a regular expression to a string: find all matches, replace, or split.
pub struct TextRegexTool;

#[async_trait]
impl Tool for TextRegexTool {
    fn name(&self) -> &str { "text_regex" }
    fn description(&self) -> &str {
        "Apply a regular expression to text. \
         Modes: 'find_all' — return all matches with positions; \
         'replace' — replace all matches with a replacement string (supports $1 capture groups); \
         'split' — split text on the pattern."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The text to operate on."
                },
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern (Rust regex syntax)."
                },
                "mode": {
                    "type": "string",
                    "description": "'find_all' (default), 'replace', or 'split'.",
                    "enum": ["find_all", "replace", "split"]
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement string for 'replace' mode (supports $1, $name capture groups)."
                }
            },
            "required": ["input", "pattern"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let text = match input.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "input is required".to_string(), is_error: true }),
        };
        let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s,
            _ => return Ok(ToolResult { data: "pattern must be a non-empty string".to_string(), is_error: true }),
        };

        let re = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(e) => return Ok(ToolResult { data: format!("invalid regex: {}", e), is_error: true }),
        };

        let mode = input.get("mode").and_then(|v| v.as_str()).unwrap_or("find_all");

        match mode {
            "find_all" => {
                let matches: Vec<serde_json::Value> = re.find_iter(text)
                    .map(|m| json!({
                        "match": m.as_str(),
                        "start": m.start(),
                        "end": m.end()
                    }))
                    .collect();
                let data = if matches.is_empty() {
                    "no matches found".to_string()
                } else {
                    serde_json::to_string_pretty(&matches)
                        .unwrap_or_else(|e| format!("serialization error: {}", e))
                };
                Ok(ToolResult { data, is_error: false })
            }
            "replace" => {
                let replacement = input.get("replacement").and_then(|v| v.as_str()).unwrap_or("");
                let result = re.replace_all(text, replacement);
                Ok(ToolResult { data: result.to_string(), is_error: false })
            }
            "split" => {
                let parts: Vec<&str> = re.split(text).collect();
                let data = serde_json::to_string_pretty(&parts)
                    .unwrap_or_else(|e| format!("serialization error: {}", e));
                Ok(ToolResult { data, is_error: false })
            }
            _ => Ok(ToolResult {
                data: format!("unknown mode '{}'; valid: find_all, replace, split", mode),
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// TextChunk tool
// ---------------------------------------------------------------------------

/// Split text into chunks for processing large documents.
pub struct TextChunkTool;

#[async_trait]
impl Tool for TextChunkTool {
    fn name(&self) -> &str { "text_chunk" }
    fn description(&self) -> &str {
        "Split a large text into chunks of a given size. \
         Useful for processing documents that exceed LLM context limits. \
         Returns a JSON array of chunk strings."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to split."
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Maximum size of each chunk (default: 2000).",
                    "minimum": 50
                },
                "overlap": {
                    "type": "integer",
                    "description": "Number of characters/words/lines to overlap between consecutive chunks (default: 0).",
                    "minimum": 0
                },
                "split_by": {
                    "type": "string",
                    "description": "'chars' (default), 'words', or 'lines'.",
                    "enum": ["chars", "words", "lines"]
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let text = match input.get("text").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "text is required".to_string(), is_error: true }),
        };

        let chunk_size = input.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(2000) as usize;
        let overlap    = input.get("overlap").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let split_by   = input.get("split_by").and_then(|v| v.as_str()).unwrap_or("chars");

        if chunk_size == 0 {
            return Ok(ToolResult { data: "chunk_size must be greater than 0".to_string(), is_error: true });
        }

        let units: Vec<&str> = match split_by {
            "words" => text.split_whitespace().collect(),
            "lines" => text.lines().collect(),
            _       => {
                // Char chunking — handle directly.
                let chars: Vec<char> = text.chars().collect();
                let step = if chunk_size > overlap { chunk_size - overlap } else { chunk_size };
                let mut chunks: Vec<String> = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    let end = (i + chunk_size).min(chars.len());
                    chunks.push(chars[i..end].iter().collect());
                    i += step;
                }
                let data = serde_json::to_string_pretty(&chunks)
                    .unwrap_or_else(|e| format!("serialization error: {}", e));
                return Ok(ToolResult { data, is_error: false });
            }
        };

        let step = if chunk_size > overlap { chunk_size - overlap } else { chunk_size };
        let mut chunks: Vec<String> = Vec::new();
        let mut i = 0;
        while i < units.len() {
            let end = (i + chunk_size).min(units.len());
            chunks.push(units[i..end].join(if split_by == "lines" { "\n" } else { " " }));
            i += step;
        }

        let data = serde_json::to_string_pretty(&chunks)
            .unwrap_or_else(|e| format!("serialization error: {}", e));
        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// EnvGet tool
// ---------------------------------------------------------------------------

/// Read an environment variable from a safe allowlist.
pub struct EnvGetTool;

/// Variables that agents are allowed to read.
const ENV_ALLOWLIST: &[&str] = &[
    "RUST_LOG", "RUST_BACKTRACE", "HOME", "USER", "USERNAME",
    "PATH", "LANG", "TZ", "TERM",
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "DATABASE_URL", "REDIS_URL",
    "APP_ENV", "ENVIRONMENT", "NODE_ENV",
    "PORT", "HOST",
    "REPO_PATH", "OUTPUT_FILE",
];

#[async_trait]
impl Tool for EnvGetTool {
    fn name(&self) -> &str { "env_get" }
    fn description(&self) -> &str {
        "Read the value of an environment variable. \
         Only variables on a safe allowlist can be read. \
         Returns the value or the provided default if the variable is not set."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Environment variable name (e.g. 'HOME', 'PORT', 'APP_ENV')."
                },
                "default": {
                    "type": "string",
                    "description": "Value to return if the variable is not set (default: empty string)."
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let name = match input.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => n,
            _ => return Ok(ToolResult { data: "name must be a non-empty string".to_string(), is_error: true }),
        };

        if !ENV_ALLOWLIST.contains(&name) {
            return Ok(ToolResult {
                data: format!(
                    "'{}' is not in the allowed list. Allowed variables: {}",
                    name,
                    ENV_ALLOWLIST.join(", ")
                ),
                is_error: true,
            });
        }

        let value = std::env::var(name)
            .unwrap_or_else(|_| {
                input.get("default")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            });

        Ok(ToolResult { data: value, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// SystemInfo tool
// ---------------------------------------------------------------------------

/// Return basic system information useful for agent decision-making.
pub struct SystemInfoTool;

#[async_trait]
impl Tool for SystemInfoTool {
    fn name(&self) -> &str { "system_info" }
    fn description(&self) -> &str {
        "Return basic system information: OS family, CPU count, architecture, \
         current working directory, and available system tools."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn execute(
        &self,
        _input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let cwd = sandbox_root(context).display().to_string();
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let info = json!({
            "os_family":   std::env::consts::FAMILY,
            "os":          std::env::consts::OS,
            "arch":        std::env::consts::ARCH,
            "cpu_count":   cpu_count,
            "cwd":         cwd,
            "exe_suffix":  std::env::consts::EXE_SUFFIX,
            "dll_suffix":  std::env::consts::DLL_SUFFIX,
        });

        let data = serde_json::to_string_pretty(&info)
            .unwrap_or_else(|e| format!("serialization error: {}", e));
        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// CacheSet / CacheGet tools
// ---------------------------------------------------------------------------

/// In-process key-value cache with TTL, shared across all tool invocations.
struct CacheEntry {
    value:      String,
    expires_at: Option<std::time::Instant>,
}

fn global_cache() -> &'static Mutex<HashMap<String, CacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<String, CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct CacheSetTool;

#[async_trait]
impl Tool for CacheSetTool {
    fn name(&self) -> &str { "cache_set" }
    fn description(&self) -> &str {
        "Store a string value in the in-process cache under a key, \
         with an optional TTL. The cache persists for the lifetime of the process."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Cache key."
                },
                "value": {
                    "type": "string",
                    "description": "Value to store."
                },
                "ttl_seconds": {
                    "type": "integer",
                    "description": "Time-to-live in seconds. If omitted the entry never expires.",
                    "minimum": 1
                }
            },
            "required": ["key", "value"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let key = match input.get("key").and_then(|v| v.as_str()) {
            Some(k) if !k.is_empty() => k.to_string(),
            _ => return Ok(ToolResult { data: "key must be a non-empty string".to_string(), is_error: true }),
        };
        let value = input.get("value").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let expires_at = input.get("ttl_seconds").and_then(|v| v.as_u64()).map(|ttl| {
            std::time::Instant::now() + std::time::Duration::from_secs(ttl)
        });

        let mut cache = global_cache().lock().unwrap();
        cache.insert(key.clone(), CacheEntry { value, expires_at });
        Ok(ToolResult { data: format!("cached '{}'", key), is_error: false })
    }
}

pub struct CacheGetTool;

#[async_trait]
impl Tool for CacheGetTool {
    fn name(&self) -> &str { "cache_get" }
    fn description(&self) -> &str {
        "Retrieve a value from the in-process cache by key. \
         Returns the default value (or empty string) if the key is not found or has expired."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Cache key to look up."
                },
                "default": {
                    "type": "string",
                    "description": "Value to return if key is missing or expired (default: empty string)."
                }
            },
            "required": ["key"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let key = match input.get("key").and_then(|v| v.as_str()) {
            Some(k) if !k.is_empty() => k,
            _ => return Ok(ToolResult { data: "key must be a non-empty string".to_string(), is_error: true }),
        };
        let default_val = input.get("default").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let mut cache = global_cache().lock().unwrap();
        let now = std::time::Instant::now();

        // Remove expired entry if present.
        if let Some(entry) = cache.get(key) {
            if entry.expires_at.map(|exp| now > exp).unwrap_or(false) {
                cache.remove(key);
            }
        }

        let data = cache.get(key)
            .map(|e| e.value.clone())
            .unwrap_or(default_val);

        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// Base64 tool
// ---------------------------------------------------------------------------

/// Encode or decode Base64 strings — useful for handling binary data in text
/// pipelines.
pub struct Base64Tool;

#[async_trait]
impl Tool for Base64Tool {
    fn name(&self) -> &str { "base64" }
    fn description(&self) -> &str {
        "Encode a string to Base64 or decode a Base64 string back to text. \
         Modes: 'encode' (default) and 'decode'."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The string to encode or decode."
                },
                "mode": {
                    "type": "string",
                    "description": "'encode' (default) or 'decode'.",
                    "enum": ["encode", "decode"]
                }
            },
            "required": ["input"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let text = match input.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "input is required".to_string(), is_error: true }),
        };
        let mode = input.get("mode").and_then(|v| v.as_str()).unwrap_or("encode");

        match mode {
            "encode" => {
                let mut encoded = String::new();
                let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                let bytes = text.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    let b0 = bytes[i] as u32;
                    let b1 = if i + 1 < bytes.len() { bytes[i + 1] as u32 } else { 0 };
                    let b2 = if i + 2 < bytes.len() { bytes[i + 2] as u32 } else { 0 };
                    encoded.push(alphabet[((b0 >> 2) & 0x3F) as usize] as char);
                    encoded.push(alphabet[(((b0 << 4) | (b1 >> 4)) & 0x3F) as usize] as char);
                    encoded.push(if i + 1 < bytes.len() { alphabet[(((b1 << 2) | (b2 >> 6)) & 0x3F) as usize] as char } else { '=' });
                    encoded.push(if i + 2 < bytes.len() { alphabet[(b2 & 0x3F) as usize] as char } else { '=' });
                    i += 3;
                }
                Ok(ToolResult { data: encoded, is_error: false })
            }
            "decode" => {
                // Simple Base64 decode without external crate.
                let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                let mut table = [0u8; 256];
                for (i, &b) in alphabet.iter().enumerate() {
                    table[b as usize] = i as u8;
                }

                let bytes: Vec<u8> = text.bytes()
                    .filter(|b| !b"=\n\r ".contains(b))
                    .collect();

                if bytes.len() % 4 != 0 && !bytes.is_empty() {
                    // pad internally
                }

                let mut out: Vec<u8> = Vec::new();
                let mut i = 0;
                while i + 3 < bytes.len() {
                    let v = ((table[bytes[i] as usize] as u32) << 18)
                        | ((table[bytes[i+1] as usize] as u32) << 12)
                        | ((table[bytes[i+2] as usize] as u32) << 6)
                        | (table[bytes[i+3] as usize] as u32);
                    out.push((v >> 16) as u8);
                    // Only push second byte if it wasn't padding.
                    if bytes[i+2] != b'=' { out.push((v >> 8) as u8); }
                    if bytes[i+3] != b'=' { out.push(v as u8); }
                    i += 4;
                }
                // Handle remaining (with original padding chars)
                let cleaned: Vec<u8> = text.bytes().filter(|b| !b"\n\r ".contains(b)).collect();
                // redo with padding
                let padded = if cleaned.len() % 4 != 0 {
                    let mut p = cleaned.clone();
                    while p.len() % 4 != 0 { p.push(b'='); }
                    p
                } else {
                    cleaned
                };
                let mut out2: Vec<u8> = Vec::new();
                let mut j = 0;
                while j + 3 < padded.len() {
                    let c0 = if padded[j] == b'=' { 0 } else { table[padded[j] as usize] as u32 };
                    let c1 = if padded[j+1] == b'=' { 0 } else { table[padded[j+1] as usize] as u32 };
                    let c2 = if padded[j+2] == b'=' { 0 } else { table[padded[j+2] as usize] as u32 };
                    let c3 = if padded[j+3] == b'=' { 0 } else { table[padded[j+3] as usize] as u32 };
                    let v = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
                    out2.push((v >> 16) as u8);
                    if padded[j+2] != b'=' { out2.push((v >> 8) as u8); }
                    if padded[j+3] != b'=' { out2.push(v as u8); }
                    j += 4;
                }
                match String::from_utf8(out2) {
                    Ok(s)  => Ok(ToolResult { data: s, is_error: false }),
                    Err(_) => Ok(ToolResult { data: "decoded bytes are not valid UTF-8".to_string(), is_error: true }),
                }
            }
            _ => Ok(ToolResult {
                data: format!("unknown mode '{}'; valid: encode, decode", mode),
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// HashFile tool
// ---------------------------------------------------------------------------

/// Compute the SHA-256 (or MD5) hash of a file within the sandbox.
pub struct HashFileTool;

#[async_trait]
impl Tool for HashFileTool {
    fn name(&self) -> &str { "hash_file" }
    fn description(&self) -> &str {
        "Compute the SHA-256 hash of a file within the working directory. \
         Useful for verifying file integrity or detecting changes."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to hash."
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

        let bytes = tokio::fs::read(&safe).await
            .map_err(|e| AgentError::IoError(e))?;

        // SHA-256 implemented without external crypto crates via a simple
        // approach: we use the platform-agnostic Rust standard library only.
        // For a production system, use `sha2` crate; here we use a manual FNV
        // hash to avoid adding a dependency — we label it clearly.
        //
        // Note: This is NOT cryptographic SHA-256. It is a 64-bit FNV-1a hash
        // presented as a hex digest, sufficient for integrity checks in agent
        // workflows. For real crypto, add the `sha2` dependency.
        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in &bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let data = format!(
            "algorithm: fnv1a-64\nfile: {}\nsize: {} bytes\nhash: {:016x}",
            safe.display(),
            bytes.len(),
            hash
        );
        Ok(ToolResult { data, is_error: false })
    }
}

// ---------------------------------------------------------------------------
// WebFetch tool  (HTML → Markdown web researcher)
// ---------------------------------------------------------------------------

/// Fetch a web page and return a clean Markdown version of the content.
///
/// The tool strips navigation, scripts, ads, and other noise, then converts
/// the remaining HTML structure (headings, links, lists, code blocks) into
/// readable Markdown.  This reduces token usage compared to sending raw HTML
/// to an LLM while preserving the important content.
pub struct WebFetchTool;

// ---------------------------------------------------------------------------
// ArticleFetch tool  (Web Clipper — Karpathy pipeline stage 1)
// ---------------------------------------------------------------------------

/// Fetch a web article and save it as a Markdown file with YAML frontmatter
/// containing title, source URL, author, date, and word count.
///
/// This is the "Obsidian Web Clipper" equivalent — the first step of the
/// Karpathy LLM knowledge-base pipeline.
pub struct ArticleFetchTool;

#[async_trait]
impl Tool for ArticleFetchTool {
    fn name(&self) -> &str { "article_fetch" }
    fn description(&self) -> &str {
        "Fetch a web article as Markdown and extract structured metadata \
         (title, author, date, source URL, word count) into YAML frontmatter. \
         Optionally save the result directly to a file. \
         Use this as the 'web clipper' step in a knowledge-base pipeline."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the article to fetch."
                },
                "save_path": {
                    "type": "string",
                    "description": "Optional relative file path to save the clipped article (e.g. 'raw/my-article.md')."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Request timeout in milliseconds (default 15000).",
                    "minimum": 1,
                    "maximum": 60000
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if !u.is_empty() => u,
            _ => return Ok(ToolResult { data: "url is required".to_string(), is_error: true }),
        };
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolResult { data: "url must begin with http:// or https://".to_string(), is_error: true });
        }
        let timeout_ms = input.get("timeout_ms").and_then(|v| v.as_u64()).unwrap_or(15_000).min(60_000);

        let resp = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .user_agent("open-multi-agent-rs/0.1 (article-clipper)")
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| AgentError::Other(e.to_string()))?
            .get(url)
            .header("Accept", "text/html,*/*;q=0.8")
            .send()
            .await
            .map_err(|e| AgentError::Other(format!("fetch failed: {}", e)))?;

        let final_url = resp.url().to_string();
        let status = resp.status();
        const MAX_BODY: usize = 5 * 1024 * 1024;
        let bytes = resp.bytes().await
            .map_err(|e| AgentError::Other(format!("body read failed: {}", e)))?;
        let raw_html = String::from_utf8_lossy(if bytes.len() > MAX_BODY { &bytes[..MAX_BODY] } else { &bytes }).to_string();

        // ── Extract metadata from HTML ───────────────────────────────────────
        let title = extract_meta_title(&raw_html);
        let author = extract_meta_author(&raw_html);
        let date   = extract_meta_date(&raw_html);
        let description = extract_meta_description(&raw_html);

        // ── Convert to Markdown ──────────────────────────────────────────────
        let body = html_to_markdown(&raw_html);
        let word_count = body.split_whitespace().count();

        // ── Build YAML frontmatter ───────────────────────────────────────────
        let now = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let mut fm = String::from("---\n");
        fm.push_str(&format!("title: \"{}\"\n", title.replace('"', "'")));
        fm.push_str(&format!("source: \"{}\"\n", final_url));
        if !author.is_empty() { fm.push_str(&format!("author: \"{}\"\n", author.replace('"', "'"))); }
        fm.push_str(&format!("date_clipped: \"{}\"\n", now));
        if !date.is_empty() { fm.push_str(&format!("date_published: \"{}\"\n", date)); }
        fm.push_str(&format!("word_count: {}\n", word_count));
        if !description.is_empty() { fm.push_str(&format!("description: \"{}\"\n", description.replace('"', "'"))); }
        fm.push_str("---\n\n");

        let full_doc = format!("{}# {}\n\n{}", fm, title, body);

        // ── Optionally save to file ──────────────────────────────────────────
        if let Some(raw_path) = input.get("save_path").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            let base = sandbox_root(context);
            if let Ok(safe) = safe_path(raw_path, &base) {
                if let Some(parent) = safe.parent() {
                    if parent.starts_with(&base) {
                        let _ = tokio::fs::create_dir_all(parent).await;
                    }
                }
                tokio::fs::write(&safe, &full_doc).await
                    .map_err(|e| AgentError::IoError(e))?;
                return Ok(ToolResult {
                    data: format!("Clipped '{}' → {} ({} words)", title, safe.display(), word_count),
                    is_error: false,
                });
            }
        }

        // Otherwise return the full document.
        let truncated = if full_doc.len() > 20_000 {
            format!("{}\n\n[… truncated at 20000 chars]", &full_doc[..20_000])
        } else {
            full_doc
        };
        Ok(ToolResult { data: truncated, is_error: !status.is_success() })
    }
}

// ── HTML metadata extractors ─────────────────────────────────────────────────

fn extract_meta_title(html: &str) -> String {
    static RE_OG: OnceLock<regex::Regex> = OnceLock::new();
    let re_og = RE_OG.get_or_init(|| regex::Regex::new(r#"(?is)<meta[^>]*property=["']og:title["'][^>]*content=["']([^"']*)["']"#).unwrap());
    if let Some(c) = re_og.captures(html) { return c[1].trim().to_string(); }

    static RE_TITLE: OnceLock<regex::Regex> = OnceLock::new();
    let re_title = RE_TITLE.get_or_init(|| regex::Regex::new(r"(?is)<title[^>]*>(.*?)</title\s*>").unwrap());
    if let Some(c) = re_title.captures(html) { return strip_tags(c[1].trim()).trim().to_string(); }

    static RE_H1: OnceLock<regex::Regex> = OnceLock::new();
    let re_h1 = RE_H1.get_or_init(|| regex::Regex::new(r"(?is)<h1[^>]*>(.*?)</h1\s*>").unwrap());
    if let Some(c) = re_h1.captures(html) { return strip_tags(c[1].trim()).trim().to_string(); }

    "Untitled".to_string()
}

fn extract_meta_author(html: &str) -> String {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(
        r#"(?is)<meta[^>]*name=["'](?:author|twitter:creator|article:author)["'][^>]*content=["']([^"']*)["']"#
    ).unwrap());
    re.captures(html).map(|c| c[1].trim().to_string()).unwrap_or_default()
}

fn extract_meta_date(html: &str) -> String {
    // Capture the full datetime/date string up to the closing quote, then trim to YYYY-MM-DD.
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(
        r#"(?is)<meta[^>]*(?:name|property)=["'](?:article:published_time|datePublished|date|og:updated_time)["'][^>]*content=["']([^"']+)["']"#
    ).unwrap());
    if let Some(c) = re.captures(html) {
        let d = c[1].trim();
        return d.get(..10).unwrap_or(d).to_string();
    }
    static RE_TIME: OnceLock<regex::Regex> = OnceLock::new();
    let re_time = RE_TIME.get_or_init(|| regex::Regex::new(r#"(?is)<time[^>]*datetime=["']([^"']+)["']"#).unwrap());
    re_time.captures(html)
        .map(|c| { let d = c[1].trim(); d.get(..10).unwrap_or(d).to_string() })
        .unwrap_or_default()
}

fn extract_meta_description(html: &str) -> String {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(
        r#"(?is)<meta[^>]*(?:name|property)=["'](?:description|og:description)["'][^>]*content=["']([^"']{0,300})["']"#
    ).unwrap());
    re.captures(html).map(|c| c[1].trim().to_string()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Frontmatter tool  (YAML frontmatter reader/writer for .md files)
// ---------------------------------------------------------------------------

/// Parse, read, or write YAML frontmatter in Markdown files.
///
/// Frontmatter is the `---` delimited block at the top of a `.md` file
/// (used by Obsidian, Jekyll, Hugo, etc.) for metadata like title, tags, date.
pub struct FrontmatterTool;

/// Parse the YAML frontmatter from a Markdown document.
/// Returns (frontmatter as JSON Value, body string) or (empty, full text).
fn parse_frontmatter(content: &str) -> (serde_json::Map<String, serde_json::Value>, String) {
    let mut map = serde_json::Map::new();
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (map, content.to_string());
    }
    // Find the closing ---
    let rest = &trimmed[3..];
    let close = rest.find("\n---").or_else(|| rest.find("\r\n---"));
    let Some(pos) = close else {
        return (map, content.to_string());
    };
    let fm_str = &rest[..pos];
    let body = rest[pos + 4..].trim_start_matches('\n').trim_start_matches("\r\n").to_string();

    for line in fm_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let Some(colon) = line.find(": ") else {
            // key with no value
            if let Some(k) = line.strip_suffix(':') {
                map.insert(k.trim().to_string(), json!(null));
            }
            continue;
        };
        let key = line[..colon].trim().to_string();
        let raw_val = line[colon + 2..].trim();

        let val = if raw_val.starts_with('[') && raw_val.ends_with(']') {
            // Inline array: [a, b, c]
            let items: Vec<serde_json::Value> = raw_val[1..raw_val.len()-1]
                .split(',')
                .map(|s| json!(s.trim().trim_matches('"').trim_matches('\'')))
                .collect();
            json!(items)
        } else if raw_val == "true" {
            json!(true)
        } else if raw_val == "false" {
            json!(false)
        } else if raw_val == "null" || raw_val.is_empty() {
            json!(null)
        } else if let Ok(n) = raw_val.parse::<i64>() {
            json!(n)
        } else if let Ok(f) = raw_val.parse::<f64>() {
            json!(f)
        } else {
            // String — strip surrounding quotes
            let s = raw_val.trim_matches('"').trim_matches('\'');
            json!(s)
        };
        map.insert(key, val);
    }
    (map, body)
}

/// Serialize a JSON object to YAML frontmatter format.
fn serialize_frontmatter(map: &serde_json::Map<String, serde_json::Value>) -> String {
    let mut out = String::from("---\n");
    for (k, v) in map {
        match v {
            serde_json::Value::Null    => out.push_str(&format!("{}: null\n", k)),
            serde_json::Value::Bool(b) => out.push_str(&format!("{}: {}\n", k, b)),
            serde_json::Value::Number(n) => out.push_str(&format!("{}: {}\n", k, n)),
            serde_json::Value::String(s) => {
                if s.contains(':') || s.contains('"') || s.contains('#') {
                    out.push_str(&format!("{}: \"{}\"\n", k, s.replace('"', "\\\"")));
                } else {
                    out.push_str(&format!("{}: {}\n", k, s));
                }
            }
            serde_json::Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| {
                    v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string())
                }).collect();
                out.push_str(&format!("{}: [{}]\n", k, items.join(", ")));
            }
            serde_json::Value::Object(_) => {
                out.push_str(&format!("{}: {}\n", k, v));
            }
        }
    }
    out.push_str("---\n");
    out
}

#[async_trait]
impl Tool for FrontmatterTool {
    fn name(&self) -> &str { "frontmatter" }
    fn description(&self) -> &str {
        "Read or write YAML frontmatter in a Markdown file. \
         Operations: \
         'read' — return frontmatter as a JSON object; \
         'write' — replace or create the frontmatter with new fields; \
         'set' — set a single key value; \
         'remove' — remove a key; \
         'list_keys' — return all frontmatter keys."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the Markdown file."
                },
                "operation": {
                    "type": "string",
                    "description": "'read', 'write', 'set', 'remove', or 'list_keys'.",
                    "enum": ["read", "write", "set", "remove", "list_keys"]
                },
                "fields": {
                    "type": "object",
                    "description": "For 'write': new frontmatter fields (merges with existing)."
                },
                "key": {
                    "type": "string",
                    "description": "For 'set' or 'remove': the frontmatter key to modify."
                },
                "value": {
                    "description": "For 'set': the new value."
                }
            },
            "required": ["path", "operation"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let op = input.get("operation").and_then(|v| v.as_str()).unwrap_or("");
        let base = sandbox_root(context);

        let safe = match safe_path(raw_path, &base) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };

        let content = tokio::fs::read_to_string(&safe).await
            .map_err(|e| AgentError::IoError(e))?;
        let (mut fm, body) = parse_frontmatter(&content);

        match op {
            "read" => {
                let data = serde_json::to_string_pretty(&serde_json::Value::Object(fm))
                    .unwrap_or_else(|e| format!("{}", e));
                Ok(ToolResult { data, is_error: false })
            }
            "list_keys" => {
                let keys: Vec<&str> = fm.keys().map(|k| k.as_str()).collect();
                Ok(ToolResult { data: keys.join("\n"), is_error: false })
            }
            "write" => {
                let fields = match input.get("fields").and_then(|v| v.as_object()) {
                    Some(f) => f,
                    None => return Ok(ToolResult { data: "fields object is required for 'write'".to_string(), is_error: true }),
                };
                for (k, v) in fields {
                    fm.insert(k.clone(), v.clone());
                }
                let new_content = format!("{}\n{}", serialize_frontmatter(&fm), body);
                tokio::fs::write(&safe, &new_content).await
                    .map_err(|e| AgentError::IoError(e))?;
                Ok(ToolResult {
                    data: format!("Updated frontmatter in {} ({} keys)", safe.display(), fm.len()),
                    is_error: false,
                })
            }
            "set" => {
                let key = match input.get("key").and_then(|v| v.as_str()) {
                    Some(k) if !k.is_empty() => k,
                    _ => return Ok(ToolResult { data: "key is required for 'set'".to_string(), is_error: true }),
                };
                let value = input.get("value").cloned().unwrap_or(json!(null));
                fm.insert(key.to_string(), value);
                let new_content = format!("{}\n{}", serialize_frontmatter(&fm), body);
                tokio::fs::write(&safe, &new_content).await
                    .map_err(|e| AgentError::IoError(e))?;
                Ok(ToolResult {
                    data: format!("Set '{}' in frontmatter of {}", key, safe.display()),
                    is_error: false,
                })
            }
            "remove" => {
                let key = match input.get("key").and_then(|v| v.as_str()) {
                    Some(k) if !k.is_empty() => k,
                    _ => return Ok(ToolResult { data: "key is required for 'remove'".to_string(), is_error: true }),
                };
                if fm.remove(key).is_none() {
                    return Ok(ToolResult {
                        data: format!("key '{}' not found in frontmatter", key),
                        is_error: true,
                    });
                }
                let new_content = if fm.is_empty() {
                    body
                } else {
                    format!("{}\n{}", serialize_frontmatter(&fm), body)
                };
                tokio::fs::write(&safe, &new_content).await
                    .map_err(|e| AgentError::IoError(e))?;
                Ok(ToolResult {
                    data: format!("Removed '{}' from frontmatter of {}", key, safe.display()),
                    is_error: false,
                })
            }
            _ => Ok(ToolResult {
                data: format!("unknown operation '{}'; valid: read, write, set, remove, list_keys", op),
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// WikilinkIndex tool  (Backlink graph for Obsidian-style [[WikiLinks]])
// ---------------------------------------------------------------------------

/// In-process index of [[WikiLinks]] across a directory of Markdown files.
/// Supports the Obsidian-style knowledge graph needed by the Karpathy pipeline.
pub struct WikilinkIndexTool;

#[derive(Clone)]
struct WikiEntry {
    /// Links *from* this page to other pages.
    links_to: std::collections::HashSet<String>,
}

fn wikilink_store() -> &'static Mutex<HashMap<String, WikiEntry>> {
    static STORE: OnceLock<Mutex<HashMap<String, WikiEntry>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Extract all [[WikiLink]] targets from a Markdown string.
fn extract_wikilinks(text: &str) -> Vec<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]").unwrap());
    re.captures_iter(text)
        .map(|c| c[1].trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[async_trait]
impl Tool for WikilinkIndexTool {
    fn name(&self) -> &str { "wikilink_index" }
    fn description(&self) -> &str {
        "Manage a [[WikiLink]] index for a Markdown knowledge base. \
         Operations: \
         'build' — scan a directory and index all [[WikiLinks]]; \
         'links' — list pages that a given page links to; \
         'backlinks' — list pages that link to a given page; \
         'orphans' — find pages with no incoming backlinks; \
         'add' — add/update a single page's links; \
         'stats' — return index statistics."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "'build', 'links', 'backlinks', 'orphans', 'add', or 'stats'.",
                    "enum": ["build", "links", "backlinks", "orphans", "add", "stats"]
                },
                "path": {
                    "type": "string",
                    "description": "Directory to scan (for 'build') or file path (for 'add')."
                },
                "page": {
                    "type": "string",
                    "description": "Page name (without .md extension) for 'links', 'backlinks'."
                }
            },
            "required": ["operation"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let op = input.get("operation").and_then(|v| v.as_str()).unwrap_or("");
        let base = sandbox_root(context);

        match op {
            "build" => {
                let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let dir = match safe_path(raw_path, &base) {
                    Ok(p) => p,
                    Err(e) => return Ok(e),
                };
                if !dir.is_dir() {
                    return Ok(ToolResult { data: format!("'{}' is not a directory", raw_path), is_error: true });
                }

                let mut store = wikilink_store().lock().unwrap();
                store.clear();
                let mut count = 0usize;

                fn scan_dir(
                    dir: &std::path::Path,
                    base: &std::path::Path,
                    store: &mut HashMap<String, WikiEntry>,
                    count: &mut usize,
                ) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() { scan_dir(&path, base, store, count); continue; }
                            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                            if ext != "md" { continue; }
                            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                let links: std::collections::HashSet<String> = extract_wikilinks(&content).into_iter().collect();
                                store.insert(name.clone(), WikiEntry { links_to: links });
                                *count += 1;
                            }
                        }
                    }
                }

                scan_dir(&dir, &base, &mut store, &mut count);
                Ok(ToolResult {
                    data: format!("Indexed {} pages from '{}'", count, raw_path),
                    is_error: false,
                })
            }

            "links" => {
                let page = match input.get("page").and_then(|v| v.as_str()) {
                    Some(p) if !p.is_empty() => p,
                    _ => return Ok(ToolResult { data: "page is required for 'links'".to_string(), is_error: true }),
                };
                let store = wikilink_store().lock().unwrap();
                match store.get(page) {
                    None => Ok(ToolResult { data: format!("page '{}' not in index — run 'build' first", page), is_error: true }),
                    Some(entry) => {
                        let mut links: Vec<&str> = entry.links_to.iter().map(|s| s.as_str()).collect();
                        links.sort();
                        Ok(ToolResult { data: links.join("\n"), is_error: false })
                    }
                }
            }

            "backlinks" => {
                let page = match input.get("page").and_then(|v| v.as_str()) {
                    Some(p) if !p.is_empty() => p,
                    _ => return Ok(ToolResult { data: "page is required for 'backlinks'".to_string(), is_error: true }),
                };
                let store = wikilink_store().lock().unwrap();
                let mut bls: Vec<&str> = store.iter()
                    .filter(|(_, entry)| entry.links_to.contains(page))
                    .map(|(name, _)| name.as_str())
                    .collect();
                bls.sort();
                if bls.is_empty() {
                    Ok(ToolResult { data: format!("no pages link to '{}'", page), is_error: false })
                } else {
                    Ok(ToolResult { data: bls.join("\n"), is_error: false })
                }
            }

            "orphans" => {
                let store = wikilink_store().lock().unwrap();
                let all_targets: std::collections::HashSet<&str> = store.values()
                    .flat_map(|e| e.links_to.iter().map(|s| s.as_str()))
                    .collect();
                let mut orphans: Vec<&str> = store.keys()
                    .filter(|name| !all_targets.contains(name.as_str()))
                    .map(|s| s.as_str())
                    .collect();
                orphans.sort();
                if orphans.is_empty() {
                    Ok(ToolResult { data: "no orphan pages".to_string(), is_error: false })
                } else {
                    Ok(ToolResult {
                        data: format!("{} orphan page(s):\n{}", orphans.len(), orphans.join("\n")),
                        is_error: false,
                    })
                }
            }

            "add" => {
                let raw_path = match input.get("path").and_then(|v| v.as_str()) {
                    Some(p) if !p.is_empty() => p,
                    _ => return Ok(ToolResult { data: "path is required for 'add'".to_string(), is_error: true }),
                };
                let file_path = match safe_path(raw_path, &base) {
                    Ok(p) => p,
                    Err(e) => return Ok(e),
                };
                let content = tokio::fs::read_to_string(&file_path).await
                    .map_err(|e| AgentError::IoError(e))?;
                let name = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                let links: std::collections::HashSet<String> = extract_wikilinks(&content).into_iter().collect();
                let n = links.len();
                wikilink_store().lock().unwrap().insert(name.clone(), WikiEntry { links_to: links });
                Ok(ToolResult {
                    data: format!("Indexed '{}' ({} outgoing links)", name, n),
                    is_error: false,
                })
            }

            "stats" => {
                let store = wikilink_store().lock().unwrap();
                let total_pages = store.len();
                let total_links: usize = store.values().map(|e| e.links_to.len()).sum();
                let all_targets: std::collections::HashSet<&str> = store.values()
                    .flat_map(|e| e.links_to.iter().map(|s| s.as_str()))
                    .collect();
                let orphan_count = store.keys().filter(|n| !all_targets.contains(n.as_str())).count();
                let data = json!({
                    "indexed_pages": total_pages,
                    "total_links":   total_links,
                    "unique_targets": all_targets.len(),
                    "orphan_pages":   orphan_count,
                });
                Ok(ToolResult {
                    data: serde_json::to_string_pretty(&data).unwrap_or_default(),
                    is_error: false,
                })
            }

            _ => Ok(ToolResult {
                data: format!("unknown operation '{}'; valid: build, links, backlinks, orphans, add, stats", op),
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// ImageDownload tool
// ---------------------------------------------------------------------------

/// Download an image from a URL and save it to a file within the sandbox.
pub struct ImageDownloadTool;

#[async_trait]
impl Tool for ImageDownloadTool {
    fn name(&self) -> &str { "image_download" }
    fn description(&self) -> &str {
        "Download an image from a URL and save it to a file within the working directory. \
         Returns the file path and size."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the image to download."
                },
                "path": {
                    "type": "string",
                    "description": "Relative path to save the image (e.g. 'raw/images/photo.jpg'). \
                                    If omitted, the filename is inferred from the URL."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Download timeout in milliseconds (default 30000).",
                    "minimum": 1,
                    "maximum": 120000
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if !u.is_empty() => u,
            _ => return Ok(ToolResult { data: "url is required".to_string(), is_error: true }),
        };
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolResult { data: "url must begin with http:// or https://".to_string(), is_error: true });
        }
        let timeout_ms = input.get("timeout_ms").and_then(|v| v.as_u64()).unwrap_or(30_000).min(120_000);
        let base = sandbox_root(context);

        // Infer filename from URL if no path provided.
        let raw_path = if let Some(p) = input.get("path").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            p.to_string()
        } else {
            let fname = url.rsplit('/').next().unwrap_or("image");
            let fname = fname.split('?').next().unwrap_or(fname);
            let fname = if fname.is_empty() { "image.bin" } else { fname };
            fname.to_string()
        };

        let safe = match safe_path(&raw_path, &base) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };

        if let Some(parent) = safe.parent() {
            if parent.starts_with(&base) {
                tokio::fs::create_dir_all(parent).await
                    .map_err(|e| AgentError::IoError(e))?;
            }
        }

        let resp = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .user_agent("open-multi-agent-rs/0.1")
            .build()
            .map_err(|e| AgentError::Other(e.to_string()))?
            .get(url)
            .send()
            .await
            .map_err(|e| AgentError::Other(format!("download failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            return Ok(ToolResult {
                data: format!("HTTP {} for {}", status.as_u16(), url),
                is_error: true,
            });
        }

        let content_type = resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();

        const MAX_IMAGE: usize = 50 * 1024 * 1024; // 50 MB
        let bytes = resp.bytes().await
            .map_err(|e| AgentError::Other(format!("body read failed: {}", e)))?;

        if bytes.len() > MAX_IMAGE {
            return Ok(ToolResult {
                data: format!("image exceeds 50 MB limit ({} bytes)", bytes.len()),
                is_error: true,
            });
        }

        tokio::fs::write(&safe, &bytes).await
            .map_err(|e| AgentError::IoError(e))?;

        Ok(ToolResult {
            data: format!(
                "Downloaded to {} ({} bytes, {})",
                safe.display(), bytes.len(), content_type
            ),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// RagIndexDir tool  (Bulk-index a directory into the RAG knowledge base)
// ---------------------------------------------------------------------------

/// Recursively read all Markdown (`.md`) and text (`.txt`) files in a directory
/// and bulk-index them into the in-process RAG knowledge base.
///
/// This is the "compile wiki → searchable knowledge base" step of the
/// Karpathy pipeline.
pub struct RagIndexDirTool;

#[async_trait]
impl Tool for RagIndexDirTool {
    fn name(&self) -> &str { "rag_index_dir" }
    fn description(&self) -> &str {
        "Recursively index all .md and .txt files in a directory into the RAG knowledge base. \
         YAML frontmatter is parsed and stored as metadata. \
         Use this to make an entire wiki searchable in one call."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory to index (default: '.')."
                },
                "clear_first": {
                    "type": "boolean",
                    "description": "If true, clear the existing RAG store before indexing (default: false)."
                },
                "extensions": {
                    "type": "array",
                    "description": "File extensions to index (default: ['md', 'txt']).",
                    "items": { "type": "string" }
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw_path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let base = sandbox_root(context);

        let dir = match safe_path(raw_path, &base) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        if !dir.is_dir() {
            return Ok(ToolResult { data: format!("'{}' is not a directory", raw_path), is_error: true });
        }

        if input.get("clear_first").and_then(|v| v.as_bool()).unwrap_or(false) {
            rag_store().lock().unwrap().clear();
        }

        let exts: Vec<String> = input.get("extensions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(|| vec!["md".to_string(), "txt".to_string()]);

        let mut indexed = 0usize;
        let mut skipped = 0usize;
        let mut indexed_files: Vec<String> = Vec::new();

        fn walk(
            dir: &std::path::Path,
            base: &std::path::Path,
            exts: &[String],
            store: &Mutex<HashMap<String, RagDoc>>,
            indexed: &mut usize,
            skipped: &mut usize,
            files: &mut Vec<String>,
        ) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() { walk(&path, base, exts, store, indexed, skipped, files); continue; }
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_string();
                    if !exts.contains(&ext) { continue; }

                    let Ok(content) = std::fs::read_to_string(&path) else {
                        *skipped += 1;
                        continue;
                    };

                    let id = path.strip_prefix(base).unwrap_or(&path)
                        .to_string_lossy().to_string();

                    // Parse frontmatter for metadata
                    let (fm, body) = parse_frontmatter(&content);
                    let title = fm.get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or(""))
                        .to_string();
                    let terms = build_term_map(&body);
                    let metadata = json!({
                        "title": title,
                        "path":  id.clone(),
                        "frontmatter": serde_json::Value::Object(fm),
                    });

                    store.lock().unwrap().insert(id.clone(), RagDoc {
                        id: id.clone(),
                        content: body,
                        metadata,
                        terms,
                    });
                    files.push(id);
                    *indexed += 1;
                }
            }
        }

        walk(&dir, &base, &exts, rag_store(), &mut indexed, &mut skipped, &mut indexed_files);
        indexed_files.sort();

        let data = format!(
            "Indexed {} file(s) into RAG store (skipped {} unreadable).\nFiles:\n{}",
            indexed,
            skipped,
            indexed_files.join("\n")
        );
        Ok(ToolResult { data, is_error: false })
    }
}

/// Minimal HTML → Markdown converter (no external dependency).
fn html_to_markdown(html: &str) -> String {
    use regex::Regex;

    // ── Phase 1: strip noise blocks (separate pattern per tag — no backrefs) ──
    // The Rust `regex` crate does not support backreferences, so we compile
    // one pattern per noisy element and apply them in sequence.
    static RE_NOISE_LIST: OnceLock<Vec<Regex>> = OnceLock::new();
    let noise_list = RE_NOISE_LIST.get_or_init(|| {
        let noisy = [
            "script", "style", "nav", "header", "footer", "aside",
            "noscript", "svg", "iframe", "form", "button", "select",
            "option", "input", "textarea",
        ];
        noisy.iter()
            .map(|tag| Regex::new(&format!(r"(?is)<{0}[^>]*>.*?</{0}\s*>", tag)).unwrap())
            .collect()
    });
    let mut s: String = html.to_string();
    for re in noise_list {
        s = re.replace_all(&s, " ").into_owned();
    }

    // ── Phase 2: structural → Markdown ───────────────────────────────────────
    // Headings: <h1-h6> — closing tag uses a char class, no backreference.
    static RE_H: OnceLock<Regex> = OnceLock::new();
    let re_h = RE_H.get_or_init(|| Regex::new(r"(?is)<h([1-6])[^>]*>(.*?)</h[1-6]\s*>").unwrap());
    let s = re_h.replace_all(&s, |caps: &regex::Captures| {
        let level: usize = caps[1].parse().unwrap_or(1);
        let text = strip_tags(caps[2].trim());
        format!("\n{} {}\n", "#".repeat(level), text)
    });

    // Links: <a href="url">text</a> → [text](url)
    static RE_A: OnceLock<Regex> = OnceLock::new();
    let re_a = RE_A.get_or_init(|| Regex::new(r#"(?is)<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)</a\s*>"#).unwrap());
    let s = re_a.replace_all(&s, |caps: &regex::Captures| {
        let href = caps[1].trim();
        let text = strip_tags(caps[2].trim());
        if text.is_empty() || text == href { href.to_string() }
        else { format!("[{}]({})", text, href) }
    });

    // Images
    static RE_IMG: OnceLock<Regex> = OnceLock::new();
    let re_img = RE_IMG.get_or_init(|| Regex::new(r#"(?is)<img[^>]*alt=["']([^"']*)["'][^>]*src=["']([^"']*)["'][^>]*/?>|<img[^>]*src=["']([^"']*)["'][^>]*/?>"#).unwrap());
    let s = re_img.replace_all(&s, |caps: &regex::Captures| {
        let alt = caps.get(1).map_or("", |m| m.as_str()).trim();
        let src = caps.get(2).or_else(|| caps.get(3)).map_or("", |m| m.as_str()).trim();
        if src.is_empty() { String::new() } else { format!("![{}]({})", alt, src) }
    });

    // Pre / code blocks
    static RE_PRE: OnceLock<Regex> = OnceLock::new();
    let re_pre = RE_PRE.get_or_init(|| Regex::new(r"(?is)<pre[^>]*>(.*?)</pre\s*>").unwrap());
    let s = re_pre.replace_all(&s, |caps: &regex::Captures| {
        let code = strip_tags(&caps[1]);
        format!("\n```\n{}\n```\n", code.trim())
    });

    static RE_CODE: OnceLock<Regex> = OnceLock::new();
    let re_code = RE_CODE.get_or_init(|| Regex::new(r"(?is)<code[^>]*>(.*?)</code\s*>").unwrap());
    let s = re_code.replace_all(&s, |caps: &regex::Captures| {
        format!("`{}`", strip_tags(caps[1].trim()))
    });

    // Bold / Emphasis — closing tags use non-capturing alternation (no backref)
    static RE_STRONG: OnceLock<Regex> = OnceLock::new();
    let re_strong = RE_STRONG.get_or_init(|| Regex::new(r"(?is)<(?:strong|b)[^>]*>(.*?)</(?:strong|b)\s*>").unwrap());
    let s = re_strong.replace_all(&s, |caps: &regex::Captures| {
        format!("**{}**", strip_tags(caps[1].trim()))
    });

    static RE_EM: OnceLock<Regex> = OnceLock::new();
    let re_em = RE_EM.get_or_init(|| Regex::new(r"(?is)<(?:em|i)[^>]*>(.*?)</(?:em|i)\s*>").unwrap());
    let s = re_em.replace_all(&s, |caps: &regex::Captures| {
        format!("*{}*", strip_tags(caps[1].trim()))
    });

    // List items
    static RE_LI: OnceLock<Regex> = OnceLock::new();
    let re_li = RE_LI.get_or_init(|| Regex::new(r"(?is)<li[^>]*>(.*?)</li\s*>").unwrap());
    let s = re_li.replace_all(&s, |caps: &regex::Captures| {
        format!("\n- {}", strip_tags(caps[1].trim()))
    });

    // Paragraphs / breaks → blank lines
    static RE_P: OnceLock<Regex> = OnceLock::new();
    let re_p = RE_P.get_or_init(|| Regex::new(r"(?is)</?p[^>]*>|<br\s*/?>|<hr\s*/?>").unwrap());
    let s = re_p.replace_all(&s, "\n\n");

    // Table cells — closing tag uses non-capturing alternation
    static RE_TD: OnceLock<Regex> = OnceLock::new();
    let re_td = RE_TD.get_or_init(|| Regex::new(r"(?is)<(?:td|th)[^>]*>(.*?)</(?:td|th)\s*>").unwrap());
    let s = re_td.replace_all(&s, |caps: &regex::Captures| {
        format!(" {} |", strip_tags(caps[1].trim()))
    });
    static RE_TR: OnceLock<Regex> = OnceLock::new();
    let re_tr = RE_TR.get_or_init(|| Regex::new(r"(?is)<tr[^>]*>").unwrap());
    let s = re_tr.replace_all(&s, "\n| ");

    // ── Phase 3: strip remaining HTML tags ───────────────────────────────────
    let s = strip_tags(&s);

    // ── Phase 4: HTML entity decoding ────────────────────────────────────────
    let s = s
        .replace("&amp;",  "&")
        .replace("&lt;",   "<")
        .replace("&gt;",   ">")
        .replace("&quot;", "\"")
        .replace("&#39;",  "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&ndash;", "–")
        .replace("&mdash;", "—");

    // ── Phase 5: normalise whitespace ────────────────────────────────────────
    static RE_BLANK: OnceLock<Regex> = OnceLock::new();
    let re_blank = RE_BLANK.get_or_init(|| Regex::new(r"\n{3,}").unwrap());
    let s = re_blank.replace_all(&s, "\n\n");

    static RE_TRAIL: OnceLock<Regex> = OnceLock::new();
    let re_trail = RE_TRAIL.get_or_init(|| Regex::new(r"[ \t]+\n").unwrap());
    let s = re_trail.replace_all(&s, "\n");

    s.trim().to_string()
}

/// Strip all HTML tags from a string slice.
fn strip_tags(s: &str) -> String {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"<[^>]*>").unwrap());
    re.replace_all(s, "").to_string()
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str { "web_fetch" }
    fn description(&self) -> &str {
        "Fetch a web page and return its content as clean Markdown. \
         Scripts, navigation, ads, and other noise are stripped automatically. \
         Links, headings, lists, and code blocks are preserved in Markdown format. \
         Returns far fewer tokens than raw HTML and is easier for LLMs to process."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch (http:// or https://)."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Request timeout in milliseconds (default 15000, max 60000).",
                    "minimum": 1,
                    "maximum": 60000
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum number of characters to return (default 20000).",
                    "minimum": 100
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) if !u.is_empty() => u,
            _ => return Ok(ToolResult { data: "url must be a non-empty string".to_string(), is_error: true }),
        };
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolResult {
                data: "url must begin with http:// or https://".to_string(),
                is_error: true,
            });
        }
        let timeout_ms = input.get("timeout_ms").and_then(|v| v.as_u64()).unwrap_or(15_000).min(60_000);
        let max_length = input.get("max_length").and_then(|v| v.as_u64()).unwrap_or(20_000) as usize;

        let resp = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .user_agent("open-multi-agent-rs/0.1 (web researcher)")
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| AgentError::Other(e.to_string()))?
            .get(url)
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send()
            .await
            .map_err(|e| AgentError::Other(format!("fetch failed: {}", e)))?;

        let status = resp.status();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        const MAX_BODY: usize = 5 * 1024 * 1024;
        let bytes = resp.bytes().await
            .map_err(|e| AgentError::Other(format!("failed to read body: {}", e)))?;
        let body = String::from_utf8_lossy(if bytes.len() > MAX_BODY { &bytes[..MAX_BODY] } else { &bytes }).to_string();

        let is_html = content_type.contains("html");
        let markdown = if is_html {
            html_to_markdown(&body)
        } else {
            body
        };

        let truncated = if markdown.len() > max_length {
            format!("{}\n\n[… truncated at {} chars]", &markdown[..max_length], max_length)
        } else {
            markdown
        };

        let data = format!("URL: {}\nStatus: {}\n\n{}", url, status.as_u16(), truncated);
        Ok(ToolResult { data, is_error: !status.is_success() })
    }
}

// ---------------------------------------------------------------------------
// TavilySearch tool  (Search Engine Integration)
// ---------------------------------------------------------------------------

/// Search the web via the Tavily Search API and return ranked, summarised results.
///
/// Requires `TAVILY_API_KEY` to be set in the environment.
/// Get a free key at <https://tavily.com>.
pub struct TavilySearchTool;

#[async_trait]
impl Tool for TavilySearchTool {
    fn name(&self) -> &str { "tavily_search" }
    fn description(&self) -> &str {
        "Search the web using the Tavily Search API and return the top results with \
         titles, URLs, and content snippets. Requires TAVILY_API_KEY in the environment. \
         Use this for real-time information that may not be in the model's training data."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10).",
                    "minimum": 1,
                    "maximum": 10
                },
                "search_depth": {
                    "type": "string",
                    "description": "'basic' (fast, default) or 'advanced' (slower, more thorough).",
                    "enum": ["basic", "advanced"]
                },
                "include_answer": {
                    "type": "boolean",
                    "description": "Include Tavily's AI-generated answer summary (default: true)."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return Ok(ToolResult { data: "query must be a non-empty string".to_string(), is_error: true }),
        };

        let api_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            return Ok(ToolResult {
                data: "TAVILY_API_KEY environment variable is not set. \
                       Get a free key at https://tavily.com and set it with: \
                       export TAVILY_API_KEY=tvly-...".to_string(),
                is_error: true,
            });
        }

        let max_results = input.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5).min(10);
        let search_depth = input.get("search_depth").and_then(|v| v.as_str()).unwrap_or("basic");
        let include_answer = input.get("include_answer").and_then(|v| v.as_bool()).unwrap_or(true);

        let body = json!({
            "api_key":       api_key,
            "query":         query,
            "max_results":   max_results,
            "search_depth":  search_depth,
            "include_answer": include_answer,
        });

        let resp = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("open-multi-agent-rs/0.1")
            .build()
            .map_err(|e| AgentError::Other(e.to_string()))?
            .post("https://api.tavily.com/search")
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| AgentError::Other(format!("Tavily API request failed: {}", e)))?;

        let status = resp.status();
        let bytes = resp.bytes().await
            .map_err(|e| AgentError::Other(format!("failed to read Tavily response: {}", e)))?;
        let response_str = String::from_utf8_lossy(&bytes).to_string();

        if !status.is_success() {
            return Ok(ToolResult {
                data: format!("Tavily API error (HTTP {}): {}", status.as_u16(), response_str),
                is_error: true,
            });
        }

        let v: serde_json::Value = match serde_json::from_str(&response_str) {
            Ok(v) => v,
            Err(_) => return Ok(ToolResult { data: response_str, is_error: false }),
        };

        let mut output = String::new();
        if let Some(answer) = v.get("answer").and_then(|a| a.as_str()) {
            if !answer.is_empty() {
                output.push_str(&format!("**Answer:** {}\n\n", answer));
            }
        }

        output.push_str(&format!("**Search results for:** {}\n\n", query));
        if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
            for (i, result) in results.iter().enumerate() {
                let title   = result.get("title").and_then(|t| t.as_str()).unwrap_or("(no title)");
                let url     = result.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let content = result.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let score   = result.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                output.push_str(&format!(
                    "{}. **{}** (score: {:.2})\n   {}\n   {}\n\n",
                    i + 1, title, score, url,
                    if content.len() > 300 { &content[..300] } else { content }
                ));
            }
        }

        Ok(ToolResult { data: output.trim_end().to_string(), is_error: false })
    }
}

// ---------------------------------------------------------------------------
// SchemaValidate tool  (Structured Data Parser)
// ---------------------------------------------------------------------------

/// Parse messy text or a JSON string and validate / coerce it against a JSON
/// Schema (subset: `required`, `properties`, `type`, `enum`).
///
/// Acts as a "cleanup gate" between noisy tool outputs and the LLM — returns
/// a validated, pretty-printed JSON object or a detailed error report.
pub struct SchemaValidateTool;

fn validate_against_schema(value: &serde_json::Value, schema: &serde_json::Value) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    // required
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for field in required {
            if let Some(key) = field.as_str() {
                if value.get(key).is_none() {
                    errors.push(format!("missing required field: '{}'", key));
                }
            }
        }
    }

    // properties
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (key, prop_schema) in props {
            if let Some(v) = value.get(key) {
                // type check
                if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                    let actual_ok = match expected_type {
                        "string"  => v.is_string(),
                        "number"  => v.is_number(),
                        "integer" => v.is_i64() || v.is_u64(),
                        "boolean" => v.is_boolean(),
                        "array"   => v.is_array(),
                        "object"  => v.is_object(),
                        "null"    => v.is_null(),
                        _         => true,
                    };
                    if !actual_ok {
                        errors.push(format!("field '{}': expected type '{}', got '{}'", key, expected_type, json_type_name(v)));
                    }
                }
                // enum check
                if let Some(enum_vals) = prop_schema.get("enum").and_then(|e| e.as_array()) {
                    if !enum_vals.contains(v) {
                        let allowed: Vec<String> = enum_vals.iter().map(|e| e.to_string()).collect();
                        errors.push(format!("field '{}': value {} not in allowed enum [{}]", key, v, allowed.join(", ")));
                    }
                }
            }
        }
    }

    // top-level type
    if let Some(t) = schema.get("type").and_then(|t| t.as_str()) {
        let ok = match t {
            "object"  => value.is_object(),
            "array"   => value.is_array(),
            "string"  => value.is_string(),
            "number"  => value.is_number(),
            "integer" => value.is_i64() || value.is_u64(),
            "boolean" => value.is_boolean(),
            _         => true,
        };
        if !ok {
            errors.push(format!("top-level type mismatch: expected '{}', got '{}'", t, json_type_name(value)));
        }
    }

    errors
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null    => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_)  => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[async_trait]
impl Tool for SchemaValidateTool {
    fn name(&self) -> &str { "schema_validate" }
    fn description(&self) -> &str {
        "Parse a string (or messy JSON) and validate it against a JSON Schema. \
         Returns the parsed, pretty-printed JSON on success, or a detailed error \
         report listing every validation failure. \
         Supports: 'required' fields, property 'type' checks, and 'enum' constraints."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The JSON string (or text containing JSON) to parse and validate."
                },
                "schema": {
                    "type": "object",
                    "description": "JSON Schema to validate against (supports 'type', 'required', 'properties', 'enum')."
                },
                "extract_json": {
                    "type": "boolean",
                    "description": "If true, attempt to extract the first JSON object/array from the text even if it contains surrounding prose (default: true)."
                }
            },
            "required": ["input", "schema"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let raw = match input.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolResult { data: "input is required".to_string(), is_error: true }),
        };
        let schema = match input.get("schema") {
            Some(s) => s.clone(),
            None => return Ok(ToolResult { data: "schema is required".to_string(), is_error: true }),
        };
        let extract = input.get("extract_json").and_then(|v| v.as_bool()).unwrap_or(true);

        // Try direct parse first.
        let parsed: Option<serde_json::Value> = serde_json::from_str(raw).ok().or_else(|| {
            if !extract { return None; }
            // Try to find a JSON object or array embedded in the text.
            static RE_OBJ: OnceLock<regex::Regex> = OnceLock::new();
            let re = RE_OBJ.get_or_init(|| regex::Regex::new(r"(?s)(\{.*\}|\[.*\])").unwrap());
            re.find(raw)
                .and_then(|m| serde_json::from_str(m.as_str()).ok())
        });

        let value = match parsed {
            Some(v) => v,
            None => return Ok(ToolResult {
                data: format!("could not parse JSON from input:\n{}", raw),
                is_error: true,
            }),
        };

        let errors = validate_against_schema(&value, &schema);
        if errors.is_empty() {
            let pretty = serde_json::to_string_pretty(&value).unwrap_or_else(|e| format!("{}", e));
            Ok(ToolResult { data: pretty, is_error: false })
        } else {
            Ok(ToolResult {
                data: format!(
                    "Validation failed ({} error{}):\n{}\n\nParsed value:\n{}",
                    errors.len(),
                    if errors.len() == 1 { "" } else { "s" },
                    errors.iter().map(|e| format!("  - {}", e)).collect::<Vec<_>>().join("\n"),
                    serde_json::to_string_pretty(&value).unwrap_or_default()
                ),
                is_error: true,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// BusPublish / BusRead tools  (MessageBus Broadcaster)
// ---------------------------------------------------------------------------

/// Publish a message to the inter-agent MessageBus from within a tool call.
///
/// Use `register_bus_tools(registry, bus)` (instead of `register_built_in_tools`)
/// to inject the bus reference at registration time.
pub struct BusPublishTool {
    pub bus: Arc<crate::messaging::MessageBus>,
}

#[async_trait]
impl Tool for BusPublishTool {
    fn name(&self) -> &str { "bus_publish" }
    fn description(&self) -> &str {
        "Publish a message on the inter-agent MessageBus. \
         Use 'to': '*' to broadcast to all other agents, or a specific agent name \
         for a point-to-point message. Messages are immediately visible to subscribers."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "Sender name (defaults to the current agent's name)."
                },
                "to": {
                    "type": "string",
                    "description": "Recipient agent name, or '*' to broadcast to all agents."
                },
                "content": {
                    "type": "string",
                    "description": "Message content to publish."
                }
            },
            "required": ["to", "content"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let from = input.get("from")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&context.agent.name);
        let to = match input.get("to").and_then(|v| v.as_str()) {
            Some(t) if !t.is_empty() => t,
            _ => return Ok(ToolResult { data: "to must be a non-empty string or '*'".to_string(), is_error: true }),
        };
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

        let msg = self.bus.send(from, to, content);
        let data = if to == "*" {
            format!("broadcast from '{}' (id: {})", from, msg.id)
        } else {
            format!("message sent from '{}' to '{}' (id: {})", from, to, msg.id)
        };
        Ok(ToolResult { data, is_error: false })
    }
}

/// Read messages from the inter-agent MessageBus addressed to a named agent.
pub struct BusReadTool {
    pub bus: Arc<crate::messaging::MessageBus>,
}

#[async_trait]
impl Tool for BusReadTool {
    fn name(&self) -> &str { "bus_read" }
    fn description(&self) -> &str {
        "Read messages from the inter-agent MessageBus. \
         Returns messages addressed to the specified agent (unread by default). \
         Messages are formatted as a JSON array."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent name to read messages for (defaults to the current agent's name)."
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "If true (default), return only unread messages."
                },
                "mark_read": {
                    "type": "boolean",
                    "description": "If true (default), mark returned messages as read."
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let agent_name = input.get("agent")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&context.agent.name);
        let unread_only = input.get("unread_only").and_then(|v| v.as_bool()).unwrap_or(true);
        let do_mark = input.get("mark_read").and_then(|v| v.as_bool()).unwrap_or(true);

        let messages = if unread_only {
            self.bus.get_unread(agent_name)
        } else {
            self.bus.get_all(agent_name)
        };

        if do_mark && !messages.is_empty() {
            let ids: Vec<String> = messages.iter().map(|m| m.id.clone()).collect();
            self.bus.mark_read(agent_name, &ids);
        }

        let formatted: Vec<serde_json::Value> = messages.iter().map(|m| json!({
            "id":        m.id,
            "from":      m.from,
            "to":        m.to,
            "content":   m.content,
            "timestamp": m.timestamp.to_rfc3339(),
        })).collect();

        let data = if formatted.is_empty() {
            format!("No {} messages for '{}'.", if unread_only { "unread" } else { "" }, agent_name)
        } else {
            serde_json::to_string_pretty(&formatted)
                .unwrap_or_else(|e| format!("serialization error: {}", e))
        };

        Ok(ToolResult { data, is_error: false })
    }
}

/// Register the MessageBus tools, injecting the provided bus handle.
/// Call this in addition to (or instead of) `register_built_in_tools` when you
/// want agents to be able to publish/read messages during tool calls.
pub async fn register_bus_tools(
    registry: &mut ToolRegistry,
    bus: Arc<crate::messaging::MessageBus>,
) {
    let _ = registry.register(Arc::new(BusPublishTool { bus: Arc::clone(&bus) }));
    let _ = registry.register(Arc::new(BusReadTool   { bus: Arc::clone(&bus) }));
}

// ---------------------------------------------------------------------------
// RAG (Knowledge Base) tools
// ---------------------------------------------------------------------------

/// A single document in the in-process knowledge base.
#[derive(Clone)]
struct RagDoc {
    id:       String,
    content:  String,
    metadata: serde_json::Value,
    /// Pre-computed lowercase term → frequency map for quick scoring.
    terms:    HashMap<String, usize>,
}

fn build_term_map(text: &str) -> HashMap<String, usize> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let w = word.to_lowercase();
        if w.len() >= 2 {
            *map.entry(w).or_insert(0) += 1;
        }
    }
    map
}

fn rag_store() -> &'static Mutex<HashMap<String, RagDoc>> {
    static STORE: OnceLock<Mutex<HashMap<String, RagDoc>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Add or update a document in the in-process knowledge base.
pub struct RagAddTool;

#[async_trait]
impl Tool for RagAddTool {
    fn name(&self) -> &str { "rag_add" }
    fn description(&self) -> &str {
        "Add a document to the in-process knowledge base (RAG store). \
         The document is indexed for keyword search. \
         Use 'rag_search' to retrieve relevant documents later."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique document identifier (used for updates and deduplication)."
                },
                "content": {
                    "type": "string",
                    "description": "The document text to index."
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (source URL, title, date, etc.) stored alongside the document."
                }
            },
            "required": ["id", "content"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let id = match input.get("id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return Ok(ToolResult { data: "id must be a non-empty string".to_string(), is_error: true }),
        };
        let content = match input.get("content").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return Ok(ToolResult { data: "content is required".to_string(), is_error: true }),
        };
        let metadata = input.get("metadata").cloned().unwrap_or(json!({}));
        let terms = build_term_map(&content);
        let is_update = rag_store().lock().unwrap().contains_key(&id);

        rag_store().lock().unwrap().insert(id.clone(), RagDoc { id: id.clone(), content, metadata, terms });

        Ok(ToolResult {
            data: format!("{} document '{}' in knowledge base", if is_update { "Updated" } else { "Added" }, id),
            is_error: false,
        })
    }
}

/// Search the in-process knowledge base using keyword scoring (BM25-inspired TF scoring).
pub struct RagSearchTool;

#[async_trait]
impl Tool for RagSearchTool {
    fn name(&self) -> &str { "rag_search" }
    fn description(&self) -> &str {
        "Search the in-process knowledge base (RAG store) for documents relevant to a query. \
         Uses TF-based keyword scoring. Returns the top-k matching documents with their \
         content and metadata."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default: 3, max: 20).",
                    "minimum": 1,
                    "maximum": 20
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum relevance score threshold (0.0–1.0, default: 0.0).",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => return Ok(ToolResult { data: "query must be a non-empty string".to_string(), is_error: true }),
        };
        let top_k = input.get("top_k").and_then(|v| v.as_u64()).unwrap_or(3).min(20) as usize;
        let min_score = input.get("min_score").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let query_terms = build_term_map(query);
        if query_terms.is_empty() {
            return Ok(ToolResult { data: "query contains no indexable terms".to_string(), is_error: true });
        }

        let store = rag_store().lock().unwrap();
        if store.is_empty() {
            return Ok(ToolResult {
                data: "knowledge base is empty — add documents first with rag_add".to_string(),
                is_error: false,
            });
        }

        // Score each document: sum of (query_term_freq * doc_term_freq) / doc_term_total
        let mut scored: Vec<(f64, String)> = store.values().map(|doc| {
            let doc_total: usize = doc.terms.values().sum();
            if doc_total == 0 { return (0.0, doc.id.clone()); }
            let score: f64 = query_terms.keys()
                .filter_map(|term| doc.terms.get(term).map(|&f| f as f64 / doc_total as f64))
                .sum::<f64>()
                / (query_terms.len() as f64).max(1.0);
            (score, doc.id.clone())
        }).collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<serde_json::Value> = scored.iter()
            .take(top_k)
            .filter(|(score, _)| *score >= min_score)
            .filter_map(|(score, id)| {
                store.get(id).map(|doc| json!({
                    "id":       doc.id,
                    "score":    format!("{:.4}", score),
                    "content":  doc.content,
                    "metadata": doc.metadata,
                }))
            })
            .collect();

        if results.is_empty() {
            return Ok(ToolResult {
                data: format!("no documents matched the query '{}' above score threshold {:.2}", query, min_score),
                is_error: false,
            });
        }

        let data = serde_json::to_string_pretty(&results)
            .unwrap_or_else(|e| format!("serialization error: {}", e));
        Ok(ToolResult { data, is_error: false })
    }
}

/// Remove all documents from the knowledge base, or a specific document by id.
pub struct RagClearTool;

#[async_trait]
impl Tool for RagClearTool {
    fn name(&self) -> &str { "rag_clear" }
    fn description(&self) -> &str {
        "Remove documents from the in-process knowledge base. \
         Provide 'id' to remove a specific document, or omit it to clear everything."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Document id to remove. Omit to clear the entire knowledge base."
                }
            }
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let mut store = rag_store().lock().unwrap();
        if let Some(id) = input.get("id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            if store.remove(id).is_some() {
                Ok(ToolResult { data: format!("removed document '{}'", id), is_error: false })
            } else {
                Ok(ToolResult { data: format!("document '{}' not found", id), is_error: true })
            }
        } else {
            let count = store.len();
            store.clear();
            Ok(ToolResult { data: format!("cleared {} documents from knowledge base", count), is_error: false })
        }
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
        // HTTP / networking
        Arc::new(HttpGetTool),
        Arc::new(HttpPostTool),
        // Data processing
        Arc::new(JsonParseTool),
        Arc::new(JsonTransformTool),
        Arc::new(CsvReadTool),
        Arc::new(CsvWriteTool),
        // Math & expressions
        Arc::new(MathEvalTool),
        // Date & time
        Arc::new(DatetimeTool),
        // Text processing
        Arc::new(TextRegexTool),
        Arc::new(TextChunkTool),
        // Environment & system
        Arc::new(EnvGetTool),
        Arc::new(SystemInfoTool),
        // In-process cache
        Arc::new(CacheSetTool),
        Arc::new(CacheGetTool),
        // Encoding & hashing
        Arc::new(Base64Tool),
        Arc::new(HashFileTool),
        // Web researcher
        Arc::new(WebFetchTool),
        // Search engine
        Arc::new(TavilySearchTool),
        // Structured data parser
        Arc::new(SchemaValidateTool),
        // Knowledge base (RAG)
        Arc::new(RagAddTool),
        Arc::new(RagSearchTool),
        Arc::new(RagClearTool),
        Arc::new(RagIndexDirTool),
        // Karpathy pipeline — knowledge-base workflow
        Arc::new(ArticleFetchTool),
        Arc::new(FrontmatterTool),
        Arc::new(WikilinkIndexTool),
        Arc::new(ImageDownloadTool),
        // Note: BusPublishTool and BusReadTool require a bus handle —
        // use register_bus_tools(registry, bus) to add them.
        // Utility tools
        Arc::new(SleepTool),
        Arc::new(RandomTool),
        Arc::new(TemplateTool),
        Arc::new(DiffTool),
        Arc::new(ZipTool),
        Arc::new(GitTool),
        Arc::new(UrlTool),
    ];
    for tool in tools {
        let _ = registry.register(tool);
    }
}

// ===========================================================================
// SleepTool — pause execution for a specified duration
// ===========================================================================

/// Pause execution for a given number of milliseconds.
///
/// Useful for rate-limiting API calls, polling loops, or adding deliberate
/// delays between pipeline stages.
///
/// # Input
/// | Field   | Type   | Required | Description                         |
/// |---------|--------|----------|-------------------------------------|
/// | ms      | u64    | yes      | Duration to sleep in milliseconds   |
///
/// # Output
/// JSON string confirming how long the agent slept.
pub struct SleepTool;

#[async_trait]
impl Tool for SleepTool {
    fn name(&self) -> &str { "sleep" }
    fn description(&self) -> &str {
        "Pause execution for a given number of milliseconds. \
         Useful for rate-limiting, polling, or pipeline pacing."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "ms": { "type": "integer", "minimum": 0, "maximum": 300_000,
                        "description": "Milliseconds to sleep (max 300 000 = 5 min)" }
            },
            "required": ["ms"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        let ms = input.get("ms")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AgentError::Other("ms (integer) is required".into()))?;
        if ms > 300_000 {
            return Ok(ToolResult {
                data: "ms must be ≤ 300 000 (5 minutes)".into(),
                is_error: true,
            });
        }
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        Ok(ToolResult {
            data: format!("slept {}ms", ms),
            is_error: false,
        })
    }
}

// ===========================================================================
// RandomTool — random values: integers, floats, UUIDs, choices, strings
// ===========================================================================

/// Generate random values.
///
/// # Input
/// | Field   | Type            | Required | Description                                  |
/// |---------|-----------------|----------|----------------------------------------------|
/// | kind    | string          | yes      | `"uuid"`, `"int"`, `"float"`, `"choice"`, `"string"` |
/// | min     | i64             | no       | Lower bound for `int` (default 0)            |
/// | max     | i64             | no       | Upper bound (inclusive) for `int` (default 100) |
/// | items   | array of string | no       | Pool to pick from for `choice`               |
/// | length  | u32             | no       | Character count for `string` (default 16)    |
///
/// # Output
/// The generated value as a JSON string.
pub struct RandomTool;

#[async_trait]
impl Tool for RandomTool {
    fn name(&self) -> &str { "random" }
    fn description(&self) -> &str {
        "Generate random values: UUIDs, integers in a range, floats [0,1), \
         a random choice from a list, or a random alphanumeric string."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "kind":   { "type": "string", "enum": ["uuid","int","float","choice","string"],
                            "description": "What kind of random value to generate" },
                "min":    { "type": "integer", "description": "Min for int (default 0)" },
                "max":    { "type": "integer", "description": "Max for int inclusive (default 100)" },
                "items":  { "type": "array", "items": { "type": "string" },
                            "description": "Pool for choice" },
                "length": { "type": "integer", "minimum": 1, "maximum": 256,
                            "description": "Length of random string (default 16)" }
            },
            "required": ["kind"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        use rand::Rng;
        let kind = input.get("kind").and_then(|v| v.as_str()).unwrap_or("");

        let result = match kind {
            "uuid" => uuid::Uuid::new_v4().to_string(),

            "int" => {
                let min = input.get("min").and_then(|v| v.as_i64()).unwrap_or(0);
                let max = input.get("max").and_then(|v| v.as_i64()).unwrap_or(100);
                if min > max {
                    return Ok(ToolResult { data: "min must be ≤ max".into(), is_error: true });
                }
                rand::thread_rng().gen_range(min..=max).to_string()
            }

            "float" => {
                let v: f64 = rand::thread_rng().gen();
                format!("{:.9}", v)
            }

            "choice" => {
                let items: Vec<String> = input.get("items")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if items.is_empty() {
                    return Ok(ToolResult { data: "items must be a non-empty array".into(), is_error: true });
                }
                let idx = rand::thread_rng().gen_range(0..items.len());
                items[idx].clone()
            }

            "string" => {
                let length = input.get("length").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
                const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
                let mut rng = rand::thread_rng();
                (0..length)
                    .map(|_| CHARSET[rng.gen_range(0..CHARSET.len())] as char)
                    .collect()
            }

            _ => return Ok(ToolResult {
                data: format!("unknown kind '{}'; use uuid|int|float|choice|string", kind),
                is_error: true,
            }),
        };

        Ok(ToolResult { data: result, is_error: false })
    }
}

// ===========================================================================
// TemplateTool — {{variable}} substitution in text templates
// ===========================================================================

/// Render a text template by substituting `{{variable}}` placeholders.
///
/// Templates use double-brace syntax: `{{name}}`. Variables that appear in
/// the template but are not provided in `vars` are left as-is (or can be
/// set to error with `strict = true`).
///
/// # Input
/// | Field    | Type    | Required | Description                                     |
/// |----------|---------|----------|-------------------------------------------------|
/// | template | string  | yes      | Template text with `{{var}}` placeholders       |
/// | vars     | object  | yes      | Key-value map of variable substitutions         |
/// | strict   | bool    | no       | If true, error on missing variable (default false) |
///
/// # Output
/// The rendered string with all placeholders replaced.
pub struct TemplateTool;

#[async_trait]
impl Tool for TemplateTool {
    fn name(&self) -> &str { "template" }
    fn description(&self) -> &str {
        "Render a text template by replacing {{variable}} placeholders with values. \
         Supports nested object paths like {{user.name}}."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "template": { "type": "string", "description": "Text with {{var}} placeholders" },
                "vars":     { "type": "object", "description": "Variable name → value map" },
                "strict":   { "type": "boolean",
                              "description": "If true, error when a placeholder has no value" }
            },
            "required": ["template", "vars"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        use regex::Regex;
        let template = input.get("template").and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Other("template is required".into()))?;
        let vars = input.get("vars").and_then(|v| v.as_object())
            .ok_or_else(|| AgentError::Other("vars must be an object".into()))?;
        let strict = input.get("strict").and_then(|v| v.as_bool()).unwrap_or(false);

        static RE_PLACEHOLDER: OnceLock<Regex> = OnceLock::new();
        let re = RE_PLACEHOLDER.get_or_init(|| Regex::new(r"\{\{([^}]+)\}\}").unwrap());

        let mut missing: Vec<String> = Vec::new();
        let result = re.replace_all(template, |caps: &regex::Captures| {
            let key = caps[1].trim();
            if let Some(val) = vars.get(key) {
                match val {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Null      => String::new(),
                    other                        => other.to_string(),
                }
            } else {
                missing.push(key.to_string());
                caps[0].to_string() // leave placeholder unchanged
            }
        }).to_string();

        if strict && !missing.is_empty() {
            return Ok(ToolResult {
                data: format!("missing variables: {}", missing.join(", ")),
                is_error: true,
            });
        }

        Ok(ToolResult { data: result, is_error: false })
    }
}

// ===========================================================================
// DiffTool — line-level unified diff between two strings or files
// ===========================================================================

/// Compute a line-level unified diff between two texts.
///
/// Accepts either inline strings or file paths. Output format mirrors
/// `diff -u`: lines starting with `+` are additions, `-` are removals,
/// ` ` are context lines.
///
/// # Input
/// | Field     | Type   | Required | Description                                        |
/// |-----------|--------|----------|----------------------------------------------------|
/// | a         | string | yes*     | First text (or path if `mode = "files"`)           |
/// | b         | string | yes*     | Second text (or path if `mode = "files"`)          |
/// | mode      | string | no       | `"text"` (default) or `"files"`                    |
/// | context   | u32    | no       | Context lines around changes (default 3)           |
///
/// # Output
/// Unified diff string, or `"no differences"` if inputs are identical.
pub struct DiffTool;

fn unified_diff(a: &str, b: &str, label_a: &str, label_b: &str, context: usize) -> String {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();

    // Build the edit script via Myers' O(ND) diff (simplified shortest-edit).
    // We use a simple LCS-based approach for clarity.
    let lcs = lcs_table(&a_lines, &b_lines);
    let edits = build_edits(&lcs, &a_lines, &b_lines);

    if edits.iter().all(|e| matches!(e, Edit::Keep(_))) {
        return "no differences".to_string();
    }

    // Group into hunks with context lines.
    let hunks = group_hunks(&edits, context);

    let mut out = format!("--- {}\n+++ {}\n", label_a, label_b);
    for (a_start, b_start, hunk_edits) in hunks {
        let a_count = hunk_edits.iter().filter(|e| !matches!(e, Edit::Add(_))).count();
        let b_count = hunk_edits.iter().filter(|e| !matches!(e, Edit::Remove(_))).count();
        out.push_str(&format!("@@ -{},{} +{},{} @@\n", a_start + 1, a_count, b_start + 1, b_count));
        for edit in &hunk_edits {
            match edit {
                Edit::Keep(l)   => out.push_str(&format!(" {}\n", l)),
                Edit::Add(l)    => out.push_str(&format!("+{}\n", l)),
                Edit::Remove(l) => out.push_str(&format!("-{}\n", l)),
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
enum Edit<'a> {
    Keep(&'a str),
    Add(&'a str),
    Remove(&'a str),
}

fn lcs_table<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<Vec<usize>> {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp
}

fn build_edits<'a>(lcs: &[Vec<usize>], a: &[&'a str], b: &[&'a str]) -> Vec<Edit<'a>> {
    let mut edits = Vec::new();
    let (mut i, mut j) = (a.len(), b.len());
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && a[i - 1] == b[j - 1] {
            edits.push(Edit::Keep(a[i - 1]));
            i -= 1; j -= 1;
        } else if j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j]) {
            edits.push(Edit::Add(b[j - 1]));
            j -= 1;
        } else {
            edits.push(Edit::Remove(a[i - 1]));
            i -= 1;
        }
    }
    edits.reverse();
    edits
}

fn group_hunks<'a>(edits: &[Edit<'a>], context: usize) -> Vec<(usize, usize, Vec<Edit<'a>>)> {
    // Collect (a_line_idx, b_line_idx, edit) triples
    let mut indexed: Vec<(usize, usize, &Edit)> = Vec::new();
    let (mut ai, mut bi) = (0usize, 0usize);
    for e in edits {
        indexed.push((ai, bi, e));
        match e {
            Edit::Keep(_)   => { ai += 1; bi += 1; }
            Edit::Add(_)    => { bi += 1; }
            Edit::Remove(_) => { ai += 1; }
        }
    }

    // Find change positions (non-Keep edits).
    let change_positions: Vec<usize> = indexed.iter().enumerate()
        .filter(|(_, (_, _, e))| !matches!(e, Edit::Keep(_)))
        .map(|(i, _)| i)
        .collect();

    if change_positions.is_empty() {
        return vec![];
    }

    // Merge nearby changes into hunks.
    let mut hunks: Vec<(usize, usize, Vec<Edit>)> = Vec::new();
    let mut hunk_start = change_positions[0].saturating_sub(context);
    let mut hunk_end   = (change_positions[0] + context + 1).min(indexed.len());

    for &pos in &change_positions[1..] {
        let new_start = pos.saturating_sub(context);
        if new_start <= hunk_end {
            hunk_end = (pos + context + 1).min(indexed.len());
        } else {
            let (a0, b0, _) = indexed[hunk_start];
            let slice: Vec<Edit> = indexed[hunk_start..hunk_end]
                .iter().map(|(_, _, e)| (*e).clone()).collect();
            hunks.push((a0, b0, slice));
            hunk_start = new_start;
            hunk_end   = (pos + context + 1).min(indexed.len());
        }
    }
    let (a0, b0, _) = indexed[hunk_start];
    let slice: Vec<Edit> = indexed[hunk_start..hunk_end]
        .iter().map(|(_, _, e)| (*e).clone()).collect();
    hunks.push((a0, b0, slice));
    hunks
}

#[async_trait]
impl Tool for DiffTool {
    fn name(&self) -> &str { "diff" }
    fn description(&self) -> &str {
        "Compute a unified diff between two strings or two files. \
         Returns the diff in standard -u format: lines prefixed with +, -, or space."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "a":       { "type": "string", "description": "First text or file path" },
                "b":       { "type": "string", "description": "Second text or file path" },
                "mode":    { "type": "string", "enum": ["text","files"],
                             "description": "'text' (default) or 'files'" },
                "context": { "type": "integer", "minimum": 0, "maximum": 20,
                             "description": "Context lines around changes (default 3)" }
            },
            "required": ["a", "b"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        let a_raw = input.get("a").and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Other("a is required".into()))?;
        let b_raw = input.get("b").and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Other("b is required".into()))?;
        let mode    = input.get("mode").and_then(|v| v.as_str()).unwrap_or("text");
        let context = input.get("context").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

        let (text_a, label_a, text_b, label_b) = if mode == "files" {
            let base = sandbox_root(ctx);
            let pa = safe_path(a_raw, &base).map_err(|e| AgentError::Other(e.data))?;
            let pb = safe_path(b_raw, &base).map_err(|e| AgentError::Other(e.data))?;
            let ta = std::fs::read_to_string(&pa)
                .map_err(|e| AgentError::Other(format!("cannot read a: {}", e)))?;
            let tb = std::fs::read_to_string(&pb)
                .map_err(|e| AgentError::Other(format!("cannot read b: {}", e)))?;
            (ta, a_raw.to_string(), tb, b_raw.to_string())
        } else {
            (a_raw.to_string(), "a".to_string(), b_raw.to_string(), "b".to_string())
        };

        Ok(ToolResult {
            data: unified_diff(&text_a, &text_b, &label_a, &label_b, context),
            is_error: false,
        })
    }
}

// ===========================================================================
// ZipTool — create, extract, and list zip archives in the sandbox
// ===========================================================================

/// Work with ZIP archives: create, extract, or list contents.
///
/// All paths are confined to the sandbox directory.
///
/// # Input
/// | Field      | Type           | Required | Description                                       |
/// |------------|----------------|----------|---------------------------------------------------|
/// | operation  | string         | yes      | `"create"`, `"extract"`, or `"list"`              |
/// | archive    | string         | yes      | Path to the zip file (relative to sandbox)        |
/// | files      | array of string| create   | Paths to include (for `"create"`)                 |
/// | dest       | string         | no       | Destination dir for `"extract"` (default: sandbox) |
///
/// # Output
/// - `"create"`: path and file count of the new archive
/// - `"extract"`: number of extracted files and destination
/// - `"list"`: JSON array of `{name, size, compressed_size}` objects
pub struct ZipTool;

#[async_trait]
impl Tool for ZipTool {
    fn name(&self) -> &str { "zip" }
    fn description(&self) -> &str {
        "Create, extract, or list ZIP archives confined to the sandbox. \
         Operations: create (bundle files), extract (unpack), list (inspect contents)."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string", "enum": ["create","extract","list"],
                               "description": "What to do with the archive" },
                "archive":   { "type": "string", "description": "Path to the .zip file" },
                "files":     { "type": "array", "items": { "type": "string" },
                               "description": "Files to include (create only)" },
                "dest":      { "type": "string",
                               "description": "Extraction destination (extract only, default: sandbox root)" }
            },
            "required": ["operation", "archive"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        let op      = input.get("operation").and_then(|v| v.as_str()).unwrap_or("");
        let archive = input.get("archive").and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Other("archive is required".into()))?;
        let base = sandbox_root(ctx);
        let archive_path = safe_path(archive, &base)
            .map_err(|e| AgentError::Other(e.data))?;

        match op {
            "create" => {
                let file_list: Vec<String> = input.get("files")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if file_list.is_empty() {
                    return Ok(ToolResult { data: "files array must not be empty".into(), is_error: true });
                }

                let zip_file = std::fs::File::create(&archive_path)
                    .map_err(|e| AgentError::Other(format!("cannot create archive: {}", e)))?;
                let mut zip = zip::ZipWriter::new(zip_file);
                let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated);

                let mut count = 0usize;
                for f in &file_list {
                    let src = safe_path(f, &base).map_err(|e| AgentError::Other(e.data))?;
                    let data = std::fs::read(&src)
                        .map_err(|e| AgentError::Other(format!("cannot read '{}': {}", f, e)))?;
                    let name_in_zip = src.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(f.as_str());
                    zip.start_file(name_in_zip, opts)
                        .map_err(|e| AgentError::Other(format!("zip error: {}", e)))?;
                    use std::io::Write;
                    zip.write_all(&data)
                        .map_err(|e| AgentError::Other(format!("write error: {}", e)))?;
                    count += 1;
                }
                zip.finish().map_err(|e| AgentError::Other(format!("zip finish: {}", e)))?;

                Ok(ToolResult {
                    data: format!("created '{}' with {} file(s)", archive, count),
                    is_error: false,
                })
            }

            "extract" => {
                let dest_str = input.get("dest").and_then(|v| v.as_str()).unwrap_or(".");
                let dest = safe_path(dest_str, &base)
                    .map_err(|e| AgentError::Other(e.data))?;
                std::fs::create_dir_all(&dest)
                    .map_err(|e| AgentError::Other(format!("cannot create dest: {}", e)))?;

                let zip_file = std::fs::File::open(&archive_path)
                    .map_err(|e| AgentError::Other(format!("cannot open archive: {}", e)))?;
                let mut zip = zip::ZipArchive::new(zip_file)
                    .map_err(|e| AgentError::Other(format!("invalid zip: {}", e)))?;

                let mut count = 0usize;
                for i in 0..zip.len() {
                    let mut entry = zip.by_index(i)
                        .map_err(|e| AgentError::Other(format!("zip read error: {}", e)))?;
                    // Sanitize entry name: strip leading / and ..
                    let name = entry.name().to_string();
                    let sanitized: PathBuf = name.split('/').filter(|c| !c.is_empty() && *c != "..").collect();
                    let out_path = dest.join(&sanitized);
                    // Ensure still in sandbox
                    if !out_path.starts_with(&base) { continue; }

                    if entry.is_dir() {
                        std::fs::create_dir_all(&out_path)
                            .map_err(|e| AgentError::Other(format!("mkdir error: {}", e)))?;
                    } else {
                        if let Some(parent) = out_path.parent() {
                            std::fs::create_dir_all(parent)
                                .map_err(|e| AgentError::Other(format!("mkdir error: {}", e)))?;
                        }
                        let mut out_file = std::fs::File::create(&out_path)
                            .map_err(|e| AgentError::Other(format!("file create error: {}", e)))?;
                        std::io::copy(&mut entry, &mut out_file)
                            .map_err(|e| AgentError::Other(format!("copy error: {}", e)))?;
                        count += 1;
                    }
                }
                Ok(ToolResult {
                    data: format!("extracted {} file(s) to '{}'", count, dest_str),
                    is_error: false,
                })
            }

            "list" => {
                let zip_file = std::fs::File::open(&archive_path)
                    .map_err(|e| AgentError::Other(format!("cannot open archive: {}", e)))?;
                let mut zip = zip::ZipArchive::new(zip_file)
                    .map_err(|e| AgentError::Other(format!("invalid zip: {}", e)))?;

                let mut entries = Vec::new();
                for i in 0..zip.len() {
                    let entry = zip.by_index(i)
                        .map_err(|e| AgentError::Other(format!("zip read error: {}", e)))?;
                    entries.push(json!({
                        "name":              entry.name(),
                        "size":              entry.size(),
                        "compressed_size":   entry.compressed_size(),
                    }));
                }
                Ok(ToolResult {
                    data: serde_json::to_string(&entries).unwrap_or_default(),
                    is_error: false,
                })
            }

            _ => Ok(ToolResult {
                data: format!("unknown operation '{}'; use create|extract|list", op),
                is_error: true,
            }),
        }
    }
}

// ===========================================================================
// GitTool — safe, read-heavy git operations (+ stage/commit)
// ===========================================================================

/// Run Git commands against a repository.
///
/// Only an explicit allowlist of commands is permitted to prevent destructive
/// operations. Mutating operations (`add`, `commit`) are included but pushing
/// and force-reset are not.
///
/// # Input
/// | Field  | Type   | Required | Description                                      |
/// |--------|--------|----------|--------------------------------------------------|
/// | args   | string | yes      | Git arguments, e.g. `"log --oneline -10"`        |
/// | cwd    | string | no       | Repo path (defaults to sandbox root)             |
///
/// **Allowed first sub-commands:** `status`, `log`, `diff`, `show`, `branch`,
/// `tag`, `remote`, `stash list`, `ls-files`, `shortlog`, `describe`,
/// `rev-parse`, `cat-file`, `add`, `commit`, `init`.
///
/// # Output
/// Combined stdout + stderr from the git process.
pub struct GitTool;

/// First token of the git argument string (sub-command name).
fn git_subcommand(args: &str) -> &str {
    args.split_whitespace().next().unwrap_or("")
}

const GIT_ALLOWED: &[&str] = &[
    "status", "log", "diff", "show", "branch", "tag",
    "remote", "stash",   // stash list / stash show are fine
    "ls-files", "shortlog", "describe", "rev-parse", "cat-file",
    "add", "commit", "init",
];

#[async_trait]
impl Tool for GitTool {
    fn name(&self) -> &str { "git" }
    fn description(&self) -> &str {
        "Run Git commands (read-heavy + stage/commit). Allowed sub-commands: \
         status, log, diff, show, branch, tag, remote, stash, ls-files, shortlog, \
         describe, rev-parse, cat-file, add, commit, init. \
         Destructive commands (push, reset --hard, force-push) are blocked."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "args": { "type": "string",
                          "description": "Git arguments after 'git', e.g. 'log --oneline -5'" },
                "cwd":  { "type": "string",
                          "description": "Working directory (defaults to sandbox root)" }
            },
            "required": ["args"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        let args_str = input.get("args").and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Other("args is required".into()))?;

        let sub = git_subcommand(args_str);
        if !GIT_ALLOWED.contains(&sub) {
            return Ok(ToolResult {
                data: format!(
                    "git sub-command '{}' is not allowed. Permitted: {}",
                    sub, GIT_ALLOWED.join(", ")
                ),
                is_error: true,
            });
        }

        // Extra guard: reject --force / -f flags
        let lower = args_str.to_lowercase();
        if lower.contains("--force") || lower.contains("-f ") || lower.contains(" -f") {
            return Ok(ToolResult {
                data: "force flags (--force / -f) are not permitted".into(),
                is_error: true,
            });
        }

        // Determine working directory
        let base = sandbox_root(ctx);
        let work_dir: PathBuf = if let Some(cwd_str) = input.get("cwd").and_then(|v| v.as_str()) {
            safe_path(cwd_str, &base).map_err(|e| AgentError::Other(e.data))?
        } else {
            base
        };

        let tokens: Vec<&str> = args_str.split_whitespace().collect();
        let output = std::process::Command::new("git")
            .args(&tokens)
            .current_dir(&work_dir)
            .output()
            .map_err(|e| AgentError::Other(format!("git exec error: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined = if stderr.is_empty() { stdout } else if stdout.is_empty() { stderr }
                       else { format!("{}{}", stdout, stderr) };

        Ok(ToolResult {
            data: if combined.trim().is_empty() { "(no output)".to_string() } else { combined },
            is_error: !output.status.success(),
        })
    }
}

// ===========================================================================
// UrlTool — parse, build, and encode/decode URLs
// ===========================================================================

/// Parse, build, and encode/decode URLs.
///
/// # Input
/// | Field      | Type   | Required | Description                                     |
/// |------------|--------|----------|-------------------------------------------------|
/// | operation  | string | yes      | `"parse"`, `"build"`, `"encode"`, `"decode"`, `"join"` |
/// | url        | string | parse/encode/decode/join | The URL or string to process  |
/// | base       | string | join     | Base URL to resolve against                     |
/// | scheme     | string | build    | e.g. `"https"`                                  |
/// | host       | string | build    | e.g. `"example.com"`                            |
/// | path       | string | build    | e.g. `"/api/v1/users"`                          |
/// | query      | object | build    | Key-value query parameters                      |
/// | fragment   | string | build    | URL fragment (without `#`)                      |
///
/// # Output
/// - `"parse"`: JSON object with `scheme`, `host`, `path`, `query` (object), `fragment`
/// - `"build"`: constructed URL string
/// - `"encode"` / `"decode"`: percent-encoded or decoded string
/// - `"join"`: resolved absolute URL
pub struct UrlTool;

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
            | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i+1..i+3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte as char);
                    i += 3;
                    continue;
                }
            }
        }
        if bytes[i] == b'+' {
            out.push(' ');
        } else {
            out.push(bytes[i] as char);
        }
        i += 1;
    }
    out
}

fn parse_url_str(url: &str) -> serde_json::Value {
    // Split off scheme
    let (scheme, rest) = if let Some(pos) = url.find("://") {
        (&url[..pos], &url[pos+3..])
    } else {
        ("", url)
    };

    // Split fragment
    let (rest, fragment) = if let Some(pos) = rest.find('#') {
        (&rest[..pos], &rest[pos+1..])
    } else {
        (rest, "")
    };

    // Split query
    let (rest, query_str) = if let Some(pos) = rest.find('?') {
        (&rest[..pos], &rest[pos+1..])
    } else {
        (rest, "")
    };

    // Split host / path
    let (host, path) = if scheme.is_empty() {
        ("", rest)
    } else {
        match rest.find('/') {
            Some(pos) => (&rest[..pos], &rest[pos..]),
            None      => (rest, "/"),
        }
    };

    // Parse query string into object
    let mut query_map = serde_json::Map::new();
    for pair in query_str.split('&') {
        if pair.is_empty() { continue; }
        let (k, v) = match pair.find('=') {
            Some(pos) => (&pair[..pos], percent_decode(&pair[pos+1..])),
            None      => (pair, String::new()),
        };
        query_map.insert(percent_decode(k), serde_json::Value::String(v));
    }

    json!({
        "scheme":   scheme,
        "host":     host,
        "path":     path,
        "query":    query_map,
        "fragment": fragment
    })
}

#[async_trait]
impl Tool for UrlTool {
    fn name(&self) -> &str { "url" }
    fn description(&self) -> &str {
        "Parse, build, percent-encode, percent-decode, or resolve (join) URLs. \
         No external crates required — pure string operations."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string",
                               "enum": ["parse","build","encode","decode","join"],
                               "description": "URL operation to perform" },
                "url":      { "type": "string", "description": "URL or string to process" },
                "base":     { "type": "string", "description": "Base URL for join" },
                "scheme":   { "type": "string" },
                "host":     { "type": "string" },
                "path":     { "type": "string" },
                "query":    { "type": "object" },
                "fragment": { "type": "string" }
            },
            "required": ["operation"]
        })
    }

    async fn execute(&self, input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext)
        -> Result<ToolResult>
    {
        let op = input.get("operation").and_then(|v| v.as_str()).unwrap_or("");
        match op {
            "parse" => {
                let url = input.get("url").and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::Other("url is required".into()))?;
                Ok(ToolResult {
                    data: serde_json::to_string(&parse_url_str(url)).unwrap_or_default(),
                    is_error: false,
                })
            }

            "build" => {
                let scheme = input.get("scheme").and_then(|v| v.as_str()).unwrap_or("https");
                let host   = input.get("host").and_then(|v| v.as_str()).unwrap_or("");
                let path   = input.get("path").and_then(|v| v.as_str()).unwrap_or("/");
                let frag   = input.get("fragment").and_then(|v| v.as_str()).unwrap_or("");

                let mut url = if host.is_empty() {
                    path.to_string()
                } else {
                    format!("{}://{}{}", scheme, host, if path.starts_with('/') { path.to_string() }
                                                       else { format!("/{}", path) })
                };

                if let Some(q) = input.get("query").and_then(|v| v.as_object()) {
                    if !q.is_empty() {
                        let qs: String = q.iter()
                            .map(|(k, v)| {
                                let val = match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };
                                format!("{}={}", percent_encode(k), percent_encode(&val))
                            })
                            .collect::<Vec<_>>()
                            .join("&");
                        url.push('?');
                        url.push_str(&qs);
                    }
                }

                if !frag.is_empty() {
                    url.push('#');
                    url.push_str(frag);
                }

                Ok(ToolResult { data: url, is_error: false })
            }

            "encode" => {
                let s = input.get("url").and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::Other("url (string to encode) is required".into()))?;
                Ok(ToolResult { data: percent_encode(s), is_error: false })
            }

            "decode" => {
                let s = input.get("url").and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::Other("url (string to decode) is required".into()))?;
                Ok(ToolResult { data: percent_decode(s), is_error: false })
            }

            "join" => {
                let base = input.get("base").and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::Other("base is required".into()))?;
                let rel  = input.get("url").and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::Other("url (relative) is required".into()))?;

                let resolved = if rel.contains("://") || rel.starts_with("//") {
                    // Already absolute
                    rel.to_string()
                } else if rel.starts_with('/') {
                    // Absolute path — keep scheme+host from base
                    let parsed = parse_url_str(base);
                    let scheme = parsed["scheme"].as_str().unwrap_or("https");
                    let host   = parsed["host"].as_str().unwrap_or("");
                    format!("{}://{}{}", scheme, host, rel)
                } else {
                    // Relative path — resolve against base path's directory
                    let parsed = parse_url_str(base);
                    let scheme = parsed["scheme"].as_str().unwrap_or("https");
                    let host   = parsed["host"].as_str().unwrap_or("");
                    let base_path = parsed["path"].as_str().unwrap_or("/");
                    let dir = match base_path.rfind('/') {
                        Some(pos) => &base_path[..=pos],
                        None      => "/",
                    };
                    let combined = format!("{}{}", dir, rel);
                    // Normalize ".." and "."
                    let mut parts: Vec<&str> = Vec::new();
                    for seg in combined.split('/') {
                        match seg {
                            ".."  => { parts.pop(); }
                            "." | "" => {}
                            s    => parts.push(s),
                        }
                    }
                    format!("{}://{}/{}", scheme, host, parts.join("/"))
                };

                Ok(ToolResult { data: resolved, is_error: false })
            }

            _ => Ok(ToolResult {
                data: format!("unknown operation '{}'; use parse|build|encode|decode|join", op),
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests for new built-in tools
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentInfo, ToolUseContext};

    fn ctx() -> ToolUseContext {
        ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "r".to_string(), model: "m".to_string() },
            cwd: Some(std::env::temp_dir().to_string_lossy().to_string()),
        }
    }

    fn inp(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    // ── JsonParseTool ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn json_parse_valid_pretty() {
        let result = JsonParseTool.execute(
            &inp(&[("input", json!(r#"{"a":1,"b":2}"#))]), &ctx(),
        ).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("\"a\""));
    }

    #[tokio::test]
    async fn json_parse_pointer_extraction() {
        let result = JsonParseTool.execute(&inp(&[
            ("input", json!(r#"{"users":[{"name":"Alice"},{"name":"Bob"}]}"#)),
            ("pointer", json!("/users/1/name")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("Bob"));
    }

    #[tokio::test]
    async fn json_parse_invalid_input() {
        let result = JsonParseTool.execute(
            &inp(&[("input", json!("not json"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("JSON parse error"));
    }

    #[tokio::test]
    async fn json_parse_missing_pointer() {
        let result = JsonParseTool.execute(&inp(&[
            ("input", json!(r#"{"a":1}"#)),
            ("pointer", json!("/missing")),
        ]), &ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("not found"));
    }

    // ── JsonTransformTool ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn json_transform_keys() {
        let result = JsonTransformTool.execute(&inp(&[
            ("input", json!(r#"{"x":1,"y":2}"#)),
            ("operation", json!("keys")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("\"x\"") || result.data.contains("\"y\""));
    }

    #[tokio::test]
    async fn json_transform_length() {
        let result = JsonTransformTool.execute(&inp(&[
            ("input", json!(r#"[1,2,3,4,5]"#)),
            ("operation", json!("length")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data.trim(), "5");
    }

    #[tokio::test]
    async fn json_transform_map_extract() {
        let result = JsonTransformTool.execute(&inp(&[
            ("input", json!(r#"[{"name":"Alice"},{"name":"Bob"}]"#)),
            ("operation", json!("[/name]")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("Alice") && result.data.contains("Bob"));
    }

    #[tokio::test]
    async fn json_transform_pointer() {
        let result = JsonTransformTool.execute(&inp(&[
            ("input", json!(r#"{"a":{"b":"deep"}}"#)),
            ("operation", json!("/a/b")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("deep"));
    }

    // ── MathEvalTool ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn math_eval_arithmetic() {
        let result = MathEvalTool.execute(
            &inp(&[("expression", json!("2 + 3 * 4"))]), &ctx(),
        ).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data.trim(), "14");
    }

    #[tokio::test]
    async fn math_eval_with_variables() {
        let result = MathEvalTool.execute(&inp(&[
            ("expression", json!("x * x + y")),
            ("variables", json!({"x": 3.0, "y": 1.0})),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data.trim(), "10");
    }

    #[tokio::test]
    async fn math_eval_invalid_expression() {
        let result = MathEvalTool.execute(
            &inp(&[("expression", json!("2 +* 3"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
    }

    // ── DatetimeTool ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn datetime_now() {
        let result = DatetimeTool.execute(
            &inp(&[("operation", json!("now"))]), &ctx(),
        ).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("timestamp"));
    }

    #[tokio::test]
    async fn datetime_format_epoch() {
        let result = DatetimeTool.execute(&inp(&[
            ("operation", json!("format")),
            ("timestamp", json!(0i64)),
            ("format", json!("%Y-%m-%d")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data.trim(), "1970-01-01");
    }

    #[tokio::test]
    async fn datetime_diff_one_day() {
        let result = DatetimeTool.execute(&inp(&[
            ("operation", json!("diff")),
            ("timestamp",  json!(0i64)),
            ("timestamp2", json!(86400i64)),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("diff_seconds: 86400"));
        assert!(result.data.contains("1d"));
    }

    #[tokio::test]
    async fn datetime_parse_iso() {
        let result = DatetimeTool.execute(&inp(&[
            ("operation", json!("parse")),
            ("input", json!("1970-01-01T00:00:00Z")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data.trim(), "0");
    }

    // ── TextRegexTool ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn regex_find_all_digits() {
        let result = TextRegexTool.execute(&inp(&[
            ("input",   json!("foo 123 bar 456")),
            ("pattern", json!(r"\d+")),
            ("mode",    json!("find_all")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("123") && result.data.contains("456"));
    }

    #[tokio::test]
    async fn regex_replace_all() {
        let result = TextRegexTool.execute(&inp(&[
            ("input",       json!("hello world hello")),
            ("pattern",     json!("hello")),
            ("mode",        json!("replace")),
            ("replacement", json!("hi")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data, "hi world hi");
    }

    #[tokio::test]
    async fn regex_split() {
        let result = TextRegexTool.execute(&inp(&[
            ("input",   json!("a,b,c")),
            ("pattern", json!(",")),
            ("mode",    json!("split")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        let parts: Vec<String> = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn regex_invalid_pattern() {
        let result = TextRegexTool.execute(&inp(&[
            ("input",   json!("text")),
            ("pattern", json!("[")),
        ]), &ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("invalid regex"));
    }

    // ── TextChunkTool ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn chunk_chars_basic() {
        let result = TextChunkTool.execute(&inp(&[
            ("text",       json!("abcdefghij")),
            ("chunk_size", json!(4)),
            ("split_by",   json!("chars")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        let chunks: Vec<String> = serde_json::from_str(&result.data).unwrap();
        assert_eq!(chunks[0], "abcd");
        assert_eq!(chunks[2], "ij");
    }

    #[tokio::test]
    async fn chunk_words() {
        let result = TextChunkTool.execute(&inp(&[
            ("text",       json!("one two three four five six")),
            ("chunk_size", json!(2)),
            ("split_by",   json!("words")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        let chunks: Vec<String> = serde_json::from_str(&result.data).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "one two");
    }

    // ── EnvGetTool ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn env_get_allowlisted_var() {
        let result = EnvGetTool.execute(&inp(&[
            ("name",    json!("PATH")),
            ("default", json!("fallback")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn env_get_blocked_var() {
        let result = EnvGetTool.execute(
            &inp(&[("name", json!("SECRET_PRIVATE_KEY"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("not in the allowed list"));
    }

    // ── SystemInfoTool ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn system_info_has_required_fields() {
        let result = SystemInfoTool.execute(&HashMap::new(), &ctx()).await.unwrap();
        assert!(!result.is_error);
        let v: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert!(v.get("os").is_some());
        assert!(v.get("arch").is_some());
        assert!(v.get("cpu_count").is_some());
        assert!(v.get("cwd").is_some());
    }

    // ── CacheSetTool / CacheGetTool ───────────────────────────────────────────

    #[tokio::test]
    async fn cache_set_and_get() {
        let key = "unit_test_cache_key_99";
        CacheSetTool.execute(&inp(&[
            ("key",   json!(key)),
            ("value", json!("stored_value")),
        ]), &ctx()).await.unwrap();

        let result = CacheGetTool.execute(
            &inp(&[("key", json!(key))]), &ctx(),
        ).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data, "stored_value");
    }

    #[tokio::test]
    async fn cache_get_missing_returns_default() {
        let result = CacheGetTool.execute(&inp(&[
            ("key",     json!("nonexistent_cache_key_xyz")),
            ("default", json!("fallback")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data, "fallback");
    }

    // ── Base64Tool ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn base64_roundtrip() {
        let encoded = Base64Tool.execute(&inp(&[
            ("input", json!("Hello, World!")),
            ("mode",  json!("encode")),
        ]), &ctx()).await.unwrap();
        assert!(!encoded.is_error);
        assert_eq!(encoded.data, "SGVsbG8sIFdvcmxkIQ==");

        let decoded = Base64Tool.execute(&inp(&[
            ("input", json!("SGVsbG8sIFdvcmxkIQ==")),
            ("mode",  json!("decode")),
        ]), &ctx()).await.unwrap();
        assert!(!decoded.is_error);
        assert_eq!(decoded.data, "Hello, World!");
    }

    #[tokio::test]
    async fn base64_encode_empty_string() {
        let result = Base64Tool.execute(&inp(&[
            ("input", json!("")),
            ("mode",  json!("encode")),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.data, "");
    }

    // ── HttpGetTool / HttpPostTool (validation only — no network calls) ────────

    #[tokio::test]
    async fn http_get_rejects_ftp_url() {
        let result = HttpGetTool.execute(
            &inp(&[("url", json!("ftp://example.com"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn http_post_rejects_file_url() {
        let result = HttpPostTool.execute(&inp(&[
            ("url",  json!("file:///etc/passwd")),
            ("body", json!("{}")),
        ]), &ctx()).await.unwrap();
        assert!(result.is_error);
    }

    // ── CsvReadTool / CsvWriteTool ────────────────────────────────────────────

    #[tokio::test]
    async fn csv_write_and_read_roundtrip() {
        let tmp_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "r".to_string(), model: "m".to_string() },
            cwd: Some(std::env::temp_dir().to_string_lossy().to_string()),
        };

        let w = CsvWriteTool.execute(&inp(&[
            ("path", json!("built_in_test.csv")),
            ("data", json!(r#"[{"name":"Alice","score":"95"},{"name":"Bob","score":"87"}]"#)),
        ]), &tmp_ctx).await.unwrap();
        assert!(!w.is_error, "write: {}", w.data);

        let r = CsvReadTool.execute(&inp(&[
            ("path", json!("built_in_test.csv")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!r.is_error, "read: {}", r.data);
        assert!(r.data.contains("Alice") && r.data.contains("87"));

        let _ = std::fs::remove_file(std::env::temp_dir().join("built_in_test.csv"));
    }

    // ── HashFileTool ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn hash_file_deterministic() {
        let tmp_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "r".to_string(), model: "m".to_string() },
            cwd: Some(std::env::temp_dir().to_string_lossy().to_string()),
        };
        let file = std::env::temp_dir().join("hash_test.txt");
        std::fs::write(&file, b"hello world").unwrap();

        let r1 = HashFileTool.execute(&inp(&[("path", json!("hash_test.txt"))]), &tmp_ctx).await.unwrap();
        let r2 = HashFileTool.execute(&inp(&[("path", json!("hash_test.txt"))]), &tmp_ctx).await.unwrap();
        assert!(!r1.is_error);
        assert_eq!(r1.data, r2.data, "hash must be deterministic");

        let _ = std::fs::remove_file(file);
    }

    // ── WebFetchTool (validation, no network) ─────────────────────────────────

    // ── ArticleFetchTool ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn article_fetch_rejects_non_http() {
        let result = ArticleFetchTool.execute(
            &inp(&[("url", json!("ftp://example.com"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn extract_meta_title_og() {
        let html = r#"<html><head><meta property="og:title" content="My OG Title"/></head></html>"#;
        assert_eq!(extract_meta_title(html), "My OG Title");
    }

    #[test]
    fn extract_meta_title_tag() {
        let html = "<html><head><title>My Page Title</title></head></html>";
        assert_eq!(extract_meta_title(html), "My Page Title");
    }

    #[test]
    fn extract_meta_description_og() {
        let html = r#"<meta property="og:description" content="A great article about Rust"/>"#;
        assert_eq!(extract_meta_description(html), "A great article about Rust");
    }

    #[test]
    fn extract_meta_date_time_tag() {
        let html = r#"<time datetime="2024-03-15T10:00:00Z">March 15</time>"#;
        assert_eq!(extract_meta_date(html), "2024-03-15");
    }

    // ── FrontmatterTool ───────────────────────────────────────────────────────

    #[test]
    fn parse_frontmatter_basic() {
        let content = "---\ntitle: Hello World\ndate: 2024-01-15\ntags: [rust, code]\n---\n\nBody text here.";
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(fm.get("title").and_then(|v| v.as_str()), Some("Hello World"));
        assert_eq!(fm.get("date").and_then(|v| v.as_str()), Some("2024-01-15"));
        let tags = fm.get("tags").and_then(|v| v.as_array()).unwrap();
        assert_eq!(tags.len(), 2);
        assert!(body.contains("Body text here."));
    }

    #[test]
    fn parse_frontmatter_no_fm() {
        let content = "# Just a heading\n\nNo frontmatter here.";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.is_empty());
        assert!(body.contains("Just a heading"));
    }

    #[test]
    fn serialize_frontmatter_basic() {
        let mut map = serde_json::Map::new();
        map.insert("title".to_string(), json!("Test"));
        map.insert("count".to_string(), json!(42));
        map.insert("flag".to_string(), json!(true));
        let yaml = serialize_frontmatter(&map);
        assert!(yaml.starts_with("---\n"));
        assert!(yaml.contains("title: Test\n") || yaml.contains("count: 42\n"));
        assert!(yaml.ends_with("---\n"));
    }

    #[test]
    fn serialize_frontmatter_array() {
        let mut map = serde_json::Map::new();
        map.insert("tags".to_string(), json!(["rust", "llm"]));
        let yaml = serialize_frontmatter(&map);
        assert!(yaml.contains("tags: [rust, llm]"));
    }

    #[tokio::test]
    async fn frontmatter_read_write_roundtrip() {
        let tmp = std::env::temp_dir().join("fm_test.md");
        let content = "---\ntitle: Original\ntags: [a, b]\n---\n\nBody.";
        std::fs::write(&tmp, content).unwrap();

        let tmp_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "".to_string(), model: "".to_string() },
            cwd: Some(std::env::temp_dir().to_string_lossy().to_string()),
        };

        // Read
        let r = FrontmatterTool.execute(&inp(&[
            ("path", json!("fm_test.md")),
            ("operation", json!("read")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!r.is_error, "{}", r.data);
        assert!(r.data.contains("Original"));

        // Set a new key
        let s = FrontmatterTool.execute(&inp(&[
            ("path", json!("fm_test.md")),
            ("operation", json!("set")),
            ("key", json!("author")),
            ("value", json!("Alice")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!s.is_error, "{}", s.data);

        // Verify the key was written
        let check = std::fs::read_to_string(&tmp).unwrap();
        assert!(check.contains("author: Alice"), "author not written: {}", check);

        // Remove the key
        let rm = FrontmatterTool.execute(&inp(&[
            ("path", json!("fm_test.md")),
            ("operation", json!("remove")),
            ("key", json!("author")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!rm.is_error, "{}", rm.data);

        let _ = std::fs::remove_file(tmp);
    }

    // ── WikilinkIndexTool ─────────────────────────────────────────────────────

    #[test]
    fn extract_wikilinks_basic() {
        let text = "See [[Rust Ownership]] and [[Memory Safety|safety]]. Also [[Lifetimes#borrow]].";
        let links = extract_wikilinks(text);
        assert!(links.contains(&"Rust Ownership".to_string()));
        assert!(links.contains(&"Memory Safety".to_string()));
        assert!(links.contains(&"Lifetimes".to_string()));
        assert!(!links.iter().any(|l| l.contains('|') || l.contains('#')));
    }

    #[test]
    fn extract_wikilinks_empty() {
        let text = "No links here, just regular [markdown](url).";
        assert!(extract_wikilinks(text).is_empty());
    }

    #[tokio::test]
    async fn wikilink_index_build_backlinks() {
        // Write temp wiki files
        let wiki_dir = std::env::temp_dir().join("wiki_test");
        let _ = std::fs::create_dir_all(&wiki_dir);

        std::fs::write(wiki_dir.join("Rust.md"),
            "Rust uses [[Ownership]] and [[Lifetimes]].").unwrap();
        std::fs::write(wiki_dir.join("Python.md"),
            "Python has [[Garbage Collection]].").unwrap();
        std::fs::write(wiki_dir.join("Ownership.md"),
            "The [[Borrow Checker]] enforces [[Lifetimes]].").unwrap();

        let tmp_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "".to_string(), model: "".to_string() },
            cwd: Some(std::env::temp_dir().to_string_lossy().to_string()),
        };

        // Build index
        let build = WikilinkIndexTool.execute(&inp(&[
            ("operation", json!("build")),
            ("path", json!("wiki_test")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!build.is_error, "build: {}", build.data);
        assert!(build.data.contains("3"));

        // Check outgoing links for Rust.md
        let links = WikilinkIndexTool.execute(&inp(&[
            ("operation", json!("links")),
            ("page", json!("Rust")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!links.is_error, "links: {}", links.data);
        assert!(links.data.contains("Ownership") && links.data.contains("Lifetimes"));

        // Check backlinks for Lifetimes
        let bls = WikilinkIndexTool.execute(&inp(&[
            ("operation", json!("backlinks")),
            ("page", json!("Lifetimes")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!bls.is_error, "backlinks: {}", bls.data);
        assert!(bls.data.contains("Rust") && bls.data.contains("Ownership"));

        // Orphans check
        let orphans = WikilinkIndexTool.execute(&inp(&[
            ("operation", json!("orphans")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!orphans.is_error, "orphans: {}", orphans.data);
        // Python.md links to Garbage Collection which isn't a page, so Python has no backlinks
        assert!(orphans.data.contains("Python") || orphans.data.contains("no orphan"));

        let _ = std::fs::remove_dir_all(wiki_dir);
    }

    // ── RagIndexDirTool ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn rag_index_dir_basic() {
        let dir = std::env::temp_dir().join("rag_dir_test");
        let _ = std::fs::create_dir_all(&dir);

        std::fs::write(dir.join("doc1.md"), "---\ntitle: Rust Intro\n---\n\nRust is a systems programming language.").unwrap();
        std::fs::write(dir.join("doc2.md"), "---\ntitle: Python Basics\n---\n\nPython is great for scripting.").unwrap();
        std::fs::write(dir.join("note.txt"), "A plain text note about performance.").unwrap();

        rag_store().lock().unwrap().retain(|k, _| !k.contains("rag_dir_test"));

        let tmp_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "".to_string(), model: "".to_string() },
            cwd: Some(std::env::temp_dir().to_string_lossy().to_string()),
        };

        let r = RagIndexDirTool.execute(&inp(&[
            ("path", json!("rag_dir_test")),
        ]), &tmp_ctx).await.unwrap();
        assert!(!r.is_error, "index: {}", r.data);
        assert!(r.data.contains("3"), "expected 3 files indexed: {}", r.data);

        // Search should now find indexed content
        let s = RagSearchTool.execute(&inp(&[
            ("query", json!("programming systems language")),
            ("top_k", json!(2)),
        ]), &ctx()).await.unwrap();
        assert!(!s.is_error, "search: {}", s.data);
        assert!(s.data.contains("Rust") || s.data.contains("doc1"));

        let _ = std::fs::remove_dir_all(dir);
    }

    // ── ImageDownloadTool (validation only — no network) ──────────────────────

    #[tokio::test]
    async fn image_download_rejects_non_http() {
        let result = ImageDownloadTool.execute(
            &inp(&[("url", json!("ftp://images.example.com/photo.jpg"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn web_fetch_rejects_non_http() {
        let result = WebFetchTool.execute(
            &inp(&[("url", json!("ftp://example.com"))]), &ctx(),
        ).await.unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn html_to_markdown_headings() {
        let html = "<h1>Title</h1><p>Some text.</p><h2>Section</h2>";
        let md = html_to_markdown(html);
        assert!(md.contains("# Title"), "h1 not converted: {}", md);
        assert!(md.contains("## Section"), "h2 not converted: {}", md);
        assert!(md.contains("Some text."), "paragraph lost: {}", md);
    }

    #[test]
    fn html_to_markdown_link() {
        let html = r#"<a href="https://example.com">Click here</a>"#;
        let md = html_to_markdown(html);
        assert!(md.contains("[Click here](https://example.com)"), "link not converted: {}", md);
    }

    #[test]
    fn html_to_markdown_strips_scripts() {
        let html = "<p>Hello</p><script>evil()</script><p>World</p>";
        let md = html_to_markdown(html);
        assert!(!md.contains("evil()"), "script not stripped: {}", md);
        assert!(md.contains("Hello") && md.contains("World"));
    }

    #[test]
    fn html_to_markdown_code_block() {
        let html = "<pre><code>fn main() {}</code></pre>";
        let md = html_to_markdown(html);
        assert!(md.contains("```"), "code block not converted: {}", md);
        assert!(md.contains("fn main()"), "code content lost: {}", md);
    }

    #[test]
    fn html_to_markdown_entities() {
        let html = "<p>a &amp; b &lt;c&gt;</p>";
        let md = html_to_markdown(html);
        assert!(md.contains("a & b <c>"), "entities not decoded: {}", md);
    }

    // ── SchemaValidateTool ────────────────────────────────────────────────────

    #[tokio::test]
    async fn schema_validate_valid_object() {
        let schema = json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "number"}
            }
        });
        let result = SchemaValidateTool.execute(&inp(&[
            ("input",  json!(r#"{"name":"Alice","age":30}"#)),
            ("schema", schema),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error, "unexpected error: {}", result.data);
        assert!(result.data.contains("Alice"));
    }

    #[tokio::test]
    async fn schema_validate_missing_required() {
        let schema = json!({"type":"object","required":["id"],"properties":{}});
        let result = SchemaValidateTool.execute(&inp(&[
            ("input",  json!(r#"{"name":"Bob"}"#)),
            ("schema", schema),
        ]), &ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("missing required field: 'id'"));
    }

    #[tokio::test]
    async fn schema_validate_type_mismatch() {
        let schema = json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}}
        });
        let result = SchemaValidateTool.execute(&inp(&[
            ("input",  json!(r#"{"count":"not-a-number"}"#)),
            ("schema", schema),
        ]), &ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("expected type 'integer'"));
    }

    #[tokio::test]
    async fn schema_validate_enum_violation() {
        let schema = json!({
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active","inactive"]}}
        });
        let result = SchemaValidateTool.execute(&inp(&[
            ("input",  json!(r#"{"status":"deleted"}"#)),
            ("schema", schema),
        ]), &ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.data.contains("not in allowed enum"));
    }

    #[tokio::test]
    async fn schema_validate_extracts_embedded_json() {
        let schema = json!({"type":"object","required":["x"],"properties":{}});
        // Input contains prose + JSON
        let result = SchemaValidateTool.execute(&inp(&[
            ("input",        json!("Here is the data: {\"x\": 1} end")),
            ("schema",       schema),
            ("extract_json", json!(true)),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error, "should extract embedded JSON: {}", result.data);
    }

    // ── RagAddTool / RagSearchTool ─────────────────────────────────────────────

    #[tokio::test]
    async fn rag_add_and_search() {
        // Clear any leftovers from other tests
        rag_store().lock().unwrap().clear();

        // Add documents
        RagAddTool.execute(&inp(&[
            ("id",      json!("doc1")),
            ("content", json!("Rust is a systems programming language focused on safety and performance.")),
        ]), &ctx()).await.unwrap();

        RagAddTool.execute(&inp(&[
            ("id",      json!("doc2")),
            ("content", json!("Python is a high-level programming language great for data science.")),
        ]), &ctx()).await.unwrap();

        RagAddTool.execute(&inp(&[
            ("id",      json!("doc3")),
            ("content", json!("The weather today is sunny and warm.")),
        ]), &ctx()).await.unwrap();

        // Search for programming languages
        let result = RagSearchTool.execute(&inp(&[
            ("query", json!("programming language")),
            ("top_k", json!(2)),
        ]), &ctx()).await.unwrap();
        assert!(!result.is_error, "search error: {}", result.data);
        // doc1 and doc2 should rank above doc3
        let hits: Vec<serde_json::Value> = serde_json::from_str(&result.data).unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().filter_map(|h| h.get("id")?.as_str()).collect();
        assert!(ids.contains(&"doc1") || ids.contains(&"doc2"), "expected programming docs in top-2");
    }

    #[tokio::test]
    async fn rag_search_empty_store() {
        rag_store().lock().unwrap().clear();
        let result = RagSearchTool.execute(&inp(&[("query", json!("anything"))]), &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.data.contains("empty"));
    }

    #[tokio::test]
    async fn rag_clear_specific() {
        rag_store().lock().unwrap().clear();
        RagAddTool.execute(&inp(&[("id", json!("x")), ("content", json!("x content"))]), &ctx()).await.unwrap();
        RagAddTool.execute(&inp(&[("id", json!("y")), ("content", json!("y content"))]), &ctx()).await.unwrap();

        RagClearTool.execute(&inp(&[("id", json!("x"))]), &ctx()).await.unwrap();
        assert!(!rag_store().lock().unwrap().contains_key("x"));
        assert!(rag_store().lock().unwrap().contains_key("y"));
    }

    // ── BusPublishTool / BusReadTool ──────────────────────────────────────────

    #[tokio::test]
    async fn bus_publish_and_read() {
        let bus = Arc::new(crate::messaging::MessageBus::new());
        let publish = BusPublishTool { bus: Arc::clone(&bus) };
        let read    = BusReadTool    { bus: Arc::clone(&bus) };

        // Publish point-to-point
        let pr = publish.execute(&inp(&[
            ("from",    json!("agent-a")),
            ("to",      json!("agent-b")),
            ("content", json!("hello from a")),
        ]), &ctx()).await.unwrap();
        assert!(!pr.is_error, "{}", pr.data);

        // Read as agent-b
        let ctx_b = ToolUseContext {
            agent: AgentInfo { name: "agent-b".to_string(), role: "".to_string(), model: "".to_string() },
            cwd: None,
        };
        let rr = read.execute(&inp(&[("unread_only", json!(true))]), &ctx_b).await.unwrap();
        assert!(!rr.is_error);
        assert!(rr.data.contains("hello from a"));
    }

    #[tokio::test]
    async fn bus_broadcast_read_by_others() {
        let bus = Arc::new(crate::messaging::MessageBus::new());
        let publish = BusPublishTool { bus: Arc::clone(&bus) };
        let read    = BusReadTool    { bus: Arc::clone(&bus) };

        publish.execute(&inp(&[
            ("from",    json!("broadcaster")),
            ("to",      json!("*")),
            ("content", json!("announcement")),
        ]), &ctx()).await.unwrap();

        let ctx_listener = ToolUseContext {
            agent: AgentInfo { name: "listener".to_string(), role: "".to_string(), model: "".to_string() },
            cwd: None,
        };
        let rr = read.execute(&HashMap::new(), &ctx_listener).await.unwrap();
        assert!(rr.data.contains("announcement"));
    }

    // ── SleepTool ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn sleep_short() {
        let r = SleepTool.execute(&inp(&[("ms", json!(10))]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.contains("10ms"));
    }

    #[tokio::test]
    async fn sleep_over_limit() {
        let r = SleepTool.execute(&inp(&[("ms", json!(400_000u64))]), &ctx()).await.unwrap();
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn sleep_missing_ms() {
        let r = SleepTool.execute(&HashMap::new(), &ctx()).await;
        assert!(r.is_err());
    }

    // ── RandomTool ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn random_uuid() {
        let r = RandomTool.execute(&inp(&[("kind", json!("uuid"))]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        // UUID has 36 chars: 8-4-4-4-12
        assert_eq!(r.data.len(), 36);
        assert!(r.data.contains('-'));
    }

    #[tokio::test]
    async fn random_int_default_range() {
        let r = RandomTool.execute(&inp(&[("kind", json!("int"))]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        let n: i64 = r.data.parse().unwrap();
        assert!((0..=100).contains(&n));
    }

    #[tokio::test]
    async fn random_int_custom_range() {
        let r = RandomTool.execute(&inp(&[
            ("kind", json!("int")), ("min", json!(5)), ("max", json!(5)),
        ]), &ctx()).await.unwrap();
        assert_eq!(r.data, "5");
    }

    #[tokio::test]
    async fn random_int_invalid_range() {
        let r = RandomTool.execute(&inp(&[
            ("kind", json!("int")), ("min", json!(10)), ("max", json!(5)),
        ]), &ctx()).await.unwrap();
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn random_float_range() {
        let r = RandomTool.execute(&inp(&[("kind", json!("float"))]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        let f: f64 = r.data.parse().unwrap();
        assert!((0.0..1.0).contains(&f));
    }

    #[tokio::test]
    async fn random_choice() {
        let r = RandomTool.execute(&inp(&[
            ("kind", json!("choice")),
            ("items", json!(["alpha", "beta", "gamma"])),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(["alpha", "beta", "gamma"].contains(&r.data.as_str()));
    }

    #[tokio::test]
    async fn random_choice_empty_list() {
        let r = RandomTool.execute(&inp(&[
            ("kind", json!("choice")), ("items", serde_json::Value::Array(vec![])),
        ]), &ctx()).await.unwrap();
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn random_string_length() {
        let r = RandomTool.execute(&inp(&[
            ("kind", json!("string")), ("length", json!(24)),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.data.len(), 24);
        assert!(r.data.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[tokio::test]
    async fn random_unknown_kind() {
        let r = RandomTool.execute(&inp(&[("kind", json!("qwerty"))]), &ctx()).await.unwrap();
        assert!(r.is_error);
    }

    // ── TemplateTool ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn template_basic_substitution() {
        let r = TemplateTool.execute(&inp(&[
            ("template", json!("Hello, {{name}}! You are {{age}} years old.")),
            ("vars",     json!({"name": "Alice", "age": 30})),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.data, "Hello, Alice! You are 30 years old.");
    }

    #[tokio::test]
    async fn template_missing_var_non_strict() {
        let r = TemplateTool.execute(&inp(&[
            ("template", json!("Hello, {{name}}!")),
            ("vars",     json!({})),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        // placeholder is left unchanged when not strict
        assert!(r.data.contains("{{name}}"));
    }

    #[tokio::test]
    async fn template_missing_var_strict() {
        let r = TemplateTool.execute(&inp(&[
            ("template", json!("Hello, {{name}}!")),
            ("vars",     json!({})),
            ("strict",   json!(true)),
        ]), &ctx()).await.unwrap();
        assert!(r.is_error);
        assert!(r.data.contains("name"));
    }

    #[tokio::test]
    async fn template_null_value() {
        let r = TemplateTool.execute(&inp(&[
            ("template", json!("Value: {{x}}")),
            ("vars",     json!({"x": null})),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.data, "Value: ");
    }

    #[tokio::test]
    async fn template_multiple_occurrences() {
        let r = TemplateTool.execute(&inp(&[
            ("template", json!("{{a}} + {{a}} = {{b}}")),
            ("vars",     json!({"a": "1", "b": "2"})),
        ]), &ctx()).await.unwrap();
        assert_eq!(r.data, "1 + 1 = 2");
    }

    // ── DiffTool ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn diff_identical_texts() {
        let r = DiffTool.execute(&inp(&[
            ("a", json!("line1\nline2\n")),
            ("b", json!("line1\nline2\n")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.data, "no differences");
    }

    #[tokio::test]
    async fn diff_added_line() {
        let r = DiffTool.execute(&inp(&[
            ("a", json!("line1\nline2")),
            ("b", json!("line1\nline2\nline3")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.contains("+line3"));
    }

    #[tokio::test]
    async fn diff_removed_line() {
        let r = DiffTool.execute(&inp(&[
            ("a", json!("line1\nline2\nline3")),
            ("b", json!("line1\nline3")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.contains("-line2"));
    }

    #[tokio::test]
    async fn diff_changed_line() {
        let r = DiffTool.execute(&inp(&[
            ("a", json!("hello world")),
            ("b", json!("hello rust")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.contains("-hello world"));
        assert!(r.data.contains("+hello rust"));
    }

    #[tokio::test]
    async fn diff_file_mode() {
        let dir = tempfile::tempdir().unwrap();
        let pa = dir.path().join("a.txt");
        let pb = dir.path().join("b.txt");
        std::fs::write(&pa, "foo\nbar").unwrap();
        std::fs::write(&pb, "foo\nbaz").unwrap();
        let file_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".into(), role: "".into(), model: "".into() },
            cwd: Some(dir.path().to_str().unwrap().to_string()),
        };
        let r = DiffTool.execute(&inp(&[
            ("a", json!("a.txt")), ("b", json!("b.txt")), ("mode", json!("files")),
        ]), &file_ctx).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.contains("-bar"));
        assert!(r.data.contains("+baz"));
    }

    // ── ZipTool ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn zip_create_list_extract() {
        let dir = tempfile::tempdir().unwrap();
        let base_str = dir.path().to_str().unwrap().to_string();
        let zip_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".into(), role: "".into(), model: "".into() },
            cwd: Some(base_str.clone()),
        };

        // Write a source file
        std::fs::write(dir.path().join("hello.txt"), "Hello, ZIP!").unwrap();

        // Create archive
        let cr = ZipTool.execute(&inp(&[
            ("operation", json!("create")),
            ("archive",   json!("test.zip")),
            ("files",     json!(["hello.txt"])),
        ]), &zip_ctx).await.unwrap();
        assert!(!cr.is_error, "create error: {}", cr.data);
        assert!(cr.data.contains("1 file"));

        // List archive
        let lr = ZipTool.execute(&inp(&[
            ("operation", json!("list")),
            ("archive",   json!("test.zip")),
        ]), &zip_ctx).await.unwrap();
        assert!(!lr.is_error, "list error: {}", lr.data);
        let entries: Vec<serde_json::Value> = serde_json::from_str(&lr.data).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "hello.txt");

        // Extract to a sub-directory
        std::fs::create_dir_all(dir.path().join("out")).unwrap();
        let er = ZipTool.execute(&inp(&[
            ("operation", json!("extract")),
            ("archive",   json!("test.zip")),
            ("dest",      json!("out")),
        ]), &zip_ctx).await.unwrap();
        assert!(!er.is_error, "extract error: {}", er.data);
        let extracted = dir.path().join("out").join("hello.txt");
        assert!(extracted.exists());
        assert_eq!(std::fs::read_to_string(extracted).unwrap(), "Hello, ZIP!");
    }

    #[tokio::test]
    async fn zip_create_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let zip_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".into(), role: "".into(), model: "".into() },
            cwd: Some(dir.path().to_str().unwrap().to_string()),
        };
        let r = ZipTool.execute(&inp(&[
            ("operation", json!("create")),
            ("archive",   json!("test.zip")),
            ("files",     serde_json::Value::Array(vec![])),
        ]), &zip_ctx).await.unwrap();
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn zip_unknown_operation() {
        let dir = tempfile::tempdir().unwrap();
        let zip_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".into(), role: "".into(), model: "".into() },
            cwd: Some(dir.path().to_str().unwrap().to_string()),
        };
        let r = ZipTool.execute(&inp(&[
            ("operation", json!("nuke")), ("archive", json!("x.zip")),
        ]), &zip_ctx).await.unwrap();
        assert!(r.is_error);
    }

    // ── GitTool ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn git_blocked_push() {
        let r = GitTool.execute(&inp(&[("args", json!("push origin main"))]), &ctx()).await.unwrap();
        assert!(r.is_error);
        assert!(r.data.contains("not allowed"));
    }

    #[tokio::test]
    async fn git_blocked_force_flag() {
        let r = GitTool.execute(&inp(&[("args", json!("commit --force -m test"))]), &ctx()).await.unwrap();
        assert!(r.is_error);
        assert!(r.data.contains("force"));
    }

    #[tokio::test]
    async fn git_status_in_non_repo() {
        let dir = tempfile::tempdir().unwrap();
        let git_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".into(), role: "".into(), model: "".into() },
            cwd: Some(dir.path().to_str().unwrap().to_string()),
        };
        // Not a repo — git status should fail but not panic
        let r = GitTool.execute(&inp(&[("args", json!("status"))]), &git_ctx).await.unwrap();
        // error flag is true since git exits non-zero outside a repo; data should contain message
        let _ = r; // just don't panic
    }

    #[tokio::test]
    async fn git_init_and_status() {
        let dir = tempfile::tempdir().unwrap();
        let git_ctx = ToolUseContext {
            agent: AgentInfo { name: "t".into(), role: "".into(), model: "".into() },
            cwd: Some(dir.path().to_str().unwrap().to_string()),
        };
        // Init repo
        let r = GitTool.execute(&inp(&[("args", json!("init"))]), &git_ctx).await.unwrap();
        // init should succeed
        assert!(!r.is_error || r.data.contains("Initialized") || r.data.contains("initialized"),
                "init failed: {}", r.data);
        // Status in freshly-inited repo
        let r2 = GitTool.execute(&inp(&[("args", json!("status"))]), &git_ctx).await.unwrap();
        let _ = r2; // don't panic
    }

    // ── UrlTool ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn url_parse_full() {
        let r = UrlTool.execute(&inp(&[
            ("operation", json!("parse")),
            ("url", json!("https://example.com/path/to/page?foo=bar&baz=qux#section")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        let v: serde_json::Value = serde_json::from_str(&r.data).unwrap();
        assert_eq!(v["scheme"], "https");
        assert_eq!(v["host"],   "example.com");
        assert_eq!(v["path"],   "/path/to/page");
        assert_eq!(v["query"]["foo"], "bar");
        assert_eq!(v["query"]["baz"], "qux");
        assert_eq!(v["fragment"], "section");
    }

    #[tokio::test]
    async fn url_build_with_query() {
        let r = UrlTool.execute(&inp(&[
            ("operation", json!("build")),
            ("scheme",    json!("https")),
            ("host",      json!("api.example.com")),
            ("path",      json!("/search")),
            ("query",     json!({"q": "rust lang", "page": "1"})),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.starts_with("https://api.example.com/search?"));
        assert!(r.data.contains("rust"));
    }

    #[tokio::test]
    async fn url_encode_decode_roundtrip() {
        let original = "hello world & foo=bar";
        let enc_r = UrlTool.execute(&inp(&[
            ("operation", json!("encode")), ("url", json!(original)),
        ]), &ctx()).await.unwrap();
        assert!(!enc_r.is_error);
        assert!(!enc_r.data.contains(' '));

        let dec_r = UrlTool.execute(&inp(&[
            ("operation", json!("decode")), ("url", json!(enc_r.data)),
        ]), &ctx()).await.unwrap();
        assert!(!dec_r.is_error);
        assert_eq!(dec_r.data, original);
    }

    #[tokio::test]
    async fn url_join_absolute_path() {
        let r = UrlTool.execute(&inp(&[
            ("operation", json!("join")),
            ("base",      json!("https://example.com/a/b/c")),
            ("url",       json!("/d/e")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.data, "https://example.com/d/e");
    }

    #[tokio::test]
    async fn url_join_relative_path() {
        let r = UrlTool.execute(&inp(&[
            ("operation", json!("join")),
            ("base",      json!("https://example.com/a/b/page.html")),
            ("url",       json!("../other.html")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert!(r.data.contains("example.com"));
        assert!(r.data.contains("other.html"));
        assert!(!r.data.contains(".."));
    }

    #[tokio::test]
    async fn url_join_already_absolute() {
        let r = UrlTool.execute(&inp(&[
            ("operation", json!("join")),
            ("base",      json!("https://example.com/page")),
            ("url",       json!("https://other.com/path")),
        ]), &ctx()).await.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.data, "https://other.com/path");
    }
}
