use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::types::{ToolResult, ToolUseContext};
use super::{Tool, ToolRegistry};

// ---------------------------------------------------------------------------
// Bash tool
// ---------------------------------------------------------------------------

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str { "bash" }
    fn description(&self) -> &str { "Execute a shell command and return its output." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command to execute." },
                "timeout_ms": { "type": "number", "description": "Timeout in milliseconds (default 30000)." }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let data = if stderr.is_empty() {
            stdout
        } else if stdout.is_empty() {
            stderr.clone()
        } else {
            format!("{}\nSTDERR: {}", stdout, stderr)
        };

        Ok(ToolResult {
            data,
            is_error: !output.status.success(),
        })
    }
}

// ---------------------------------------------------------------------------
// FileRead tool
// ---------------------------------------------------------------------------

pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str { "file_read" }
    fn description(&self) -> &str { "Read the contents of a file." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read." },
                "start_line": { "type": "number", "description": "Starting line (1-based, optional)." },
                "end_line": { "type": "number", "description": "Ending line (inclusive, optional)." }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = tokio::fs::read_to_string(path).await?;

        let start = input.get("start_line").and_then(|v| v.as_u64()).map(|n| n as usize);
        let end = input.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);

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

pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str { "file_write" }
    fn description(&self) -> &str { "Write content to a file." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to write the file." },
                "content": { "type": "string", "description": "Content to write." }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

        tokio::fs::write(path, content).await?;

        Ok(ToolResult {
            data: format!("File written to {}", path),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Grep tool
// ---------------------------------------------------------------------------

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str { "grep" }
    fn description(&self) -> &str { "Search for a pattern in files." }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Pattern to search for." },
                "path": { "type": "string", "description": "File or directory to search in." },
                "recursive": { "type": "boolean", "description": "Search recursively (default: false)." }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let recursive = input.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);

        let mut cmd = tokio::process::Command::new("grep");
        if recursive {
            cmd.arg("-r");
        }
        cmd.arg("-n").arg(pattern).arg(path);

        let output = cmd.output().await?;
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
