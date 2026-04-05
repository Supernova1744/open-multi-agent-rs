pub mod built_in;

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{AgentError, Result};
use crate::types::{LLMToolDef, ToolResult, ToolUseContext};

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// A tool that can be registered and invoked by agents.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the tool's input parameters.
    fn input_schema(&self) -> serde_json::Value;
    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult>;
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(AgentError::Other(format!(
                "ToolRegistry: a tool named '{}' is already registered.",
                name
            )));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
    }

    /// Convert registered tools to LLMToolDef format for API calls.
    pub fn to_tool_defs(&self, allowed: Option<&[String]>) -> Vec<LLMToolDef> {
        self.tools
            .values()
            .filter(|t| {
                allowed
                    .map(|list| list.iter().any(|n| n == t.name()))
                    .unwrap_or(true)
            })
            .map(|t| LLMToolDef {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ToolExecutor
// ---------------------------------------------------------------------------

/// Executes tool calls, catching errors and returning ToolResult.
pub struct ToolExecutor {
    pub registry: Arc<tokio::sync::Mutex<ToolRegistry>>,
}

impl ToolExecutor {
    pub fn new(registry: Arc<tokio::sync::Mutex<ToolRegistry>>) -> Self {
        ToolExecutor { registry }
    }

    pub async fn execute(
        &self,
        name: &str,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> ToolResult {
        let tool = {
            let reg = self.registry.lock().await;
            reg.get(name)
        };

        match tool {
            None => ToolResult {
                data: format!("Tool '{}' not found.", name),
                is_error: true,
            },
            Some(t) => match t.execute(input, context).await {
                Ok(result) => result,
                Err(e) => ToolResult {
                    data: format!("Tool '{}' error: {}", name, e),
                    is_error: true,
                },
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::types::{AgentInfo, ToolResult, ToolUseContext};
    use serde_json::json;

    // Minimal echo tool used in tests
    struct EchoTool;
    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echoes input back" }
        fn input_schema(&self) -> serde_json::Value {
            json!({"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]})
        }
        async fn execute(&self, input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext) -> Result<ToolResult> {
            let msg = input.get("msg").and_then(|v| v.as_str()).unwrap_or("").to_string();
            Ok(ToolResult { data: msg, is_error: false })
        }
    }

    struct ErrorTool;
    #[async_trait::async_trait]
    impl Tool for ErrorTool {
        fn name(&self) -> &str { "bad" }
        fn description(&self) -> &str { "Always errors" }
        fn input_schema(&self) -> serde_json::Value { json!({}) }
        async fn execute(&self, _input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext) -> Result<ToolResult> {
            Err(crate::error::AgentError::Other("intentional error".to_string()))
        }
    }

    fn ctx() -> ToolUseContext {
        ToolUseContext {
            agent: AgentInfo { name: "t".to_string(), role: "r".to_string(), model: "m".to_string() },
            cwd: None,
        }
    }

    // -------------------------------------------------------------------------
    // ToolRegistry
    // -------------------------------------------------------------------------

    #[test]
    fn register_and_get() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        assert!(reg.get("echo").is_some());
    }

    #[test]
    fn register_duplicate_fails() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        assert!(reg.register(Arc::new(EchoTool)).is_err());
    }

    #[test]
    fn get_missing_returns_none() {
        let reg = ToolRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn has_returns_correct_bool() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        assert!(reg.has("echo"));
        assert!(!reg.has("bad"));
    }

    #[test]
    fn unregister_removes_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        reg.unregister("echo");
        assert!(!reg.has("echo"));
    }

    #[test]
    fn unregister_missing_is_noop() {
        let mut reg = ToolRegistry::new();
        reg.unregister("nonexistent"); // must not panic
    }

    #[test]
    fn list_returns_all_tools() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        reg.register(Arc::new(ErrorTool)).unwrap();
        assert_eq!(reg.list().len(), 2);
    }

    #[test]
    fn to_tool_defs_all() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        let defs = reg.to_tool_defs(None);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }

    #[test]
    fn to_tool_defs_filtered() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        reg.register(Arc::new(ErrorTool)).unwrap();
        let defs = reg.to_tool_defs(Some(&["echo".to_string()]));
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }

    // -------------------------------------------------------------------------
    // ToolExecutor
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn executor_runs_tool_successfully() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).unwrap();
        let reg = Arc::new(tokio::sync::Mutex::new(reg));
        let exec = ToolExecutor::new(reg);

        let mut input = HashMap::new();
        input.insert("msg".to_string(), json!("hello"));
        let result = exec.execute("echo", &input, &ctx()).await;
        assert!(!result.is_error);
        assert_eq!(result.data, "hello");
    }

    #[tokio::test]
    async fn executor_returns_error_for_missing_tool() {
        let reg = Arc::new(tokio::sync::Mutex::new(ToolRegistry::new()));
        let exec = ToolExecutor::new(reg);
        let result = exec.execute("nope", &HashMap::new(), &ctx()).await;
        assert!(result.is_error);
        assert!(result.data.contains("not found"));
    }

    #[tokio::test]
    async fn executor_wraps_tool_error() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(ErrorTool)).unwrap();
        let reg = Arc::new(tokio::sync::Mutex::new(reg));
        let exec = ToolExecutor::new(reg);
        let result = exec.execute("bad", &HashMap::new(), &ctx()).await;
        assert!(result.is_error);
        assert!(result.data.contains("intentional error"));
    }
}
