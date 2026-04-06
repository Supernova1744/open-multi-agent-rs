# Guide: Custom Tools

Tools let the agent take actions in the real world — query a database, call an API,
run a calculation, read a file — and receive structured results back before continuing.

## Implementing the `Tool` trait

```rust
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use open_multi_agent_rs::{
    error::Result,
    tool::{ToolRegistry, ToolExecutor},
    types::{ToolResult, ToolUseContext},
};
use open_multi_agent_rs::tool::Tool;

struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn name(&self) -> &str { "word_count" }

    fn description(&self) -> &str {
        "Count the number of words in a given text passage."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in."
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
        let text = input
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let count = text.split_whitespace().count();
        Ok(ToolResult {
            data: count.to_string(),
            is_error: false,
        })
    }
}
```

## Registering tools

```rust
use tokio::sync::Mutex;

let registry = Arc::new(Mutex::new(ToolRegistry::new()));

{
    let mut reg = registry.lock().await;
    reg.register(Arc::new(WordCount))?;
}

let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
```

Pass the same `registry` and `executor` to every `Agent` that should have access.

## Running an agent with tools

```rust
let mut agent = Agent::new(
    AgentConfig {
        name: "analyst".to_string(),
        system_prompt: Some(
            "You are a text analyst. Use word_count when asked about word counts.".to_string()
        ),
        // Optionally restrict to specific tools:
        tools: Some(vec!["word_count".to_string()]),
        ..Default::default()
    },
    Arc::clone(&registry),
    Arc::clone(&executor),
);

let result = agent
    .run("How many words are in: 'the quick brown fox'?", adapter)
    .await?;

println!("Tool calls made: {}", result.tool_calls.len());
for tc in &result.tool_calls {
    println!("  {} → {} ({}ms)", tc.tool_name, tc.output, tc.duration_ms);
}
```

## Input schema best practices

The JSON Schema is forwarded verbatim to the model. Write clear `description` strings
— models use them to understand when and how to call the tool.

```rust
fn input_schema(&self) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT query to execute (read-only)."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows to return. Default 10, max 100.",
                "default": 10
            }
        },
        "required": ["query"]
    })
}
```

## Returning errors

Set `is_error: true` in `ToolResult` to signal failure. The agent will be informed
that the tool failed and can adapt its strategy:

```rust
async fn execute(&self, input: &HashMap<String, serde_json::Value>, _ctx: &ToolUseContext) -> Result<ToolResult> {
    match do_work(input) {
        Ok(output) => Ok(ToolResult { data: output, is_error: false }),
        Err(e)     => Ok(ToolResult { data: e.to_string(), is_error: true }),
    }
}
```

Return `Err(AgentError::…)` only for unrecoverable infrastructure failures (the
executor will convert it to a `ToolResult { is_error: true }` anyway, but returning
`Ok(ToolResult { is_error: true })` gives you control over the error message shown
to the model).

## Restricting tool access per agent

Use `AgentConfig.tools` to give different agents different subsets:

```rust
// Research agent gets web search only
let researcher = AgentConfig {
    tools: Some(vec!["web_search".to_string()]),
    ..Default::default()
};

// Writer agent gets file system tools only
let writer = AgentConfig {
    tools: Some(vec!["read_file".to_string(), "write_file".to_string()]),
    ..Default::default()
};
```

`None` means the agent sees all registered tools.

## Tool execution context

`ToolUseContext` carries information about the calling agent:

```rust
pub struct ToolUseContext {
    pub agent: AgentInfo,  // name, role, model
    pub cwd: Option<String>,
}
```

Use `context.agent.name` to implement per-agent authorization in a shared tool.

## Observing tool calls

Attach callbacks to `RunOptions`:

```rust
let opts = RunOptions {
    on_tool_call: Some(Arc::new(|name, input| {
        println!("[tool] calling {} with {:?}", name, input);
    })),
    on_tool_result: Some(Arc::new(|name, is_err| {
        println!("[tool] {} finished (error={})", name, is_err);
    })),
    ..Default::default()
};
agent.run_with_opts(prompt, adapter, opts).await?;
```
