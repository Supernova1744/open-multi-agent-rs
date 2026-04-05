# Guide: Single Agent

The simplest way to run a model is through the `OpenMultiAgent` orchestrator, which
handles adapter construction, retries on transient errors, and lifecycle hooks for you.

## Minimal example

```rust
use open_multi_agent::{OrchestratorConfig, OpenMultiAgent, AgentConfig};

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "claude-opus-4-6".to_string(),
        default_provider: "anthropic".to_string(),
        default_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        max_concurrency: 1,
        ..Default::default()
    });

    let config = AgentConfig {
        name: "assistant".to_string(),
        system_prompt: Some("You are a concise assistant. Answer in one sentence.".to_string()),
        max_turns: Some(3),
        ..Default::default()
    };

    let result = orchestrator
        .run_agent(config, "What is the capital of France?")
        .await
        .expect("agent failed");

    println!("Output: {}", result.output);
    println!("Turns:  {}", result.turns);
    println!("Tokens: {}in / {}out",
        result.token_usage.input_tokens,
        result.token_usage.output_tokens,
    );
}
```

## Using a different provider

Set `provider` and the matching API key on `AgentConfig` to override the orchestrator
defaults for a single agent:

```rust
let config = AgentConfig {
    name: "openai-agent".to_string(),
    model: "gpt-4o".to_string(),
    provider: Some("openai".to_string()),
    api_key: Some(std::env::var("OPENAI_API_KEY").unwrap()),
    ..Default::default()
};
```

## Low-level: building Agent directly

When you need finer control — custom tool registry, streaming, or multi-turn history —
construct `Agent` directly and build your own adapter:

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use open_multi_agent::{
    agent::Agent,
    llm::create_adapter,
    tool::{ToolRegistry, ToolExecutor},
    AgentConfig,
};

let adapter = Arc::from(create_adapter("openrouter", None, None));

let registry = Arc::new(Mutex::new(ToolRegistry::new()));
let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

let mut agent = Agent::new(
    AgentConfig {
        name: "worker".to_string(),
        ..Default::default()
    },
    registry,
    executor,
);

let result = agent.run("Explain Rust lifetimes.", Arc::clone(&adapter)).await?;
println!("{}", result.output);
```

## `AgentRunResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | `true` if the agent produced output without error |
| `output` | `String` | Final text from the model |
| `messages` | `Vec<LLMMessage>` | All messages added during this run |
| `token_usage` | `TokenUsage` | Accumulated across all turns |
| `tool_calls` | `Vec<ToolCallRecord>` | Audit log of every tool call |
| `turns` | `usize` | Number of LLM round-trips |
| `structured` | `Option<Value>` | Parsed JSON when `output_schema` is set |

## Configuration reference

| `AgentConfig` field | Default | Notes |
|---------------------|---------|-------|
| `name` | `"agent"` | Appears in trace events |
| `model` | `"qwen/qwen3.6-plus:free"` | Model string forwarded to the provider |
| `provider` | `None` → orchestrator default | `"anthropic"`, `"openai"`, or `"openrouter"` |
| `system_prompt` | `None` | Injected before the first user message |
| `max_turns` | `Some(10)` | Hard stop; last assistant text is used as output |
| `max_tokens` | `None` | Provider's default if omitted |
| `temperature` | `None` | Provider's default if omitted |
| `tools` | `None` (all registered) | Allowlist of tool names |
| `output_schema` | `None` | Enables structured output mode |
