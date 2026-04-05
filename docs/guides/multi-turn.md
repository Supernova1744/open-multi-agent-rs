# Guide: Multi-Turn Conversations

`Agent::prompt` preserves the full conversation history between calls, enabling
stateful dialogue where the model remembers everything said earlier in the session.

## How it works

Each call to `agent.prompt(message, adapter)` appends the new user message to the
existing history, runs the conversation loop, then appends the assistant reply.
The next call sees the entire history.

`agent.run(prompt, adapter)` does NOT preserve history — it starts fresh each time.

## Example

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use open_multi_agent::{
    agent::Agent,
    llm::create_adapter,
    tool::{ToolRegistry, ToolExecutor},
    AgentConfig,
};

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let adapter = Arc::from(create_adapter("openrouter", None, None));
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

    let mut agent = Agent::new(
        AgentConfig {
            name: "tutor".to_string(),
            system_prompt: Some("You are a patient Rust tutor.".to_string()),
            max_turns: Some(5),
            ..Default::default()
        },
        registry,
        executor,
    );

    // Turn 1
    let r1 = agent.prompt("What is ownership in Rust?", Arc::clone(&adapter)).await.unwrap();
    println!("Turn 1: {}", r1.output);

    // Turn 2 — agent remembers turn 1
    let r2 = agent.prompt("Can you give a short code example?", Arc::clone(&adapter)).await.unwrap();
    println!("Turn 2: {}", r2.output);

    // Turn 3 — agent remembers turns 1 and 2
    let r3 = agent.prompt("What happens if I clone instead?", Arc::clone(&adapter)).await.unwrap();
    println!("Turn 3: {}", r3.output);

    println!(
        "\nTotal tokens: {}in / {}out",
        r1.token_usage.input_tokens + r2.token_usage.input_tokens + r3.token_usage.input_tokens,
        r1.token_usage.output_tokens + r2.token_usage.output_tokens + r3.token_usage.output_tokens,
    );
}
```

## Inspecting and resetting history

```rust
// Read the full conversation
let history: Vec<LLMMessage> = agent.get_history();
println!("{} messages in history", history.len());

// Clear history to start a new conversation with the same agent instance
agent.reset();
assert!(agent.get_history().is_empty());
assert_eq!(agent.status, AgentStatus::Idle);
```

## Token growth

Each turn sends the entire conversation to the provider — token costs grow linearly
with history length. Two mitigations:

1. **`max_turns`** hard-stops the inner loop per `prompt()` call (not across calls).
   Set it small for prompt-style conversations.
2. Call `agent.reset()` between topics to discard old context.

## Tracking cumulative usage

Each `AgentRunResult` contains the usage for that single `prompt()` call only.
Accumulate across turns yourself:

```rust
let mut total = TokenUsage::default();
for turn_result in &results {
    total = total.add(&turn_result.token_usage);
}
```
