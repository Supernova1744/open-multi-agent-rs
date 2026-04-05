# open-multi-agent-rs

A Rust port of [open-multi-agent](https://github.com/JackChen-me/open-multi-agent) — a framework for orchestrating multi-agent LLM workflows with dependency-aware task scheduling, shared memory, streaming, observability hooks, and parallel execution.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [API Keys](#api-keys)
  - [LLM Providers](#llm-providers)
- [Running the Demo](#running-the-demo)
- [Running Tests](#running-tests)
  - [All Tests](#all-tests)
  - [Unit Tests Only](#unit-tests-only)
  - [Integration Tests](#integration-tests)
  - [Stress Tests](#stress-tests)
  - [Specific Test by Name](#specific-test-by-name)
- [Using the Library](#using-the-library)
  - [Single Agent](#single-agent)
  - [Multi-Turn Conversation](#multi-turn-conversation)
  - [Task Pipeline](#task-pipeline)
  - [Team with Coordinator](#team-with-coordinator)
  - [Streaming](#streaming)
  - [Tools](#tools)
  - [Structured Output](#structured-output)
  - [Lifecycle Hooks](#lifecycle-hooks)
  - [Trace / Observability](#trace--observability)
  - [MessageBus](#messagebus)
  - [Retry and Backoff](#retry-and-backoff)
  - [Approval Gates](#approval-gates)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Minimum Version | Install |
|------|----------------|---------|
| Rust | 1.70+ | [rustup.rs](https://rustup.rs) |
| Cargo | ships with Rust | — |

Verify your installation:

```bash
rustc --version
cargo --version
```

No other system dependencies are required. All Rust crate dependencies are fetched automatically by Cargo.

---

## Project Structure

```
open-multi-agent-rs/
├── Cargo.toml                   # Workspace manifest & dependencies
├── src/
│   ├── lib.rs                   # Public API re-exports
│   ├── main.rs                  # Demo binary (cargo run --bin demo)
│   ├── types.rs                 # All shared types
│   ├── error.rs                 # AgentError + Result alias
│   ├── trace.rs                 # Trace event helpers (emit_trace, now_ms)
│   ├── messaging.rs             # MessageBus pub/sub system
│   ├── agent/
│   │   ├── mod.rs               # Agent high-level wrapper + hooks + structured output
│   │   ├── runner.rs            # AgentRunner — core turn loop + streaming
│   │   └── pool.rs              # AgentPool + Semaphore for concurrency limiting
│   ├── llm/
│   │   ├── mod.rs               # create_adapter() factory
│   │   ├── anthropic.rs         # Anthropic Messages API adapter
│   │   ├── openai.rs            # OpenAI adapter (thin wrapper)
│   │   └── openrouter.rs        # OpenRouter / OpenAI-compatible adapter
│   ├── orchestrator/
│   │   └── mod.rs               # OpenMultiAgent, execute_with_retry, compute_retry_delay
│   ├── task/
│   │   ├── mod.rs               # Task CRUD + topological sort
│   │   ├── queue.rs             # TaskQueue — event-driven dependency queue
│   │   └── scheduler.rs        # Scheduler — round-robin, least-busy, etc.
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── store.rs             # InMemoryStore
│   │   └── shared.rs           # SharedMemory — namespaced cross-agent memory
│   └── tool/
│       └── mod.rs               # ToolRegistry + ToolExecutor
└── tests/
    ├── mock_adapter.rs          # Shared deterministic mock LLM adapter
    ├── integration_tests.rs     # Core integration tests
    ├── new_features_integration.rs  # v2 feature tests (streaming, hooks, trace, etc.)
    └── stress_tests.rs          # Concurrency + throughput stress tests
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Supernova1744/open-multi-agent-rs
cd open-multi-agent-rs

# Build (downloads all dependencies automatically)
cargo build

# Run all tests (no API key required — uses mock adapter)
cargo test

# Run the live demo (requires an API key — see Configuration below)
cargo run --bin demo
```

---

## Configuration

### API Keys

The demo binary and live adapter tests read API keys either from constants in `src/main.rs` or from environment variables.

**Environment variables** (preferred for security):

| Variable | Provider |
|----------|----------|
| `OPENROUTER_API_KEY` | OpenRouter (default in demo) |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `OPENAI_API_KEY` | OpenAI |

Set them in your shell before running:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
cargo run --bin demo
```

Or on Windows (PowerShell):

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
cargo run --bin demo
```

> **Note:** The demo reads `OPENROUTER_API_KEY` from the environment and will panic with a clear message if it is not set.

### LLM Providers

Three providers are supported out of the box:

| Provider string | Adapter | Default base URL |
|-----------------|---------|-----------------|
| `"openrouter"` | `OpenRouterAdapter` | `https://openrouter.ai/api/v1` |
| `"anthropic"` | `AnthropicAdapter` | `https://api.anthropic.com` |
| `"openai"` | `OpenAIAdapter` | `https://api.openai.com/v1` |

Create an adapter programmatically:

```rust
use open_multi_agent::create_adapter;

let adapter = create_adapter(
    "openrouter",
    Some("sk-or-v1-...".to_string()),
    Some("https://openrouter.ai/api/v1".to_string()),
);
```

Any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio) works with the `"openrouter"` or `"openai"` provider and a custom `base_url`.

---

## Running the Demo

The demo runs four scenarios against a live LLM and prints results to stdout:

1. **Single agent** — one-shot question answering
2. **Multi-turn conversation** — stateful back-and-forth
3. **Task pipeline** — two tasks with a dependency (Research → Write)
4. **Team with coordinator** — LLM decomposes a goal into tasks automatically

```bash
cargo run --bin demo
```

Expected output (timing and content will vary):

```
open-multi-agent-rs — Rust port demo
Using model: qwen/qwen3.6-plus:free
Using API: OpenRouter

============================================================
TEST 1: Single Agent
============================================================
Prompt: What is the capital of France? Answer in one word.
Success: true
Output: Paris
Tokens: input=32, output=3

============================================================
TEST 2: Multi-Turn Conversation
...
```

---

## Running Tests

All tests use a deterministic in-memory mock adapter — **no API key or network access is required**.

### All Tests

```bash
cargo test
```

This runs all test suites in parallel and prints a summary:

```
test result: ok. 125 passed; 0 failed   (lib unit tests)
test result: ok. 50 passed; 0 failed    (new_features_integration)
test result: ok. 20 passed; 0 failed    (stress_tests)
test result: ok. 16 passed; 0 failed    (integration_tests)
```

### Unit Tests Only

Unit tests live alongside the source code in `#[cfg(test)]` modules:

```bash
cargo test --lib
```

Key unit test modules:

| Module | What is tested |
|--------|---------------|
| `llm::anthropic::tests` | Wire-format serialization (15 tests) |
| `agent::mod::tests` | JSON extraction, schema validation (10 tests) |
| `orchestrator::tests` | `compute_retry_delay`, `execute_with_retry` (8 tests) |
| `trace::tests` | `emit_trace`, `generate_run_id`, `now_ms` (4 tests) |
| `messaging::tests` | MessageBus send/receive/subscribe (8 tests) |
| `task::tests` | Task creation, dependency resolution, topo-sort (11 tests) |
| `tool::tests` | ToolRegistry, ToolExecutor (11 tests) |
| `types::tests` | Type helpers and serialization (9 tests) |

### Integration Tests

```bash
cargo test --test integration_tests
cargo test --test new_features_integration
```

`new_features_integration` covers all v2 features — 50 tests across:

- **Streaming** — `StreamEvent` sequence, `Done` result, tool round-trips
- **before_run hooks** — prompt mutation, abort, multi-call
- **after_run hooks** — output mutation, abort
- **Structured output** — valid JSON, fenced blocks, schema validation, retry-on-failure
- **Trace events** — `LlmCall`, `ToolCall`, `Agent` traces, panic safety, run_id forwarding
- **RunOptions callbacks** — `on_tool_call`, `on_message`
- **MessageBus** — send, broadcast, mark-read, subscribe/unsubscribe, shared state across clones
- **Approval gates** — approve, reject with `skip_remaining`
- **Retry** — immediate success, retry-then-succeed, exhaust retries, token accumulation, `on_retry` callback, exponential delay
- **Parallel tool calls** — `MultiToolCall` response → three concurrent tool executions
- **Adapter construction** — `AnthropicAdapter`, `OpenAIAdapter`, `OpenRouterAdapter`, `create_adapter` factory

### Stress Tests

```bash
cargo test --test stress_tests
```

20 stress tests that verify correctness and stability under high concurrency:

| Test | Scenario |
|------|---------|
| 1–5 | 50–200 concurrent agent runs, large task queues |
| 6–12 | Concurrent memory access, rapid queue ops, scheduler at scale |
| 13 | 30 concurrent streams, each completes correctly |
| 14 | 50 agents × 4 trace events = 200 total events collected |
| 15 | MessageBus: 100 senders × 10 receiver subscribers |
| 16 | MessageBus: 50 broadcasts × 20 subscribers |
| 17 | 100 tasks × 2 retries each (fail, fail, succeed) = 300 LLM calls |
| 18 | 10 000 `compute_retry_delay` calls complete in < 10ms |
| 19 | 20 concurrent subscribe/receive/unsubscribe cycles |
| 20 | `skip_remaining` on 500-task queue completes in < 100ms |

### Specific Test by Name

Run a single test by passing its name (substring match):

```bash
# Run one specific test
cargo test retry_retries_on_error_and_succeeds

# Run all retry-related tests
cargo test retry

# Run all MessageBus tests
cargo test message_bus

# Run all streaming tests
cargo test streaming

# Show stdout even for passing tests
cargo test retry -- --nocapture
```

---

## Using the Library

Add to your `Cargo.toml` (when used as a local path dependency):

```toml
[dependencies]
open-multi-agent-rs = { path = "../open-multi-agent-rs" }
tokio = { version = "1", features = ["full"] }
```

### Single Agent

```rust
use open_multi_agent::{AgentConfig, OrchestratorConfig, OpenMultiAgent};

#[tokio::main]
async fn main() {
    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "claude-opus-4-6".to_string(),
        default_provider: "anthropic".to_string(),
        default_api_key: Some(std::env::var("ANTHROPIC_API_KEY").unwrap()),
        default_base_url: None,
        max_concurrency: 4,
        on_progress: None,
        on_trace: None,
        on_approval: None,
    });

    let config = AgentConfig {
        name: "assistant".to_string(),
        model: "claude-opus-4-6".to_string(),
        system_prompt: Some("You are a helpful assistant.".to_string()),
        ..Default::default()
    };

    let result = orchestrator
        .run_agent(config, "What is 2 + 2?")
        .await
        .unwrap();

    println!("{}", result.output);
}
```

### Multi-Turn Conversation

`agent.prompt()` preserves conversation history across calls:

```rust
use open_multi_agent::{agent::Agent, AgentConfig, create_adapter, ToolRegistry, ToolExecutor};
use std::sync::Arc;
use tokio::sync::Mutex;

let registry = Arc::new(Mutex::new(ToolRegistry::new()));
let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
let adapter = Arc::from(create_adapter("anthropic", Some(api_key), None));

let config = AgentConfig {
    name: "tutor".to_string(),
    model: "claude-opus-4-6".to_string(),
    ..Default::default()
};

let mut agent = Agent::new(config, registry, executor);

let r1 = agent.prompt("What is 2 + 2?", Arc::clone(&adapter)).await?;
println!("{}", r1.output); // "4"

let r2 = agent.prompt("Now double it.", Arc::clone(&adapter)).await?;
println!("{}", r2.output); // "8" — agent remembers previous context
```

### Task Pipeline

Define tasks with explicit dependencies. The orchestrator resolves them in topological order:

```rust
use open_multi_agent::{create_task, TeamConfig, OpenMultiAgent, OrchestratorConfig};

let orchestrator = OpenMultiAgent::new(OrchestratorConfig { /* ... */ });

let team = TeamConfig {
    name: "pipeline".to_string(),
    agents: vec![
        make_agent("researcher", "Research topics concisely."),
        make_agent("writer", "Write clear summaries based on research."),
    ],
    shared_memory: Some(true),
    max_concurrency: Some(2),
};

let task1 = create_task(
    "Research topic",
    "Summarize the key points of Rust ownership in 3 bullets.",
    Some("researcher".to_string()),
    vec![],  // no dependencies
);

let task2 = create_task(
    "Write article",
    "Using the research, write a two-paragraph blog post.",
    Some("writer".to_string()),
    vec![task1.id.clone()],  // depends on task1
);

let result = orchestrator.run_tasks(&team, vec![task1, task2]).await?;
println!("Success: {}", result.success);
for (task_id, agent_result) in &result.agent_results {
    println!("Task {}: {}", task_id, agent_result.output);
}
```

### Team with Coordinator

The orchestrator uses an LLM to decompose a high-level goal into tasks automatically:

```rust
let result = orchestrator.run_team(&team, "Write a blog post about Rust ownership.").await?;

if let Some(final_output) = result.agent_results.get("coordinator") {
    println!("{}", final_output.output);
}
```

### Streaming

```rust
use futures::StreamExt;
use open_multi_agent::{agent::Agent, types::StreamEvent, create_adapter};

let adapter = Arc::from(create_adapter("anthropic", Some(api_key), None));
let stream = agent.stream("Tell me a short story.", adapter);
tokio::pin!(stream);

while let Some(event) = stream.next().await {
    match event {
        StreamEvent::Text { text } => print!("{}", text),
        StreamEvent::ToolUse { name, .. } => println!("\n[calling tool: {}]", name),
        StreamEvent::ToolResult { content, .. } => println!("[result: {}]", content),
        StreamEvent::Done { result } => {
            println!("\nFinished. Tokens: {}", result.token_usage.output_tokens);
            break;
        }
        StreamEvent::Error { message } => {
            eprintln!("Error: {}", message);
            break;
        }
    }
}
```

### Tools

Register tools on the agent's `ToolRegistry`. The agent calls them automatically during its turn loop:

```rust
use open_multi_agent::{Tool, ToolRegistry, types::LLMToolDef};
use std::sync::Arc;

let tool = Arc::new(MyTool); // implements the Tool trait
registry.lock().await.register(tool)?;

// Tool trait:
// async fn execute(&self, input: HashMap<String, Value>, context: &ToolUseContext) -> ToolResult
// fn definition(&self) -> LLMToolDef
```

### Structured Output

Set `output_schema` on `AgentConfig` to instruct the agent to respond in JSON matching a schema. If the response doesn't match, the framework retries once with error feedback:

```rust
use open_multi_agent::AgentConfig;

let config = AgentConfig {
    name: "extractor".to_string(),
    model: "claude-opus-4-6".to_string(),
    output_schema: Some(serde_json::json!({
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": { "type": "string" },
            "age":  { "type": "number" }
        }
    })),
    ..Default::default()
};

let result = orchestrator.run_agent(config, "Extract: Alice is 30 years old.").await?;
if let Some(structured) = result.structured {
    println!("{}", structured["name"]); // "Alice"
    println!("{}", structured["age"]);  // 30
}
```

### Lifecycle Hooks

`before_run` fires before each agent call (can modify the prompt or abort). `after_run` fires after (can modify the output or abort):

```rust
use open_multi_agent::types::{AgentConfig, BeforeRunHookContext, AgentRunResult};
use futures::future::BoxFuture;
use std::sync::Arc;

let config = AgentConfig {
    before_run: Some(Arc::new(|mut ctx: BeforeRunHookContext| -> BoxFuture<'static, Result<BeforeRunHookContext, String>> {
        Box::pin(async move {
            // Append context to every prompt
            ctx.prompt = format!("{}\n\nBe concise.", ctx.prompt);
            Ok(ctx)
        })
    })),
    after_run: Some(Arc::new(|mut result: AgentRunResult| -> BoxFuture<'static, Result<AgentRunResult, String>> {
        Box::pin(async move {
            result.output = result.output.trim().to_string();
            Ok(result)
        })
    })),
    ..Default::default()
};
```

To abort from a hook, return `Err(reason)` — the agent immediately returns a failed `AgentRunResult`.

### Trace / Observability

Attach an `on_trace` callback to `OrchestratorConfig` to receive structured spans for every LLM call, tool call, and agent run:

```rust
use open_multi_agent::{OrchestratorConfig, types::TraceEvent, OnTraceFn};
use std::sync::Arc;

let config = OrchestratorConfig {
    on_trace: Some(Arc::new(|event: TraceEvent| {
        match &event {
            TraceEvent::LlmCall(t) => {
                println!("[trace] llm_call model={} tokens_in={} duration={}ms",
                    t.model, t.tokens.input_tokens, t.base.duration_ms);
            }
            TraceEvent::ToolCall(t) => {
                println!("[trace] tool_call tool={} is_error={}", t.tool, t.is_error);
            }
            TraceEvent::Agent(t) => {
                println!("[trace] agent={} turns={} total_tokens={}",
                    t.base.agent, t.turns,
                    t.tokens.input_tokens + t.tokens.output_tokens);
            }
            TraceEvent::Task(t) => {
                println!("[trace] task={} success={} retries={}",
                    t.task_title, t.success, t.retries);
            }
        }
    })),
    // ...other fields
};
```

Panics inside the callback are caught and do not affect execution.

### MessageBus

An in-process pub/sub bus for agent-to-agent communication. Cloning a `MessageBus` shares the same underlying state:

```rust
use open_multi_agent::MessageBus;

let bus = MessageBus::new();

// Point-to-point
bus.send("alice", "bob", "Hello Bob!");

// Broadcast to all subscribers
bus.broadcast("system", "Shutdown in 5s");

// Read unread messages for an agent
for msg in bus.get_unread("bob") {
    println!("{}: {}", msg.from, msg.content);
}
bus.mark_read("bob");

// Subscribe (returns an unsubscribe closure)
let unsub = bus.subscribe("alice", |msg| {
    println!("alice received: {}", msg.content);
});
// Later:
unsub();
```

### Retry and Backoff

Per-task retry is configured directly on each `Task`:

```rust
use open_multi_agent::create_task;

let mut task = create_task("title", "description", Some("agent".to_string()), vec![]);
task.max_retries = Some(3);       // retry up to 3 times
task.retry_delay_ms = Some(500);  // base delay 500ms
task.retry_backoff = Some(2.0);   // exponential: 500 → 1000 → 2000ms (capped at 30s)
```

Use `execute_with_retry` directly for custom retry loops:

```rust
use open_multi_agent::{execute_with_retry, AgentRunResult, types::TokenUsage};
use std::sync::Arc;

let result = execute_with_retry(
    || Box::pin(async { /* your async fn */ }),
    &task,
    Some(Arc::new(|attempt: u32, max: u32, err: String, delay_ms: u64| {
        println!("Retry {}/{} after {}ms: {}", attempt, max, delay_ms, err);
    })),
).await;
```

### Approval Gates

Pause execution between task rounds for human or automated review:

```rust
use open_multi_agent::{OrchestratorConfig, types::Task};
use futures::future::BoxFuture;
use std::sync::Arc;

let config = OrchestratorConfig {
    on_approval: Some(Arc::new(|completed: Vec<Task>, pending: Vec<Task>| -> BoxFuture<'static, bool> {
        Box::pin(async move {
            println!("{} tasks done, {} pending. Approve? (auto: yes)", completed.len(), pending.len());
            true  // return false to skip all remaining tasks
        })
    })),
    ..Default::default()
};
```

---

## Architecture Overview

```
OpenMultiAgent (orchestrator)
  │
  ├── run_agent()          — single agent, one-shot
  ├── run_team()           — coordinator decomposes goal → TaskQueue
  └── run_tasks()          — explicit task pipeline
          │
          ▼
      TaskQueue (dependency graph, event-driven)
          │
      Scheduler (round-robin / least-busy / dependency-first)
          │
      AgentPool (concurrency semaphore, max N simultaneous)
          │
      execute_with_retry() (exponential backoff, token accumulation)
          │
      Agent
        ├── before_run hook
        ├── AgentRunner (turn loop)
        │     ├── LLMAdapter.chat()       ← Anthropic / OpenAI / OpenRouter
        │     └── ToolExecutor.execute()  ← parallel tool calls per turn
        ├── Structured output validation (+ 1 retry)
        └── after_run hook
```

Data flows between agents via `SharedMemory` — each task result is written to the shared store and injected as context into the next task's prompt.

---

## Troubleshooting

**`cargo build` fails with linker errors on Windows**

Install the MSVC build tools or use the GNU toolchain:

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --target x86_64-pc-windows-gnu
```

**Demo fails with `HTTP 401`**

Your API key is invalid or expired. Make sure `OPENROUTER_API_KEY` is set to a valid key before running.

**Demo fails with `HTTP 429`**

Rate limit exceeded on the free-tier model. Wait a moment and retry, or switch to a paid model/plan.

**Tests hang indefinitely**

The stress tests use `tokio::time::sleep` with near-zero delays (1ms). If your system clock resolution is low, set `RUST_TEST_THREADS=1` to run tests serially:

```bash
RUST_TEST_THREADS=1 cargo test --test stress_tests
```

**`error: no such command: demo`**

Use the full form:

```bash
cargo run --bin demo    # NOT cargo demo
```

**Changing the model**

Edit the `MODEL` constant in `src/main.rs`:

```rust
const MODEL: &str = "claude-opus-4-6";          // Anthropic via OpenRouter
const MODEL: &str = "openai/gpt-4o";            // OpenAI via OpenRouter
const MODEL: &str = "meta-llama/llama-3.3-70b-instruct:free"; // Free tier
```

To use a local model (Ollama):

```rust
const OPENROUTER_BASE_URL: &str = "http://localhost:11434/v1";
const MODEL: &str = "llama3.2";
```
