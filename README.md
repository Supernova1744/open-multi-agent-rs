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
- [Built-in Tools](#built-in-tools)
  - [File Management](#file-management-tools)
  - [Python Coding](#python-coding-tools)
  - [Repository Ingestion](#repository-ingestion-tool)
  - [HTTP / Networking](#http--networking-tools)
  - [Data Processing](#data-processing-tools)
  - [Math & Expressions](#math--expressions-tool)
  - [Date & Time](#date--time-tool)
  - [Text Processing](#text-processing-tools)
  - [Environment & System](#environment--system-tools)
  - [Cache](#cache-tools)
  - [Encoding & Hashing](#encoding--hashing-tools)
  - [Web Researcher](#web-researcher-tool)
  - [Search Engine](#search-engine-tool)
  - [Structured Data Parser](#structured-data-parser-tool)
  - [Knowledge Base (RAG)](#knowledge-base-rag-tools)
  - [MessageBus Tools](#messagebus-tools)
  - [Shell & Search](#shell--search-tools)
  - [Utility Tools](#utility-tools)
- [Feedback Loop](#feedback-loop)
- [Examples](#examples)
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
│       ├── mod.rs               # ToolRegistry + ToolExecutor
│       └── built_in.rs          # 45 built-in tools (file, python, HTTP, data, RAG, utility, …)
│   └── feedback.rs              # FeedbackLoop — iterative worker ↔ critic cycle
├── examples/
│   ├── 01_single_agent.rs       # One-shot agent
│   ├── 02_multi_turn_chat.rs    # Multi-turn conversation
│   ├── 03_streaming.rs          # Token-by-token streaming
│   ├── 04_custom_tool.rs        # Custom Tool implementation
│   ├── 05_structured_output.rs  # JSON schema output
│   ├── 06_task_pipeline.rs      # Dependency-aware pipeline
│   ├── 07_multi_agent_system.rs # Multi-agent team
│   ├── 08_hooks_and_trace.rs    # Lifecycle hooks + observability
│   ├── 09_message_bus.rs        # Agent-to-agent messaging
│   ├── 10_retry_and_approval.rs # Retry backoff + approval gates
│   ├── 11_python_coding_agent.rs# Python write/run/test workflow
│   ├── 12_repo_mindmap.rs       # Repo ingestion + Mermaid mindmap
│   ├── 13_http_tools.rs         # HTTP GET/POST + JSON processing
│   ├── 14_data_tools.rs         # CSV, math eval, datetime, regex, chunking
│   ├── 15_system_tools.rs       # System info, env, Base64, hash, cache
│   ├── 16_web_search.rs         # web_fetch HTML→MD, Tavily search, schema_validate
│   ├── 17_rag_knowledge_base.rs # In-process RAG (rag_add/search/clear)
│   ├── 18_bus_agents.rs         # Two agents communicating over MessageBus
│   ├── 19_knowledge_base_pipeline.rs # Full Karpathy LLM knowledge-base pipeline
│   └── 20_utility_tools.rs      # sleep, random, template, diff, zip, git, url
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
use open_multi_agent_rs::create_adapter;

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
use open_multi_agent_rs::{AgentConfig, OrchestratorConfig, OpenMultiAgent};

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
use open_multi_agent_rs::{agent::Agent, AgentConfig, create_adapter, ToolRegistry, ToolExecutor};
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
use open_multi_agent_rs::{create_task, TeamConfig, OpenMultiAgent, OrchestratorConfig};

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
use open_multi_agent_rs::{agent::Agent, types::StreamEvent, create_adapter};

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

Register tools on the agent's `ToolRegistry`. The agent calls them automatically during its turn loop. Use `register_built_in_tools` to add all built-in tools at once (see [Built-in Tools](#built-in-tools)):

```rust
use open_multi_agent_rs::{Tool, ToolRegistry, types::LLMToolDef};
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
use open_multi_agent_rs::AgentConfig;

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
use open_multi_agent_rs::types::{AgentConfig, BeforeRunHookContext, AgentRunResult};
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
use open_multi_agent_rs::{OrchestratorConfig, types::TraceEvent, OnTraceFn};
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
use open_multi_agent_rs::MessageBus;

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
use open_multi_agent_rs::create_task;

let mut task = create_task("title", "description", Some("agent".to_string()), vec![]);
task.max_retries = Some(3);       // retry up to 3 times
task.retry_delay_ms = Some(500);  // base delay 500ms
task.retry_backoff = Some(2.0);   // exponential: 500 → 1000 → 2000ms (capped at 30s)
```

Use `execute_with_retry` directly for custom retry loops:

```rust
use open_multi_agent_rs::{execute_with_retry, AgentRunResult, types::TokenUsage};
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
use open_multi_agent_rs::{OrchestratorConfig, types::Task};
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

## Built-in Tools

Register all built-in tools with a single call:

```rust
use open_multi_agent_rs::tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry};
use std::sync::Arc;
use tokio::sync::Mutex;

let registry = Arc::new(Mutex::new(ToolRegistry::new()));
{
    let mut reg = registry.lock().await;
    register_built_in_tools(&mut reg).await;
}
let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
```

Give an agent access to a subset of tools via `AgentConfig::tools`:

```rust
let config = AgentConfig {
    tools: Some(vec![
        "file_read".to_string(),
        "file_write".to_string(),
        "python_run".to_string(),
    ]),
    ..Default::default()
};
```

All file and directory tools sandbox their paths to the current working directory. Paths that escape the sandbox are rejected.

---

### File Management Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `file_read` | `FileReadTool` | Read a file's contents. Input: `path`. |
| `file_write` | `FileWriteTool` | Write or overwrite a file. Input: `path`, `content`. |
| `file_update` | `FileUpdateTool` | Patch a file by literal replacement (`old`→`new`) or line-range replacement (`start_line`/`end_line`). Input: `path`, `old`, `new` OR `path`, `start_line`, `end_line`, `new_content`. |
| `file_delete` | `FileDeleteTool` | Delete a file. Input: `path`. |
| `file_list` | `FileListTool` | List directory contents. Input: `path`, optional `recursive` (bool). |
| `file_move` | `FileMoveTool` | Move or rename a file. Input: `src`, `dst`. |
| `dir_create` | `DirCreateTool` | Create a directory (including parents). Input: `path`. |
| `dir_delete` | `DirDeleteTool` | Delete a directory and all contents. Input: `path`. (Rejects root/sandbox root.) |

---

### Python Coding Tools

Requires Python 3 (`python3` or `python`) to be installed and on `PATH`.

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `python_write` | `PythonWriteTool` | Write a `.py` file and immediately syntax-check it with `py_compile`. Input: `path`, `code`. |
| `python_run` | `PythonRunTool` | Execute a Python script or an inline code snippet. Input: `path` (existing file) OR `code` (inline string). |
| `python_test` | `PythonTestTool` | Run `pytest` on a test file (`--tb=short --no-header`). Input: `path`. |

Example — agent-driven Python coding workflow (see `examples/11_python_coding_agent.rs`):

```rust
let config = AgentConfig {
    tools: Some(vec![
        "python_write".to_string(),
        "python_run".to_string(),
        "python_test".to_string(),
        "file_read".to_string(),
        "file_update".to_string(),
    ]),
    max_turns: Some(12),
    ..Default::default()
};
// Agent can now: write modules, run code, write tests, fix failures, re-run tests.
```

---

### Repository Ingestion Tool

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `repo_ingest` | `RepoIngestTool` | Walk a directory, detect languages, read key files, extract code outlines (functions, structs, classes), and return a rich Markdown analysis report. Input: `path`. |

The report includes:
- Language breakdown and file tree
- Priority files read in full (or up to 8 KB)
- Per-file code outlines: function/struct/class declarations extracted without regex
- Dependency files (`Cargo.toml`, `package.json`, `requirements.txt`, …)
- Overall statistics (files, languages, estimated lines)

Use this tool as the first step of a codebase analysis workflow. A downstream agent can then write a Mermaid mindmap, documentation, or architecture diagrams (see `examples/12_repo_mindmap.rs`).

```rust
// Two-step repo analysis:
// 1. repo_ingest  → rich Markdown analysis
// 2. Agent writes a Mermaid mindmap .md using file_write
let config = AgentConfig {
    tools: Some(vec![
        "repo_ingest".to_string(),
        "file_write".to_string(),
        "file_read".to_string(),
    ]),
    max_turns: Some(6),
    ..Default::default()
};
```

---

### HTTP / Networking Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `http_get` | `HttpGetTool` | HTTP GET request. Input: `url`, optional `headers`, `timeout_ms`. Response body capped at 4 MB. Follows up to 10 redirects. |
| `http_post` | `HttpPostTool` | HTTP POST request. Input: `url`, `body`, optional `content_type` (default: `application/json`), `headers`, `timeout_ms`. |

Both tools return `HTTP <status> <reason>\n\n<body>` and set `is_error` for non-2xx responses.

---

### Data Processing Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `json_parse` | `JsonParseTool` | Parse a JSON string and optionally extract a sub-value via JSON Pointer (e.g. `/users/0/name`) or pretty-print the whole document. Input: `input`, optional `pointer`, `pretty`. |
| `json_transform` | `JsonTransformTool` | Transform JSON with a simple operation: `keys`, `values`, `length`, `[/field]` (map array → extract field), or `/pointer` (extract sub-value). Input: `input`, `operation`. |
| `csv_read` | `CsvReadTool` | Read a CSV file and return rows as a JSON array of objects or a Markdown table. Input: `path`, optional `delimiter`, `has_headers`, `limit`, `format` (`json`\|`markdown`). |
| `csv_write` | `CsvWriteTool` | Write a JSON array of objects or arrays to a CSV file (headers inferred from first object's keys). Input: `path`, `data` (JSON array string), optional `delimiter`. |

---

### Math & Expressions Tool

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `math_eval` | `MathEvalTool` | Safely evaluate a mathematical expression. Supports arithmetic, `sqrt`, `abs`, `min`, `max`, `floor`, `ceil`, `round`, trig functions, comparisons, and optional variable bindings. No code execution. Input: `expression`, optional `variables` (JSON object of `name → number`). |

```rust
// Example: 3x² + y where x=4, y=7
// expression: "3 * x^2 + y", variables: {"x": 4.0, "y": 7.0} → 55
```

---

### Date & Time Tool

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `datetime` | `DatetimeTool` | Date/time operations. `operation` selects the mode: `now` (current UTC), `format` (Unix timestamp → string), `parse` (string → Unix timestamp), `diff` (seconds between two timestamps). |

```rust
// "now"    → "2024-01-15 10:30:00 UTC\ntimestamp: 1705313400"
// "format" → requires timestamp + format (strftime)
// "parse"  → requires input string; supports ISO 8601, RFC 3339, and common date formats
// "diff"   → requires timestamp + timestamp2; returns diff_seconds and human-readable breakdown
```

---

### Text Processing Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `text_regex` | `TextRegexTool` | Apply a regex pattern to text. Modes: `find_all` (returns JSON array of `{match, start, end}`), `replace` (replace all matches; supports `$1` capture groups), `split`. Input: `input`, `pattern`, optional `mode`, `replacement`. |
| `text_chunk` | `TextChunkTool` | Split large text into chunks for LLM context management. Split by `chars`, `words`, or `lines`, with configurable `chunk_size` and `overlap`. Returns a JSON array of strings. Input: `text`, optional `chunk_size`, `overlap`, `split_by`. |

---

### Environment & System Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `env_get` | `EnvGetTool` | Read an environment variable from a safe allowlist (`HOME`, `PATH`, `PORT`, `APP_ENV`, API keys, `REPO_PATH`, etc.). Returns the value or a configurable default. Input: `name`, optional `default`. |
| `system_info` | `SystemInfoTool` | Return OS family, OS name, architecture, CPU count, and current working directory as a JSON object. No inputs. |

The `env_get` allowlist prevents agents from reading arbitrary sensitive environment variables. To expand it, edit `ENV_ALLOWLIST` in `src/tool/built_in.rs`.

---

### Cache Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `cache_set` | `CacheSetTool` | Store a string value in an in-process key-value cache with optional TTL (seconds). The cache persists for the lifetime of the process and is shared across all agents. Input: `key`, `value`, optional `ttl_seconds`. |
| `cache_get` | `CacheGetTool` | Retrieve a cached value by key. Returns the default (or empty string) if the key is missing or expired. Input: `key`, optional `default`. |

---

### Encoding & Hashing Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `base64` | `Base64Tool` | Encode a string to Base64 or decode a Base64 string. Input: `input`, optional `mode` (`encode`\|`decode`, default: `encode`). |
| `hash_file` | `HashFileTool` | Compute the FNV-1a 64-bit hash of a file within the sandbox. Useful for detecting file changes or verifying integrity. Input: `path`. |

---

### Web Researcher Tool

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `web_fetch` | `WebFetchTool` | Fetch a URL and return clean Markdown. Scripts, nav, ads, and noise are stripped. Headings, links, lists, and code blocks are converted to Markdown. Input: `url`, optional `timeout_ms`, `max_length`. |

```rust
// web_fetch returns far fewer tokens than raw HTML.
// Example output for a documentation page:
// "URL: https://example.com\nStatus: 200\n\n# Page Title\n\nContent as Markdown..."
```

---

### Search Engine Tool

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `tavily_search` | `TavilySearchTool` | Real-time web search via the Tavily API. Returns ranked results with titles, URLs, content snippets, and an AI-generated answer. Requires `TAVILY_API_KEY`. Input: `query`, optional `max_results`, `search_depth` (`basic`\|`advanced`), `include_answer`. |

Get a free Tavily API key at <https://tavily.com>.

---

### Structured Data Parser Tool

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `schema_validate` | `SchemaValidateTool` | Parse a string (or messy text with embedded JSON) and validate it against a JSON Schema. Supports `required`, property `type` checks, and `enum` constraints. Returns pretty-printed JSON on success, or a detailed error report listing every violation. Input: `input`, `schema`, optional `extract_json`. |

---

### Knowledge Base (RAG) Tools

An in-process knowledge base for Retrieval-Augmented Generation (RAG). Documents are indexed with a lightweight TF-based keyword scorer. No external vector database required.

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `rag_add` | `RagAddTool` | Add or update a document in the knowledge base. Input: `id`, `content`, optional `metadata` (JSON). |
| `rag_search` | `RagSearchTool` | Search by keyword query; returns top-k matching documents with scores and metadata. Input: `query`, optional `top_k`, `min_score`. |
| `rag_clear` | `RagClearTool` | Remove a specific document by `id`, or omit `id` to clear the entire store. |

```rust
// Typical RAG workflow:
// 1. Agent calls rag_add for each document to index
// 2. For each user question, agent calls rag_search to find relevant docs
// 3. Agent uses retrieved docs as context to generate an accurate answer
```

---

### MessageBus Tools

Allow agents to communicate with each other during a tool call. Requires injecting a shared `MessageBus` instance at registration time:

```rust
use open_multi_agent_rs::{messaging::MessageBus, tool::built_in::register_bus_tools};

let bus = Arc::new(MessageBus::new());
register_bus_tools(&mut registry, Arc::clone(&bus)).await;
```

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `bus_publish` | `BusPublishTool` | Publish a message to a specific agent (`to: "agent-name"`) or broadcast (`to: "*"`). Input: `to`, `content`, optional `from`. |
| `bus_read` | `BusReadTool` | Read messages addressed to the current agent. Input: optional `agent`, `unread_only` (default: true), `mark_read` (default: true). |

---

### Shell & Search Tools

| Tool name | Struct | Description |
|-----------|--------|-------------|
| `bash` | `BashTool` | Run an arbitrary shell command with configurable timeout. Input: `command`, optional `timeout_ms`. |
| `grep` | `GrepTool` | Search file contents with a pattern. Input: `pattern`, `path`, optional `recursive` (bool). |

### Utility Tools

General-purpose helpers that cover gaps across nearly every agent workflow.

| Tool name  | Struct        | Description |
|------------|---------------|-------------|
| `sleep`    | `SleepTool`   | Pause execution for `ms` milliseconds (max 300 000). Essential for rate-limiting and polling loops. |
| `random`   | `RandomTool`  | Generate random values: `uuid` (v4), `int` (range), `float` [0,1), `choice` from a list, or `string` (alphanumeric). |
| `template` | `TemplateTool`| Render `{{variable}}` placeholders in a template string using a provided vars object. Supports `strict` mode. |
| `diff`     | `DiffTool`    | Compute a unified diff between two strings or two files (`mode="files"`). Output mirrors `diff -u`. |
| `zip`      | `ZipTool`     | Create, extract, or list ZIP archives within the sandbox. Operations: `create`, `extract`, `list`. |
| `git`      | `GitTool`     | Run read-heavy + staging Git operations. Allowed: `status`, `log`, `diff`, `show`, `branch`, `tag`, `remote`, `stash`, `ls-files`, `shortlog`, `describe`, `rev-parse`, `cat-file`, `add`, `commit`, `init`. Force flags are blocked. |
| `url`      | `UrlTool`     | Parse, build, percent-encode, percent-decode, or resolve (join) URLs. No external URL crate required. |

#### Usage snapshot

```rust
// Pause for 500ms
sleep ms=500

// Generate a UUID
random kind="uuid"

// Roll a dice
random kind="int" min=1 max=6

// Pick randomly from a list
random kind="choice" items=["red","green","blue"]

// Render a template
template template="Hello, {{name}}! Order #{{id}} shipped."
         vars={"name":"Alice","id":"9821"}

// Diff two strings
diff a="old line\nsecond" b="new line\nsecond"

// Diff two files
diff a="before.txt" b="after.txt" mode="files"

// Create a zip archive
zip operation="create" archive="bundle.zip" files=["a.txt","b.txt"]

// List archive contents
zip operation="list" archive="bundle.zip"

// Extract to a directory
zip operation="extract" archive="bundle.zip" dest="output/"

// Parse a URL
url operation="parse" url="https://example.com/api?q=rust#docs"

// Build a URL with query parameters
url operation="build" scheme="https" host="api.example.com" path="/search"
    query={"q":"hello world"}

// Percent-encode a string
url operation="encode" url="hello world & foo=bar"

// Resolve a relative URL
url operation="join" base="https://docs.rs/tokio/latest/tokio/index.html"
                     url="../time/index.html"

// Git status
git args="status"

// Git log (last 5 commits, one line each)
git args="log --oneline -5"
```

---

## Feedback Loop

`FeedbackLoop` pairs a **worker** agent with a **critic** agent and runs them in alternating turns until the critic approves or `max_rounds` is reached. Each revision round gives the worker richer context: original task + previous draft + critic feedback.

```
Round 1:   worker ← original task
           critic ← worker output

Round 2+:  worker ← original task + previous draft + critic feedback
           critic ← revised output

Exit when critic satisfies the approval predicate, or max_rounds hit.
```

### Basic usage

```rust
use open_multi_agent_rs::FeedbackLoop;

let result = FeedbackLoop::new(writer_config, editor_config)
    .max_rounds(3)
    .approval_signal("APPROVED")   // critic must include this word
    .run(task, registry, executor, adapter)
    .await?;

println!("approved={} rounds={}", result.approved, result.rounds);
println!("{}", result.final_output);  // pass this to the next agent
```

### Custom approval logic

```rust
// Approve when critic gives a score of 9 or 10
FeedbackLoop::new(coder_config, reviewer_config)
    .max_rounds(5)
    .approve_when(|output| output.contains("score: 9") || output.contains("score: 10"))
    .on_round(|round, worker_out, critic_out, approved| {
        println!("Round {round}: approved={approved}");
    })
    .run(task, registry, executor, adapter)
    .await?;
```

### Four-agent pipeline: A → (B ↔ C) → D

```rust
// Step 1: A researches
let result_a = Agent::new(researcher_config, ...).run(topic, adapter).await?;

// Step 2: B writes, C edits — loop until APPROVED
let result_bc = FeedbackLoop::new(writer_config, editor_config)
    .max_rounds(3)
    .approval_signal("APPROVED")
    .run(&result_a.output, registry, executor, adapter)
    .await?;

// Step 3: D publishes the approved output
let result_d = Agent::new(publisher_config, ...).run(&result_bc.final_output, adapter).await?;
```

### `FeedbackLoopResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `final_output` | `String` | Worker's last output — pass to downstream agents |
| `approved` | `bool` | `true` if critic approved before `max_rounds` |
| `rounds` | `usize` | How many iterations ran |
| `history` | `Vec<Round>` | Full transcript for debugging |

Each `Round` has: `round` (1-based), `worker_output`, `critic_output`, `approved`.

### Builder API

| Method | Default | Description |
|--------|---------|-------------|
| `max_rounds(n)` | `3` | Maximum iterations (minimum 1) |
| `approval_signal(s)` | `"APPROVED"` | Approve when critic output contains `s` (case-insensitive) |
| `approve_when(fn)` | — | Custom closure, overrides `approval_signal` |
| `on_round(fn)` | — | Callback after each round: `(round, worker_out, critic_out, approved)` |

---

## Examples

All examples require `OPENROUTER_API_KEY` to be set (add it to a `.env` file or export it).

```bash
cp .env.example .env   # add your key
cargo run --example <name>
```

| # | Name | What it demonstrates |
|---|------|---------------------|
| 01 | `01_single_agent` | One-shot agent with `OpenMultiAgent::run_agent` |
| 02 | `02_multi_turn_chat` | Multi-turn conversation via `agent.prompt()` |
| 03 | `03_streaming` | Token-by-token streaming with `agent.stream()` |
| 04 | `04_custom_tool` | Implementing and registering a custom `Tool` |
| 05 | `05_structured_output` | JSON output with schema validation |
| 06 | `06_task_pipeline` | Dependency-aware task pipeline |
| 07 | `07_multi_agent_system` | Multi-agent team with coordinator |
| 08 | `08_hooks_and_trace` | Lifecycle hooks and trace observability |
| 09 | `09_message_bus` | Agent-to-agent messaging with `MessageBus` |
| 10 | `10_retry_and_approval` | Retry backoff and approval gates |
| 11 | `11_python_coding_agent` | Agent writes, tests, and fixes Python code |
| 12 | `12_repo_mindmap` | Ingest a repo and generate a Mermaid mindmap |
| 13 | `13_http_tools` | HTTP GET/POST + JSON parse/transform |
| 14 | `14_data_tools` | CSV, math eval, datetime, regex, text chunking |
| 15 | `15_system_tools` | System info, env vars, Base64, file hashing, cache |
| 16 | `16_web_search` | `web_fetch` (HTML→Markdown) + `tavily_search` + `schema_validate` |
| 17 | `17_rag_knowledge_base` | In-process RAG with `rag_add` / `rag_search` / `rag_clear` |
| 18 | `18_bus_agents` | Two agents communicating via `bus_publish` / `bus_read` |
| 19 | `19_knowledge_base_pipeline` | Full Karpathy LLM knowledge-base pipeline (ingest → compile → RAG → health-check) |
| 20 | `20_utility_tools` | `sleep`, `random`, `template`, `diff`, `zip`, `git`, `url` in a single agent demo |
| 21 | `21_karpathy_full_pipeline` | Full 6-stage Karpathy pipeline: ingest → compile → index → stubs → Q&A → health check |
| 22 | `22_feedback_loop` | Four-agent pipeline A → (B ↔ C) → D with iterative writer/editor feedback loop |

### Example 22 — Feedback Loop: A → (B ↔ C) → D

```bash
cargo run --example 22_feedback_loop
```

Four agents run as a linear pipeline. Two of them (writer B and editor C) form a `FeedbackLoop` inside it:

```
A (Researcher) → research brief
B (Writer) ↔ C (Editor)   ← up to 3 rounds
D (Publisher)  → formatted post
```

- **A** researches the topic and produces a structured brief
- **B** writes a blog post draft using the brief; **C** reviews and either approves or gives numbered feedback
- Each revision round B gets: original brief + previous draft + C's feedback
- **D** receives the approved post and formats it for publication with title, meta description, and tags

### Example 11 — Python Coding Agent

```bash
cargo run --example 11_python_coding_agent
```

The agent:
1. Writes `calculator.py` with `add`, `subtract`, `multiply`, `divide` (raises `ValueError` on zero)
2. Writes `test_calculator.py` with pytest tests for all functions
3. Runs the tests and fixes any failures
4. Reports the final pytest output

### Example 16 — Web Research & Search

```bash
cargo run --example 16_web_search

# With Tavily real-time search:
TAVILY_API_KEY=tvly-... cargo run --example 16_web_search
```

The agent uses `web_fetch` to retrieve a page as clean Markdown, or `tavily_search` for real-time results if `TAVILY_API_KEY` is set. Then uses `schema_validate` to force structured JSON output.

### Example 17 — RAG Knowledge Base

```bash
cargo run --example 17_rag_knowledge_base
```

The agent adds 4 programming-topic documents with `rag_add`, searches them by keyword with `rag_search`, updates a document, removes one with `rag_clear`, and confirms the removal.

### Example 18 — MessageBus Multi-Agent Communication

```bash
cargo run --example 18_bus_agents
```

Two agents share a `MessageBus`. The researcher agent computes facts and publishes findings with `bus_publish`. The writer agent reads them with `bus_read` and writes a report, then broadcasts completion. Demonstrates inter-agent coordination during a single run.

### Example 13 — HTTP Tools

```bash
cargo run --example 13_http_tools
```

The agent fetches `https://httpbin.org/json`, extracts a field with `json_parse`, lists top-level keys with `json_transform`, and POSTs a JSON payload to `https://httpbin.org/post`.

### Example 14 — Data Processing Tools

```bash
cargo run --example 14_data_tools
```

The agent: creates a CSV sales file with `csv_write`, reads it back with `csv_read`, extracts columns with `json_transform`, computes revenue figures with `math_eval`, inspects timestamps with `datetime`, and extracts dates from text with `text_regex`.

### Example 15 — System & Utility Tools

```bash
cargo run --example 15_system_tools
```

The agent: inspects the runtime environment with `system_info`, reads an env var with `env_get`, encodes and decodes a message with `base64`, hashes a file with `hash_file`, and stores/retrieves a value with `cache_set`/`cache_get`.

### Example 19 — Karpathy LLM Knowledge-Base Pipeline

```bash
cargo run --example 19_knowledge_base_pipeline

# With live Tavily web search:
TAVILY_API_KEY=tvly-... cargo run --example 19_knowledge_base_pipeline
```

Implements the four-stage knowledge-base pipeline described by Andrej Karpathy:

| Stage | Tools | What happens |
|-------|-------|-------------|
| 1 — Ingest | `article_fetch`, `file_list` | Clip Rust docs pages into `raw/` as Markdown |
| 2 — Compile | `file_read`, `file_write`, `frontmatter` | LLM writes wiki pages with `[[WikiLinks]]` and YAML metadata |
| 3 — Index & Q&A | `rag_index_dir`, `wikilink_index`, `rag_search` | Bulk-index wiki, build link graph, answer a question |
| 4 — Health Check | `wikilink_index`, `grep`, `frontmatter`, `datetime` | Find orphans, check metadata, stamp `last_checked` |

### Example 20 — Utility Tools

```bash
cargo run --example 20_utility_tools
```

A single agent that exercises all seven utility tools:
- `sleep` — 200ms pause
- `random` — UUID, dice roll, choice, alphanumeric string
- `template` — order confirmation email render
- `diff` — unified diff between two multi-line strings
- `zip` — create, list, and extract an archive
- `git` — `git status` in the sandbox
- `url` — parse, build, encode, and resolve URLs

### Example 12 — Repository Mindmap

```bash
# Analyse this repo
cargo run --example 12_repo_mindmap

# Analyse a different directory
REPO_PATH=/path/to/project OUTPUT_FILE=my_mindmap.md cargo run --example 12_repo_mindmap
```

The agent calls `repo_ingest` on the target directory then writes a `mindmap`-type Mermaid diagram to the output `.md` file. The file renders in GitHub, VS Code (Mermaid Preview), and Obsidian.

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

Rate limit exceeded on the free-tier model. The OpenRouter adapter automatically retries up to 5 times with exponential backoff (5 s → 10 s → 20 s → 40 s → 80 s), honouring the `Retry-After` header. If all retries are exhausted, switch to a paid model/plan or wait before retrying.

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

---

## Powered By

This project is powered by **[Tahaluf Al Emarat](https://tahaluf.ai)**.
