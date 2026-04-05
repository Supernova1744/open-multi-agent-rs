# Guide: Lifecycle Hooks and Observability

## Lifecycle Hooks

Hooks let you intercept and modify agent execution at two points: just before each
run starts, and just after each successful run completes.

### `before_run` — modify or abort before the LLM is called

```rust
use futures::future::BoxFuture;
use std::sync::Arc;
use open_multi_agent::{
    AgentConfig,
    types::{BeforeRunHookContext, BeforeRunHookAsync},
    error::Result,
};

let before: BeforeRunHookAsync = Arc::new(|mut ctx: BeforeRunHookContext| -> BoxFuture<'static, Result<BeforeRunHookContext>> {
    Box::pin(async move {
        // Modify the prompt
        ctx.prompt = format!("{}\n\nRespond in at most 50 words.", ctx.prompt);
        println!("[before_run] agent={} prompt_len={}", ctx.agent_name, ctx.prompt.len());
        Ok(ctx)
        // Return Err(...) to abort the run entirely
    })
});

let config = AgentConfig {
    name: "assistant".to_string(),
    before_run: Some(before),
    ..Default::default()
};
```

`BeforeRunHookContext` fields:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `String` | The user prompt — modify this to change what the model sees |
| `agent_name` | `String` | Read-only agent name |
| `agent_model` | `String` | Read-only model string |

### `after_run` — modify or abort after the LLM responds

```rust
use open_multi_agent::types::{AgentRunResult, AfterRunHookAsync};

let after: AfterRunHookAsync = Arc::new(|mut result: AgentRunResult| -> BoxFuture<'static, Result<AgentRunResult>> {
    Box::pin(async move {
        // Uppercase the output
        result.output = result.output.to_uppercase();
        println!("[after_run] output_len={}", result.output.len());
        Ok(result)
        // Return Err(...) to mark the run as failed
    })
});

let config = AgentConfig {
    name: "assistant".to_string(),
    after_run: Some(after),
    ..Default::default()
};
```

### Aborting a run from a hook

Return `Err(AgentError::Other("reason".to_string()))` from either hook to abort.
The orchestrator records the run as failed and the error propagates as normal.

```rust
let guard: BeforeRunHookAsync = Arc::new(|ctx| Box::pin(async move {
    if ctx.prompt.contains("FORBIDDEN") {
        return Err(open_multi_agent::error::AgentError::Other(
            "forbidden keyword detected".to_string()
        ));
    }
    Ok(ctx)
}));
```

---

## Trace / Observability

The trace system emits a structured event at the end of every LLM call, tool
execution, task, and agent run. Attach a callback to `OrchestratorConfig.on_trace`
or `RunOptions.on_trace`.

### Attaching a trace callback

```rust
use open_multi_agent::{OrchestratorConfig, OpenMultiAgent};
use open_multi_agent::types::{TraceEvent, OnTraceFn};
use std::sync::Arc;

let on_trace: OnTraceFn = Arc::new(|event| {
    match event {
        TraceEvent::LlmCall(e) => println!(
            "[llm_call] agent={} turn={} model={} tokens={}in/{}out duration={}ms",
            e.base.agent, e.turn, e.model,
            e.tokens.input_tokens, e.tokens.output_tokens,
            e.base.duration_ms
        ),
        TraceEvent::ToolCall(e) => println!(
            "[tool_call] agent={} tool={} error={} duration={}ms",
            e.base.agent, e.tool, e.is_error, e.base.duration_ms
        ),
        TraceEvent::Task(e) => println!(
            "[task] id={} title={} success={} retries={}",
            e.task_id, e.task_title, e.success, e.retries
        ),
        TraceEvent::Agent(e) => println!(
            "[agent] name={} turns={} tool_calls={}",
            e.base.agent, e.turns, e.tool_calls
        ),
    }
});

let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
    on_trace: Some(on_trace),
    ..Default::default()
});
```

### Trace event fields

Every event carries a `TraceEventBase`:

```rust
pub struct TraceEventBase {
    pub run_id: String,           // Unique per orchestrator.run_* call
    pub start_ms: u64,            // Unix epoch milliseconds
    pub end_ms: u64,
    pub duration_ms: u64,
    pub agent: String,
    pub task_id: Option<String>,  // Set when running inside a task pipeline
}
```

### Generating run IDs

```rust
use open_multi_agent::trace::generate_run_id;

let run_id = generate_run_id();  // UUID v4 string
```

Pass it to `RunOptions.run_id` to correlate all trace events for a single run:

```rust
let opts = RunOptions {
    run_id: Some(generate_run_id()),
    on_trace: Some(my_trace_fn),
    ..Default::default()
};
agent.run_with_opts(prompt, adapter, opts).await?;
```

### Panic safety

`emit_trace` catches and swallows any panics that occur inside your callback.
A panicking trace callback will never crash the agent loop.

### Forwarding to external systems

```rust
let sender = tokio::sync::mpsc::unbounded_channel::<TraceEvent>().0;

let on_trace: OnTraceFn = Arc::new(move |event| {
    // Clone-able events can be forwarded to a channel, OpenTelemetry, etc.
    let _ = sender.send(event);
});
```

### Per-call trace

To attach a trace to a single `Agent::run_with_opts` call rather than all calls
through the orchestrator:

```rust
let opts = RunOptions {
    on_trace: Some(Arc::new(|event| { /* ... */ })),
    run_id: Some("my-run-123".to_string()),
    task_id: Some("task-abc".to_string()),
    trace_agent: Some("custom-agent-name".to_string()),
    ..Default::default()
};
```
