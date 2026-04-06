# Guide: Retry with Exponential Backoff

Every task in a pipeline can be configured to retry automatically on failure, using
exponential backoff to space out attempts.

## Per-task retry configuration

Set these fields on a `Task` before adding it to a pipeline:

```rust
use open_multi_agent_rs::create_task;

let mut task = create_task(
    "Flaky API Call",
    "Fetch data from the external service and summarise it.",
    Some("worker".to_string()),
    vec![],
);

task.max_retries = Some(3);       // up to 3 retries (4 total attempts)
task.retry_delay_ms = Some(500);  // base delay: 500 ms
task.retry_backoff = Some(2.0);   // multiply delay by 2.0 each attempt
```

## Delay schedule

```
attempt 1 (initial):   0 ms  (no delay before first try)
attempt 2 (retry 1): 500 ms
attempt 3 (retry 2): 1 000 ms
attempt 4 (retry 3): 2 000 ms
```

Formula: `delay = base_delay_ms * backoff^(attempt - 1)`, capped at **30 000 ms**.

Use `compute_retry_delay` to preview the schedule:

```rust
use open_multi_agent_rs::compute_retry_delay;

for attempt in 1..=5u32 {
    let delay = compute_retry_delay(500, 2.0, attempt);
    println!("attempt {} → {}ms", attempt, delay);
}
// attempt 1 →    0ms  (first attempt, no delay)
// attempt 2 →  500ms
// attempt 3 → 1000ms
// attempt 4 → 2000ms
// attempt 5 → 4000ms
```

## Orchestrator integration

Pass tasks with retry config to `run_tasks` — the orchestrator handles retries
automatically without any additional code:

```rust
orchestrator.run_tasks(&team, vec![task]).await?;
```

A task is considered failed (and retried) when the agent's `success` field is `false`.

## Using `execute_with_retry` directly

For custom retry logic outside of a full pipeline:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use open_multi_agent_rs::{execute_with_retry, create_task};
use open_multi_agent_rs::types::{AgentRunResult, TokenUsage};

let call_count = Arc::new(AtomicUsize::new(0));
let cc = Arc::clone(&call_count);

let mut task = create_task("my-task", "do something", None, vec![]);
task.max_retries = Some(3);
task.retry_delay_ms = Some(200);
task.retry_backoff = Some(1.5);

let result = execute_with_retry(
    move || {
        let cc = Arc::clone(&cc);
        Box::pin(async move {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Ok(AgentRunResult {
                    success: false,
                    output: format!("attempt {} failed", n + 1),
                    messages: vec![],
                    token_usage: TokenUsage { input_tokens: 5, output_tokens: 2 },
                    tool_calls: vec![],
                    turns: 1,
                    structured: None,
                })
            } else {
                Ok(AgentRunResult {
                    success: true,
                    output: "success!".to_string(),
                    messages: vec![],
                    token_usage: TokenUsage { input_tokens: 5, output_tokens: 10 },
                    tool_calls: vec![],
                    turns: 1,
                    structured: None,
                })
            }
        })
    },
    &task,
    // Optional: called between attempts
    Some(Arc::new(|attempt, max, error, delay_ms| {
        println!("retry {}/{} after {} error — waiting {}ms", attempt, max, error, delay_ms);
    })),
)
.await;

println!("Final: success={} output={}", result.success, result.output);
println!("Total tokens: {}in / {}out",
    result.token_usage.input_tokens,
    result.token_usage.output_tokens,
);
```

## Token accumulation

`execute_with_retry` accumulates token usage across all attempts and reports the total
in the final `AgentRunResult.token_usage`. This gives accurate billing/monitoring data
even when multiple attempts were needed.

## Retry callback signature

```rust
Option<Arc<dyn Fn(
    attempt: u32,   // Which retry this is (1-based, so 1 = first retry after initial fail)
    max: u32,       // Max retries configured on the task
    error: String,  // The failure message from the previous attempt
    delay: u64,     // Milliseconds the library will sleep before the next attempt
) + Send + Sync>>
```

## Defaults

| Field | Default | Meaning |
|-------|---------|---------|
| `max_retries` | `None` → 0 | No retry; first failure is final |
| `retry_delay_ms` | `None` → 1000 | 1 second base delay |
| `retry_backoff` | `None` → 2.0 | Double the delay each attempt |
