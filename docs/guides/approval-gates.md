# Guide: Approval Gates

An approval gate is an async callback that fires between pipeline stages. The
orchestrator passes the completed tasks and the next batch of tasks to run; the
callback returns `true` to continue or `false` to stop.

## Attaching an approval gate

```rust
use futures::future::BoxFuture;
use std::sync::Arc;
use open_multi_agent::{OrchestratorConfig, OpenMultiAgent};
use open_multi_agent::types::Task;

let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
    default_provider: "openrouter".to_string(),
    default_api_key: std::env::var("OPENROUTER_API_KEY").ok(),
    on_approval: Some(Arc::new(
        |completed: Vec<Task>, pending: Vec<Task>| -> BoxFuture<'static, bool> {
            Box::pin(async move {
                println!(
                    "[approval] {} task(s) done, {} task(s) pending",
                    completed.len(), pending.len()
                );

                // Inspect completed results
                for task in &completed {
                    if let Some(ref result) = task.result {
                        println!("  completed: {} → {:.80}", task.title, result);
                    }
                }

                // Inspect what is about to run
                for task in &pending {
                    println!("  pending: {}", task.title);
                }

                // Return true to proceed, false to stop
                true
            })
        },
    )),
    ..Default::default()
});
```

## When the gate fires

The gate is called after each batch of tasks completes, before the next batch begins.
"Batch" means all tasks that became unblocked at the same time (i.e., their
dependencies all completed in the previous round).

```
[stage 1 tasks complete]
  → approval gate fires (completed=[stage1], pending=[stage2])
    → true  → [stage 2 tasks run]
              → approval gate fires (completed=[stage2], pending=[stage3])
                → false → all remaining tasks are marked Skipped
                        → pipeline ends
```

## Rejecting a gate

Return `false` to stop. All remaining non-terminal tasks are marked `Skipped`.
`TeamRunResult.success` will be `false`.

```rust
on_approval: Some(Arc::new(
    |completed: Vec<Task>, pending: Vec<Task>| -> BoxFuture<'static, bool> {
        Box::pin(async move {
            // Check for quality gate
            let all_passed = completed.iter().all(|t| {
                t.result.as_deref().map(|r| !r.contains("ERROR")).unwrap_or(false)
            });
            all_passed  // false → pipeline stops here
        })
    }
)),
```

## Human-in-the-loop

Use `tokio::io` to read console input inside the approval callback:

```rust
on_approval: Some(Arc::new(
    |completed, pending| -> BoxFuture<'static, bool> {
        Box::pin(async move {
            println!("{} task(s) completed. Proceed? [y/N]", completed.len());
            let mut line = String::new();
            tokio::io::AsyncBufReadExt::read_line(
                &mut tokio::io::BufReader::new(tokio::io::stdin()),
                &mut line,
            ).await.ok();
            line.trim().eq_ignore_ascii_case("y")
        })
    }
)),
```

## Gate callback signature

```rust
type ApprovalGate = Arc<
    dyn Fn(
        completed: Vec<Task>,   // Tasks that finished in the last batch
        pending: Vec<Task>,     // Tasks that are next in line
    ) -> BoxFuture<'static, bool>   // true = proceed, false = stop
    + Send + Sync
>;
```

The `Vec<Task>` arguments include full task state: `id`, `title`, `description`,
`result`, `status`, `assignee`, timing fields, and retry configuration.

## Approval gate vs. hooks

| | Approval gate | `before_run` / `after_run` hook |
|---|---|---|
| Scope | Between pipeline stages | Per individual agent run |
| Trigger | After each batch of tasks | Before / after every LLM call |
| Can abort | Yes (return false) | Yes (return Err) |
| Receives | Completed + pending `Task` lists | `BeforeRunHookContext` / `AgentRunResult` |
