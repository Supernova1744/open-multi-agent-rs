# Guide: Feedback Loop

A `FeedbackLoop` pairs a **worker** agent with a **critic** agent and runs them in
alternating turns. The worker produces output; the critic evaluates it. If the critic
approves, the loop exits. Otherwise the worker receives its previous draft plus the
critic's feedback and revises. This repeats until approval or `max_rounds` is reached.

```
Round 1:   worker ← original task
           critic ← worker output

Round 2+:  worker ← original task + previous draft + critic feedback
           critic ← revised output

Exit when critic satisfies the approval predicate, or max_rounds hit.
```

## Quick start

```rust
use open_multi_agent_rs::FeedbackLoop;

let result = FeedbackLoop::new(writer_config, editor_config)
    .max_rounds(3)
    .approval_signal("APPROVED")
    .run(task, registry, executor, adapter)
    .await?;

println!("approved={} rounds={}", result.approved, result.rounds);
println!("{}", result.final_output);
```

## Builder API

### `FeedbackLoop::new(worker, critic)`

Both arguments are `AgentConfig` values. The worker receives the task and produces
drafts; the critic receives each draft and either approves or gives feedback.

Design the critic's `system_prompt` to always either say your approval word or give
numbered actionable feedback:

```rust
let editor = AgentConfig {
    name: "editor".to_string(),
    system_prompt: Some(
        "You are a strict technical editor. \
         If the post is ready, respond with APPROVED followed by a brief reason. \
         Otherwise provide numbered, specific feedback on what must be improved."
            .to_string(),
    ),
    ..Default::default()
};
```

### `.max_rounds(n)`

Maximum iterations before the loop exits regardless of approval. Default: `3`.
Minimum enforced: `1`.

```rust
FeedbackLoop::new(worker, critic).max_rounds(5)
```

### `.approval_signal("APPROVED")`

The loop exits when the critic's output contains this string (case-insensitive).
Default signal is `"APPROVED"`.

```rust
.approval_signal("LGTM")   // approve on "lgtm", "LGTM", "Lgtm"
```

### `.approve_when(closure)`

For custom approval logic. Overrides `approval_signal`.

```rust
// Approve when critic gives a score of 9 or 10
.approve_when(|output| output.contains("score: 9") || output.contains("score: 10"))

// Approve when output is valid JSON
.approve_when(|output| serde_json::from_str::<serde_json::Value>(output).is_ok())
```

### `.on_round(callback)`

Called after every round. Useful for progress logging.

```rust
.on_round(|round, worker_output, critic_output, approved| {
    println!("Round {round}: approved={approved}");
    println!("  worker: {}...", &worker_output[..80.min(worker_output.len())]);
    println!("  critic: {}", &critic_output[..80.min(critic_output.len())]);
})
```

Signature: `fn(round: usize, worker_output: &str, critic_output: &str, approved: bool)`

## Using the result

```rust
let result = loop_.run(task, registry, executor, adapter).await?;

// Pass to downstream agent
next_agent.run(&result.final_output, adapter).await?;

// Check if approved or exhausted
if !result.approved {
    eprintln!("Warning: max rounds reached without approval");
}

// Inspect full history
for round in &result.history {
    println!("Round {}: worker={} chars, approved={}",
        round.round, round.worker_output.len(), round.approved);
}
```

## Four-agent pipeline pattern

```
A (Researcher) → brief
B (Writer) ↔ C (Editor)   ← feedback loop
D (Publisher)  → final output
```

```rust
// Step 1: A researches
let brief = Agent::new(researcher_config, Arc::clone(&registry), Arc::clone(&executor))
    .run(&format!("Research: {}", topic), Arc::clone(&adapter))
    .await?;

// Step 2: B writes, C edits — iterate until approved
let loop_result = FeedbackLoop::new(writer_config, editor_config)
    .max_rounds(3)
    .approval_signal("APPROVED")
    .run(&brief.output, Arc::clone(&registry), Arc::clone(&executor), Arc::clone(&adapter))
    .await?;

// Step 3: D publishes the approved output
let published = Agent::new(publisher_config, Arc::clone(&registry), Arc::clone(&executor))
    .run(&loop_result.final_output, adapter)
    .await?;
```

## Adjusting context passed to the worker

On round 2+, the worker receives:

```
<original task>

---
Your previous draft:
<previous worker output>

Reviewer feedback (address all points):
<critic output>
```

You can shape how the worker uses this by its `system_prompt`:

```rust
let writer = AgentConfig {
    system_prompt: Some(
        "You are a technical writer. When given reviewer feedback, \
         address every numbered point explicitly before rewriting."
            .to_string(),
    ),
    ..Default::default()
};
```

## Limiting tool access per agent

Use `AgentConfig.tools` to give each agent only the tools it needs:

```rust
let writer = AgentConfig {
    tools: Some(vec!["file_read".to_string(), "rag_search".to_string()]),
    ..Default::default()
};

let editor = AgentConfig {
    tools: Some(vec!["file_read".to_string()]),
    ..Default::default()
};
```

## Run the example

```bash
cargo run --example 22_feedback_loop
```

See [examples.md](../examples.md#22----feedback-loop-a--b--c--d) for a full walkthrough.
