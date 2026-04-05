# Examples Walkthrough

All examples live in the `examples/` directory and can be run with:

```bash
cargo run --example <name>
```

Every example loads API credentials from `.env` via `dotenvy::dotenv().ok()`. Copy
`.env.copy` to `.env` and fill in your key(s) before running.

---

## 01 — Single Agent

**File:** `examples/01_single_agent.rs`  
**Run:** `cargo run --example 01_single_agent`

Demonstrates the simplest possible interaction: create an orchestrator, define one
agent, ask it a question.

**What it shows:**
- `OrchestratorConfig` with provider, API key, and concurrency
- `AgentConfig` with a system prompt
- `orchestrator.run_agent(config, prompt)` returning `AgentRunResult`
- Printing `output`, `turns`, and `token_usage`

**Key code pattern:**
```rust
let result = orchestrator.run_agent(config, "What year was Rust created?").await?;
println!("{}", result.output);
println!("tokens: {}in / {}out", result.token_usage.input_tokens, result.token_usage.output_tokens);
```

---

## 02 — Multi-Turn Chat

**File:** `examples/02_multi_turn_chat.rs`  
**Run:** `cargo run --example 02_multi_turn_chat`

Demonstrates stateful multi-turn conversation: three sequential questions where the
model remembers all previous exchanges.

**What it shows:**
- Constructing `Agent` directly with `ToolRegistry` and `ToolExecutor`
- `agent.prompt(message, adapter)` — appends to history each call
- `agent.get_history()` — inspects the full conversation
- Cumulative token counting across turns

**Key code pattern:**
```rust
let r1 = agent.prompt("What is ownership?", Arc::clone(&adapter)).await?;
let r2 = agent.prompt("Give me a short example.", Arc::clone(&adapter)).await?;
// r2's LLM call includes the full context of r1
```

---

## 03 — Streaming

**File:** `examples/03_streaming.rs`  
**Run:** `cargo run --example 03_streaming`

Demonstrates real-time token streaming — tokens appear on screen as the model
generates them, not as one bulk response at the end.

**What it shows:**
- `agent.stream(prompt, adapter)` returning `impl Stream<Item = StreamEvent>`
- `tokio::pin!` requirement for unboxed streams
- Matching on `StreamEvent::Text`, `ToolUse`, `ToolResult`, `Done`, `Error`
- `std::io::Write::flush()` to push partial output to the terminal immediately

**Key code pattern:**
```rust
let stream = agent.stream("Write a haiku about Rust.", Arc::clone(&adapter));
tokio::pin!(stream);
while let Some(event) = stream.next().await {
    if let StreamEvent::Text(chunk) = event {
        print!("{}", chunk);
        std::io::stdout().flush().ok();
    }
}
```

---

## 04 — Custom Tool

**File:** `examples/04_custom_tool.rs`  
**Run:** `cargo run --example 04_custom_tool`

Demonstrates defining and registering a custom tool that the agent autonomously calls
when solving a task.

**What it shows:**
- Implementing the `Tool` trait (`name`, `description`, `input_schema`, `execute`)
- Registering with `ToolRegistry::register`
- The agent automatically deciding to call the tool based on the prompt
- `result.tool_calls` audit log

**Tool defined:** `word_count(text: String) → count: usize`

**Key code pattern:**
```rust
struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn name(&self) -> &str { "word_count" }
    async fn execute(&self, input: &HashMap<String, Value>, _ctx: &ToolUseContext) -> Result<ToolResult> {
        let count = input["text"].as_str().unwrap_or("").split_whitespace().count();
        Ok(ToolResult { data: count.to_string(), is_error: false })
    }
    // ...
}
```

---

## 05 — Structured Output

**File:** `examples/05_structured_output.rs`  
**Run:** `cargo run --example 05_structured_output`

Demonstrates schema-validated JSON output: the agent is given a sentence and must
extract structured fields; the library validates and retries automatically.

**What it shows:**
- `AgentConfig.output_schema` set to a JSON Schema object
- `AgentRunResult.structured` containing the parsed value on success
- Automatic single retry with validation error feedback
- Graceful fallback to raw `output` if validation still fails

**Schema used:**
```json
{ "type": "object", "properties": { "name": {"type":"string"}, "age": {"type":"integer"}, "city": {"type":"string"} }, "required": ["name","age","city"] }
```

**Key code pattern:**
```rust
if let Some(ref structured) = result.structured {
    println!("name={} age={} city={}", structured["name"], structured["age"], structured["city"]);
}
```

---

## 06 — Task Pipeline

**File:** `examples/06_task_pipeline.rs`  
**Run:** `cargo run --example 06_task_pipeline`

Demonstrates a 3-stage dependency pipeline: Researcher → Analyst → Writer. Each agent
reads the previous stage's output from shared memory.

**What it shows:**
- `create_task` with `depends_on` IDs
- `TeamConfig` with `shared_memory: Some(true)`
- `orchestrator.run_tasks(&team, tasks)` resolving the graph automatically
- Downstream agents receiving earlier results in their prompts
- `TeamRunResult.agent_results` keyed by task ID

**Pipeline:**
```
[research task] → [analysis task] → [report task]
```

**Key code pattern:**
```rust
let t2 = create_task("Analysis", "...", Some("analyst".to_string()), vec![t1_id.clone()]);
let t3 = create_task("Report",   "...", Some("writer".to_string()),  vec![t2_id.clone()]);
let result = orchestrator.run_tasks(&team, vec![t1, t2, t3]).await?;
```

---

## 07 — Multi-Agent System

**File:** `examples/07_multi_agent_system.rs`  
**Run:** `cargo run --example 07_multi_agent_system`

A complete multi-agent system combining tracing, approval gates, retry, and inter-agent
context passing — closest to a real production workflow.

**What it shows:**
- Three-agent pipeline: Architect → Developer → Reviewer
- `on_trace` callback logging every LLM call and tool call span
- Approval gate that inspects completed results before proceeding
- `max_retries: Some(2)` on the Reviewer task for resilience
- `MessageBus` used for status coordination between agents

**System design:**
```
[Architect] → [Developer] → [Reviewer]
                   ↑ approval gate between each stage
```

---

## 08 — Lifecycle Hooks and Trace

**File:** `examples/08_hooks_and_trace.rs`  
**Run:** `cargo run --example 08_hooks_and_trace`

Demonstrates `before_run` and `after_run` hooks that modify the prompt and output,
plus a global trace callback.

**What it shows:**
- `before_run` hook appending a word limit to the prompt
- `after_run` hook uppercasing the result
- `OrchestratorConfig.on_trace` capturing all `LlmCall` and `Agent` events
- Hook return values: `Ok(ctx)` to proceed, `Err(…)` to abort

**Execution order:**
```
prompt entered →
  before_run fires (prompt modified) →
    LLM called →
  after_run fires (output modified) →
on_trace: Agent event emitted →
result returned
```

---

## 09 — MessageBus

**File:** `examples/09_message_bus.rs`  
**Run:** `cargo run --example 09_message_bus`

Demonstrates the full `MessageBus` API without any LLM calls — focuses entirely on
the messaging primitives.

**What it shows:**
- `bus.send(from, to, content)` — point-to-point
- `bus.broadcast(from, content)` — to all except sender
- `bus.subscribe(agent, callback)` — push model with closure
- `bus.get_unread(agent)` + `bus.mark_read(agent, ids)` — poll model
- `bus.get_conversation(a, b)` — full thread between two agents
- Cloning the bus to pass to multiple "agents"

**Pattern:**
```rust
let _unsub = bus.subscribe("worker", |msg| println!("got: {}", msg.content));
bus.send("coordinator", "worker", "start processing");
// callback fires immediately
let unread = bus.get_unread("worker");
```

---

## 10 — Retry and Approval

**File:** `examples/10_retry_and_approval.rs`  
**Run:** `cargo run --example 10_retry_and_approval`

Demonstrates retry mechanics and approval gate rejection in three self-contained
parts.

**Part 1 — Delay schedule:**
Prints the delay in milliseconds for each attempt using `compute_retry_delay`.

**Part 2 — `execute_with_retry` directly:**
A closure that fails the first two attempts and succeeds on the third. Shows the retry
callback, token accumulation, and final result.

**Part 3 — Approval gate rejection:**
Two-stage pipeline where the approval gate approves the first round but rejects the
second. Demonstrates `skip_remaining` behaviour and `TeamRunResult.success = false`.

**Key code pattern:**
```rust
// Part 2
let result = execute_with_retry(
    move || Box::pin(async move { /* ... */ }),
    &task,
    Some(Arc::new(|attempt, max, err, delay| {
        println!("[retry {}/{}] \"{}\" — waiting {}ms", attempt, max, err, delay);
    })),
).await;

// Part 3
on_approval: Some(Arc::new(move |_completed, _pending| Box::pin(async move {
    let round = counter.fetch_add(1, Ordering::SeqCst);
    round == 0  // approve first round only
}))),
```
