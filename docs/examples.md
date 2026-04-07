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

---

## 11 — Python Coding Agent

**File:** `examples/11_python_coding_agent.rs`  
**Run:** `cargo run --example 11_python_coding_agent`

Demonstrates an agent that writes, tests, and self-corrects Python code using the
`python_write`, `python_run`, and `python_test` built-in tools.

**What it shows:**
- `python_write` — create `.py` files inside the sandbox
- `python_run` — execute a script and capture stdout/stderr
- `python_test` — run pytest and return pass/fail output
- Self-healing: agent reads test failure output and fixes the code

**Pipeline:**
```
python_write calculator.py →
python_write test_calculator.py →
python_test test_calculator.py →
(if fail) python_write fixed → python_test again
```

---

## 12 — Repository Mindmap

**File:** `examples/12_repo_mindmap.rs`  
**Run:** `cargo run --example 12_repo_mindmap`

Uses `repo_ingest` to analyse a codebase and write a Mermaid `mindmap` diagram.

**What it shows:**
- `repo_ingest` — recursively reads a directory tree into a single context string
- `file_write` — saves the Mermaid diagram to a `.md` file
- `REPO_PATH` / `OUTPUT_FILE` environment variables for customisation

**Output:** A `.md` file with a `mindmap` diagram renderable in GitHub, VS Code, and Obsidian.

---

## 13 — HTTP Tools

**File:** `examples/13_http_tools.rs`  
**Run:** `cargo run --example 13_http_tools`

Demonstrates `http_get`, `http_post`, `json_parse`, and `json_transform` against
the public httpbin.org test service.

**What it shows:**
- `http_get url='https://httpbin.org/json'` — fetch JSON
- `json_parse pointer='/slideshow/title'` — extract a nested field
- `json_transform operation='keys'` — list top-level keys
- `http_post` with a JSON body and response inspection

---

## 14 — Data Processing Tools

**File:** `examples/14_data_tools.rs`  
**Run:** `cargo run --example 14_data_tools`

Demonstrates CSV, math, datetime, regex, and text chunking in a single workflow.

**What it shows:**
- `csv_write` / `csv_read` — create and read a sales CSV file
- `json_transform` — extract a column from the parsed rows
- `math_eval` — compute totals with `evalexpr` expressions
- `datetime` — parse timestamps and compute differences
- `text_regex` — extract date patterns from free text

---

## 15 — System & Utility Tools

**File:** `examples/15_system_tools.rs`  
**Run:** `cargo run --example 15_system_tools`

Demonstrates system introspection, encoding, hashing, and cache tools.

**What it shows:**
- `system_info` — OS, architecture, CPU count, CWD
- `env_get` — read an allow-listed environment variable
- `base64 operation='encode'` / `'decode'` — round-trip encoding
- `hash_file` — FNV-1a 64-bit hash of a file
- `cache_set` / `cache_get` — in-process key-value store with TTL

---

## 16 — Web Search

**File:** `examples/16_web_search.rs`  
**Run:** `cargo run --example 16_web_search`  
**With Tavily:** `TAVILY_API_KEY=tvly-... cargo run --example 16_web_search`

Demonstrates web fetching, search, and schema validation.

**What it shows:**
- `web_fetch` — retrieves a URL and converts HTML to clean Markdown
- `tavily_search` — real-time web search (skipped if `TAVILY_API_KEY` is absent)
- `schema_validate` — validates a JSON value against a JSON Schema
- Adaptive behaviour based on available environment variables

---

## 17 — RAG Knowledge Base

**File:** `examples/17_rag_knowledge_base.rs`  
**Run:** `cargo run --example 17_rag_knowledge_base`

Demonstrates in-process retrieval-augmented generation with the full CRUD lifecycle.

**What it shows:**
- `rag_add id content` — add four documents on different topics
- `rag_search query top_k=2` — TF-scored retrieval
- `rag_add` on an existing ID — update a document
- `rag_clear id` — remove a single document
- Verifying the removal with another search

---

## 18 — MessageBus Multi-Agent Communication

**File:** `examples/18_bus_agents.rs`  
**Run:** `cargo run --example 18_bus_agents`

Two agents share a `MessageBus`. Researcher publishes findings; writer reads them
and broadcasts completion.

**What it shows:**
- `register_bus_tools(registry, bus)` — inject the bus into the registry
- `bus_publish from to content` — point-to-point message
- `bus_read unread_only=true` — read messages addressed to the current agent
- `bus_publish to='*'` — broadcast to all agents

---

## 19 — Knowledge-Base Pipeline (quick demo)

**File:** `examples/19_knowledge_base_pipeline.rs`  
**Run:** `cargo run --example 19_knowledge_base_pipeline`

A four-stage skeleton of the Karpathy LLM knowledge-base pipeline using Rust
documentation as source material.

**Stages:**
1. **Ingest** — `article_fetch` two Rust docs pages into `raw/`
2. **Compile** — LLM reads raw files, writes `wiki/` pages with `[[WikiLinks]]`
3. **Index & Q&A** — `rag_index_dir` + `wikilink_index build` + `rag_search`
4. **Health** — `wikilink_index orphans` + `grep` + `frontmatter set`

See example 21 for the full six-stage version.

---

## 20 — Utility Tools

**File:** `examples/20_utility_tools.rs`  
**Run:** `cargo run --example 20_utility_tools`

Exercises all seven utility tools in a single agent session.

**Demonstrates:**
- `sleep ms=200` — rate-limiting pause
- `random kind='uuid'` / `'int'` / `'choice'` / `'string'` — random values
- `template` — `{{variable}}` substitution with strict mode
- `diff` — unified diff between two multi-line strings
- `zip` — create, list, extract an archive
- `git args='status'` — safe git wrapper
- `url` — parse, build, encode, and resolve URLs

---

## 21 — Full Karpathy Knowledge-Base Pipeline

**File:** `examples/21_karpathy_full_pipeline.rs`  
**Run:** `cargo run --example 21_karpathy_full_pipeline`  
**With Tavily:** `TAVILY_API_KEY=tvly-... cargo run --example 21_karpathy_full_pipeline`

The complete six-stage LLM knowledge-base pipeline over five interrelated Wikipedia
AI/ML articles.

**Stages:**

| # | Name | Key tools |
|---|------|-----------|
| 1 | Ingest | `article_fetch`, `image_download`, `file_list` |
| 2 | Compile | `file_read`, `file_write`, `frontmatter` |
| 3 | Index & Link Graph | `rag_index_dir`, `wikilink_index` |
| 4 | Stub Generation | `file_write`, `wikilink_index` |
| 5 | Multi-Hop Q&A | `rag_search`, `file_read` |
| 6 | Health Check | `grep`, `frontmatter`, `datetime`, `file_list` |

**Output:** Full wiki in a temp directory (`karpathy_full_kb/`) with main pages,
auto-generated stubs, Q&A answers, and a health report.

---

## 22 — Feedback Loop: A → (B ↔ C) → D

**File:** `examples/22_feedback_loop.rs`  
**Run:** `cargo run --example 22_feedback_loop`

A four-agent pipeline where a writer (B) and editor (C) iterate inside a
`FeedbackLoop` until the editor approves.

**Pipeline:**
```
A (Researcher) → research brief
                       ↓
             B (Writer) ↔ C (Editor)   ← up to 3 rounds
                       ↓ approved output
             D (Publisher) → formatted post
```

**What it shows:**
- `FeedbackLoop::new(worker, critic).max_rounds(3).approval_signal("APPROVED")`
- `.on_round(|round, worker, critic, approved| …)` — per-round progress callback
- Passing the loop's `final_output` to a downstream agent
- `FeedbackLoopResult`: `approved`, `rounds`, `history`

**Key code pattern:**
```rust
let result_bc = FeedbackLoop::new(agent_b, agent_c)
    .max_rounds(3)
    .approval_signal("APPROVED")
    .on_round(|round, w, c, ok| println!("Round {round}: approved={ok}"))
    .run(&result_a.output, registry, executor, adapter)
    .await?;

let result_d = Agent::new(agent_d, ...).run(&result_bc.final_output, adapter).await?;
```
