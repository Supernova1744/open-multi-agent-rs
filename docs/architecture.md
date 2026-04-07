# Architecture

## Module Map

```
open_multi_agent_rs
├── agent/          Agent state machine + streaming conversation loop
│   ├── mod.rs      Agent struct, run / prompt / stream / reset
│   └── runner.rs   AgentRunner — LLM-call → tool-exec → loop (internal)
├── llm/            Provider adapters
│   ├── mod.rs      LLMAdapter trait + create_adapter factory
│   ├── anthropic.rs  Anthropic Messages API (native tool-use blocks)
│   ├── openai.rs     OpenAI Chat Completions (tool_calls array)
│   └── openrouter.rs OpenRouter (OpenAI-compat + SSE streaming)
├── feedback.rs     FeedbackLoop — iterative worker ↔ critic cycle
├── tool/           Tool registry and execution
│   ├── mod.rs      Tool trait, ToolRegistry, ToolExecutor
│   └── built_in.rs 45 built-in tools + register_built_in_tools / register_bus_tools
├── orchestrator/   Multi-agent coordination
│   └── mod.rs      OpenMultiAgent, TeamCoordinator, retry helpers
├── task/           Task graph and scheduling
│   ├── mod.rs      Task helpers, topological_sort
│   ├── queue.rs    TaskQueue — dependency-aware state machine
│   └── scheduler.rs Scheduler — RoundRobin / LeastBusy / etc.
├── memory/         Shared state between agents
│   └── mod.rs      MemoryStore trait, InMemoryStore, SharedMemory
├── messaging/      Pub/sub message bus
│   └── mod.rs      MessageBus, Message
├── trace/          Observability helpers
│   └── mod.rs      emit_trace, generate_run_id, now_ms
├── error.rs        AgentError enum, Result<T> alias
└── types.rs        All shared data types
```

## Data Flow

### Single-agent turn

```
Caller
  │
  ▼
Agent::run / Agent::prompt / Agent::stream
  │   builds initial_messages, invokes AgentRunner
  ▼
AgentRunner::stream_internal  ← main agentic loop
  │
  ├─► LLMAdapter::stream(conversation, options)
  │       │
  │       ├─ SSE chunks ──► StreamEvent::Text  (yielded in real-time)
  │       └─ Complete   ──► LLMResponse { content, usage }
  │
  ├─► extract ToolUseBlock[]  ──► StreamEvent::ToolUse
  │
  ├─► ToolExecutor::execute (all tool calls in parallel)
  │       │
  │       └─ Tool::execute(input, context) ──► ToolResult
  │
  ├─► StreamEvent::ToolResult for each result
  │
  └─► push tool_result message → loop back ──► LLMAdapter::stream
          (break when no tool calls returned)
  │
  ▼
StreamEvent::Done(RunResult)
```

### Task pipeline (orchestrator)

```
orchestrator.run_tasks(team, tasks)
  │
  ├─ topological_sort(tasks)
  ├─ TaskQueue::add_batch(tasks)
  │
  ▼
TeamCoordinator loop (respects max_concurrency)
  │
  ├─ approval gate? ──► on_approval(completed, pending) → bool
  │
  ├─ Scheduler::schedule(pending, agents) → [(task_id, agent_name)]
  │
  ├─ for each assignment (concurrent):
  │       Agent::run(prompt) wrapped in execute_with_retry
  │           ▼
  │       TaskQueue::complete / fail
  │           ▼
  │       SharedMemory::write(agent_name, task_id, result)
  │
  └─ repeat until TaskQueue::is_complete()
  │
  ▼
TeamRunResult { success, agent_results, token_usage }
```

## Key Design Decisions

### Streaming is the primary path

`AgentRunner` always uses `LLMAdapter::stream()`. The default implementation on the
`LLMAdapter` trait wraps `chat()` with a one-shot stream that emits one `Text` event
and one `Complete` event — so non-streaming adapters get streaming semantics for free.
Adapters that support SSE (currently `OpenRouterAdapter`) override `stream()` to emit
incremental `Text` chunks.

### Conversation ownership

`AgentRunner::stream_internal` is a `async_stream::stream!` generator. To avoid a
lifetime conflict between the immutable borrow required by `adapter.stream(&conversation, …)`
and the later mutable `conversation.push(…)`, the conversation is cloned into a snapshot
(`conv_snap`) before each LLM call. The snapshot is consumed by the stream; the live
`conversation` vec grows with each turn.

### Parallel tool execution

All tool calls returned in a single LLM turn are dispatched with
`futures::future::join_all` — fully concurrent. Results are collected in order and
appended as a single `tool_result` message, matching the Anthropic and OpenAI
multi-tool conventions.

### Error propagation

`AgentError` is the library's single error type. It implements `std::error::Error` via
`thiserror`. All fallible public functions return `Result<T, AgentError>`. Panic in a
trace callback is caught by `emit_trace` and silently swallowed to avoid crashing the
agent loop.

### Thread safety

Every shared piece of state (`ToolRegistry`, `InMemoryStore`, `MessageBus`) is wrapped
in `Arc<Mutex<_>>` (tokio's async mutex where lock is held across `.await`, `std::sync`
elsewhere). Callbacks and hooks are `Arc<dyn Fn(…) + Send + Sync>` so they can be
cloned freely across task boundaries.
