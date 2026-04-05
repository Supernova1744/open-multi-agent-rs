# API Reference

Complete public surface of the `open_multi_agent` crate.

---

## Error Handling

### `AgentError`

```rust
pub enum AgentError {
    LlmError(String),
    ToolNotFound(String),
    ToolError(String),
    TaskNotFound(String),
    DependencyCycle,
    MaxTurnsExceeded(usize),
    RequestError(reqwest::Error),
    JsonError(serde_json::Error),
    IoError(std::io::Error),
    Other(String),
}
```

All fallible functions return `open_multi_agent::error::Result<T>`, which is
`std::result::Result<T, AgentError>`.

---

## Core Types (`types`)

### `ContentBlock`

The union of every block type that can appear inside an `LLMMessage`.

```rust
pub enum ContentBlock {
    Text { text: String },
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Image { source: ImageSource },
}
```

**Methods:**

```rust
impl ContentBlock {
    /// Returns the text content if this is a Text block, otherwise None.
    pub fn as_text(&self) -> Option<&str>;

    /// Returns the inner ToolUseBlock if this is a ToolUse block, otherwise None.
    pub fn as_tool_use(&self) -> Option<&ToolUseBlock>;
}
```

---

### `ToolUseBlock`

Represents a tool call requested by the LLM.

```rust
pub struct ToolUseBlock {
    pub id: String,                                // Provider-assigned unique ID for this call
    pub name: String,                              // Tool name
    pub input: HashMap<String, serde_json::Value>, // Arguments
}
```

---

### `ToolResultBlock`

Carries the result of a tool execution back to the LLM.

```rust
pub struct ToolResultBlock {
    pub tool_use_id: String,      // Matches the ToolUseBlock.id that triggered this
    pub content: String,          // Tool output (stringified)
    pub is_error: Option<bool>,   // Some(true) on failure, None or Some(false) on success
}
```

---

### `ImageSource`

```rust
pub struct ImageSource {
    pub source_type: String,   // e.g. "base64"
    pub media_type: String,    // e.g. "image/png"
    pub data: String,          // Base64-encoded bytes
}
```

---

### `LLMMessage`

A single message in the conversation history.

```rust
pub struct LLMMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

pub enum Role {
    User,
    Assistant,
}
```

---

### `TokenUsage`

```rust
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsage {
    pub fn add(&self, other: &TokenUsage) -> TokenUsage;
}

impl Default for TokenUsage {
    fn default() -> Self; // { input_tokens: 0, output_tokens: 0 }
}
```

---

### `LLMResponse`

The complete result of one LLM API call.

```rust
pub struct LLMResponse {
    pub id: String,                  // Provider response ID
    pub content: Vec<ContentBlock>,  // Text blocks + tool-use blocks
    pub model: String,               // Model string that actually served the request
    pub stop_reason: String,         // "end_turn", "tool_use", "max_tokens", etc.
    pub usage: TokenUsage,
}
```

---

### `LLMChatOptions`

Options forwarded to the LLM on every call.

```rust
pub struct LLMChatOptions {
    pub model: String,
    pub tools: Option<Vec<LLMToolDef>>,    // None → no tools exposed
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system_prompt: Option<String>,
}
```

---

### `LLMToolDef`

The JSON-Schema tool definition sent to the LLM.

```rust
pub struct LLMToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,   // JSON Schema object
}
```

---

### `LLMStreamDelta`

Events emitted by `LLMAdapter::stream`.

```rust
pub enum LLMStreamDelta {
    /// A chunk of generated text (one token to a few words).
    Text(String),
    /// Stream finished. Carries tool-use blocks and metadata.
    /// Text content is NOT repeated inside the LLMResponse.content.
    Complete(LLMResponse),
}
```

---

### `StreamEvent`

Events yielded by `Agent::stream`.

```rust
pub enum StreamEvent {
    Text(String),                    // Incremental text token(s)
    ToolUse(ToolUseBlock),           // Tool the agent is about to call
    ToolResult(ToolResultBlock),     // Result returned from the tool
    Done(RunResult),                 // Final result; stream is over
    Error(String),                   // Unrecoverable error; stream is over
}
```

---

### `RunResult`

Low-level result from `AgentRunner` (wrapped into `AgentRunResult` by `Agent`).

```rust
pub struct RunResult {
    pub messages: Vec<LLMMessage>,      // Only the new messages added this run
    pub output: String,                 // Final text output
    pub tool_calls: Vec<ToolCallRecord>,
    pub token_usage: TokenUsage,
    pub turns: usize,
}
```

---

### `AgentRunResult`

High-level result returned by `Agent::run` and `orchestrator.run_agent`.

```rust
pub struct AgentRunResult {
    pub success: bool,
    pub output: String,
    pub messages: Vec<LLMMessage>,
    pub token_usage: TokenUsage,
    pub tool_calls: Vec<ToolCallRecord>,
    pub turns: usize,
    /// Populated only when AgentConfig.output_schema is set and validation succeeds.
    pub structured: Option<serde_json::Value>,
}
```

---

### `ToolCallRecord`

Audit record for a single tool execution.

```rust
pub struct ToolCallRecord {
    pub tool_name: String,
    pub input: HashMap<String, serde_json::Value>,
    pub output: String,
    pub duration_ms: u64,
}
```

---

### `AgentConfig`

Configuration for a single agent.

```rust
pub struct AgentConfig {
    pub name: String,
    pub model: String,                        // Default: "qwen/qwen3.6-plus:free"
    pub provider: Option<String>,             // "anthropic" | "openai" | "openrouter"
    pub base_url: Option<String>,             // Override provider endpoint
    pub api_key: Option<String>,              // Falls back to env var if None
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<String>>,           // Allowed tool names; None = all registered
    pub max_turns: Option<usize>,             // Default: 10
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub output_schema: Option<serde_json::Value>, // JSON Schema for structured output
    pub before_run: Option<BeforeRunHookAsync>,
    pub after_run: Option<AfterRunHookAsync>,
}

impl Default for AgentConfig { ... }
impl std::fmt::Debug for AgentConfig { ... }
```

---

### `AgentStatus`

```rust
pub enum AgentStatus {
    Idle,
    Running,
    Completed,
    Error,
}
```

---

### `AgentInfo`

Context passed to tools identifying the calling agent.

```rust
pub struct AgentInfo {
    pub name: String,
    pub role: String,
    pub model: String,
}
```

---

### `ToolUseContext`

Full execution context passed to every `Tool::execute` call.

```rust
pub struct ToolUseContext {
    pub agent: AgentInfo,
    pub cwd: Option<String>,  // Working directory hint (None by default)
}
```

---

### `ToolResult`

Return value from `Tool::execute`.

```rust
pub struct ToolResult {
    pub data: String,     // Stringified output sent back to the LLM
    pub is_error: bool,   // True → LLM is informed this call failed
}
```

---

### `TeamConfig`

```rust
pub struct TeamConfig {
    pub name: String,
    pub agents: Vec<AgentConfig>,
    pub shared_memory: Option<bool>,       // Default: false
    pub max_concurrency: Option<usize>,    // Agents running in parallel; default: 4
}
```

---

### `Task`

A unit of work in a pipeline.

```rust
pub struct Task {
    pub id: String,                          // UUID v4
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub assignee: Option<String>,            // Agent name
    pub depends_on: Vec<String>,             // Task IDs that must complete first
    pub result: Option<String>,              // Populated after completion
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub max_retries: Option<u32>,            // Default: no retry
    pub retry_delay_ms: Option<u64>,         // Base delay; default: 1000
    pub retry_backoff: Option<f64>,          // Multiplier per attempt; default: 2.0
}
```

---

### `TaskStatus`

```rust
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Blocked,   // Has unresolved dependencies
    Skipped,   // Skipped by approval gate or skip_remaining
}

impl TaskStatus {
    pub fn is_terminal(&self) -> bool;  // Completed | Failed | Skipped → true
}
```

---

### `OrchestratorConfig`

```rust
pub struct OrchestratorConfig {
    pub default_model: String,
    pub default_provider: String,
    pub default_base_url: Option<String>,
    pub default_api_key: Option<String>,
    pub max_concurrency: usize,
    pub on_progress: Option<Box<dyn Fn(String) + Send + Sync>>,
    pub on_trace: Option<OnTraceFn>,
    pub on_approval: Option<Arc<
        dyn Fn(Vec<Task>, Vec<Task>) -> BoxFuture<'static, bool> + Send + Sync
    >>,
}
```

---

### `OrchestratorEvent` / `OrchestratorEventType`

```rust
pub enum OrchestratorEventType {
    AgentStart, AgentComplete,
    TaskStart,  TaskComplete, TaskSkipped, TaskRetry,
    Message, Error,
}

pub struct OrchestratorEvent {
    pub event_type: OrchestratorEventType,
    pub agent: Option<String>,
    pub task:  Option<String>,
    pub data:  Option<String>,
}
```

---

### `BeforeRunHookContext`

```rust
pub struct BeforeRunHookContext {
    pub prompt: String,       // May be modified by the hook
    pub agent_name: String,
    pub agent_model: String,
}
```

---

### Hook Type Aliases

```rust
/// Sync before-run hook.
pub type BeforeRunHook =
    Arc<dyn Fn(BeforeRunHookContext) -> BeforeRunHookContext + Send + Sync>;

/// Async before-run hook. Return Err to abort the run.
pub type BeforeRunHookAsync = Arc<
    dyn Fn(BeforeRunHookContext) -> BoxFuture<'static, Result<BeforeRunHookContext>>
        + Send + Sync,
>;

/// Sync after-run hook.
pub type AfterRunHook = Arc<dyn Fn(AgentRunResult) -> AgentRunResult + Send + Sync>;

/// Async after-run hook. Return Err to abort and mark the run failed.
pub type AfterRunHookAsync = Arc<
    dyn Fn(AgentRunResult) -> BoxFuture<'static, Result<AgentRunResult>>
        + Send + Sync,
>;
```

---

### Trace Types

```rust
pub struct TraceEventBase {
    pub run_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub duration_ms: u64,
    pub agent: String,
    pub task_id: Option<String>,
}

pub struct LlmCallTrace  { pub base: TraceEventBase, pub model: String, pub turn: usize, pub tokens: TokenUsage }
pub struct ToolCallTrace { pub base: TraceEventBase, pub tool: String,  pub is_error: bool }
pub struct TaskTrace     { pub base: TraceEventBase, pub task_id: String, pub task_title: String, pub success: bool, pub retries: u32 }
pub struct AgentTrace    { pub base: TraceEventBase, pub turns: usize, pub tokens: TokenUsage, pub tool_calls: usize }

pub enum TraceEvent {
    LlmCall(LlmCallTrace),
    ToolCall(ToolCallTrace),
    Task(TaskTrace),
    Agent(AgentTrace),
}

pub type OnTraceFn = Arc<dyn Fn(TraceEvent) + Send + Sync>;
```

---

### `MemoryEntry`

```rust
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub created_at: DateTime<Utc>,
}
```

---

### `Message`

```rust
pub struct Message {
    pub id: String,
    pub from: String,
    pub to: String,         // Agent name or "*" for broadcast
    pub content: String,
    pub timestamp: DateTime<Utc>,
}
```

---

## Agent (`agent`)

### `Agent`

```rust
pub struct Agent {
    pub config: AgentConfig,
    pub status: AgentStatus,
    // private: registry, executor, history, total_token_usage
}
```

#### Constructor

```rust
impl Agent {
    pub fn new(
        config: AgentConfig,
        registry: Arc<tokio::sync::Mutex<ToolRegistry>>,
        executor: Arc<ToolExecutor>,
    ) -> Self;
}
```

#### Running

```rust
impl Agent {
    /// Run a one-shot prompt. History is NOT preserved.
    pub async fn run(
        &mut self,
        prompt: &str,
        adapter: Arc<dyn LLMAdapter>,
    ) -> Result<AgentRunResult>;

    /// Run with custom RunOptions (callbacks, trace IDs, etc.).
    pub async fn run_with_opts(
        &mut self,
        prompt: &str,
        adapter: Arc<dyn LLMAdapter>,
        opts: RunOptions,
    ) -> Result<AgentRunResult>;

    /// Append a message and run — history IS preserved across calls.
    pub async fn prompt(
        &mut self,
        message: &str,
        adapter: Arc<dyn LLMAdapter>,
    ) -> Result<AgentRunResult>;

    /// Stream output token-by-token.
    pub fn stream<'a>(
        &'a mut self,
        prompt: &str,
        adapter: Arc<dyn LLMAdapter>,
    ) -> impl futures::Stream<Item = StreamEvent> + 'a;
}
```

#### State

```rust
impl Agent {
    pub fn get_history(&self) -> Vec<LLMMessage>;
    pub fn reset(&mut self);  // Clears history, resets status to Idle
}
```

---

### `RunOptions`

Per-call observation hooks. All fields are optional.

```rust
pub struct RunOptions {
    /// Called just before each tool is dispatched. Receives tool name and input.
    pub on_tool_call: Option<Arc<dyn Fn(&str, &HashMap<String, serde_json::Value>) + Send + Sync>>,
    /// Called after each tool result arrives. Receives tool name and is_error flag.
    pub on_tool_result: Option<Arc<dyn Fn(&str, bool) + Send + Sync>>,
    /// Called after each complete LLMMessage is appended to the conversation.
    pub on_message: Option<Arc<dyn Fn(&LLMMessage) + Send + Sync>>,
    /// Trace callback for observability spans.
    pub on_trace: Option<OnTraceFn>,
    /// Trace correlation ID for the entire run.
    pub run_id: Option<String>,
    /// Task ID for trace correlation (if running inside a task pipeline).
    pub task_id: Option<String>,
    /// Agent name override in trace events (defaults to AgentConfig.name).
    pub trace_agent: Option<String>,
}

impl Default for RunOptions { ... }
```

---

## LLM Adapters (`llm`)

### `LLMAdapter` trait

```rust
pub trait LLMAdapter: Send + Sync {
    fn name(&self) -> &str;

    /// Non-streaming — waits for the full response.
    async fn chat(
        &self,
        messages: &[LLMMessage],
        options: &LLMChatOptions,
    ) -> Result<LLMResponse>;

    /// Streaming — yields Text deltas then one Complete.
    /// Default impl wraps chat() into a one-shot stream.
    fn stream<'a>(
        &'a self,
        messages: &'a [LLMMessage],
        options: &'a LLMChatOptions,
    ) -> LLMStream<'a>;
}

pub type LLMStream<'a> = Pin<Box<dyn Stream<Item = Result<LLMStreamDelta>> + Send + 'a>>;
```

### `create_adapter`

```rust
/// Build a boxed adapter by provider name.
///
/// provider:  "anthropic" | "openai" | "openrouter" (default)
/// api_key:   Falls back to env var for the matching provider.
/// base_url:  Override the provider's default endpoint.
pub fn create_adapter(
    provider: &str,
    api_key: Option<String>,
    base_url: Option<String>,
) -> Box<dyn LLMAdapter>;
```

### `AnthropicAdapter`

```rust
pub struct AnthropicAdapter { /* private */ }

impl AnthropicAdapter {
    pub fn new(api_key: String, base_url: Option<String>) -> Self;
}

impl LLMAdapter for AnthropicAdapter { ... }
```

Wire format: native Anthropic Messages API. System prompt is a top-level field.
Tools use Anthropic's `tool_use` / `tool_result` content block types.

### `OpenAIAdapter`

```rust
pub struct OpenAIAdapter { /* private */ }

impl OpenAIAdapter {
    pub fn new(api_key: String, base_url: Option<String>) -> Self;
}

impl LLMAdapter for OpenAIAdapter { ... }
```

Wire format: OpenAI Chat Completions. System prompt is a `{"role":"system"}` message.
Tools use `tool_calls` array and `tool_call_id` references.

### `OpenRouterAdapter`

```rust
pub struct OpenRouterAdapter { /* private */ }

impl OpenRouterAdapter {
    pub fn new(api_key: String, base_url: String) -> Self;
}

impl LLMAdapter for OpenRouterAdapter { ... }
```

Wire format: OpenAI-compatible. Additionally handles:
- Array-form content (`[{"type":"text","text":"..."}]`) from Qwen3, DeepSeek, etc.
- `reasoning_content` field emitted by reasoning models
- `<think>…</think>` tag stripping
- SSE streaming via `stream: true` (overrides the default `stream()` method)

---

## Tool System (`tool`)

### `Tool` trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;  // JSON Schema object

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> Result<ToolResult>;
}
```

### `ToolRegistry`

```rust
pub struct ToolRegistry { /* private */ }

impl ToolRegistry {
    pub fn new() -> Self;

    /// Register a tool. Returns Err if a tool with the same name already exists.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<()>;

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn has(&self, name: &str) -> bool;
    pub fn list(&self) -> Vec<Arc<dyn Tool>>;
    pub fn unregister(&mut self, name: &str);  // No-op if not found

    /// Convert registered tools to LLM tool definitions.
    /// allowed: Some(names) restricts to that subset; None returns all.
    pub fn to_tool_defs(&self, allowed: Option<&[String]>) -> Vec<LLMToolDef>;
}
```

### `ToolExecutor`

```rust
pub struct ToolExecutor { /* private */ }

impl ToolExecutor {
    pub fn new(registry: Arc<tokio::sync::Mutex<ToolRegistry>>) -> Self;

    /// Execute a named tool. Returns a ToolResult with is_error=true
    /// if the tool is not found or its execute() returns Err.
    pub async fn execute(
        &self,
        name: &str,
        input: &HashMap<String, serde_json::Value>,
        context: &ToolUseContext,
    ) -> ToolResult;
}
```

---

## Orchestrator (`orchestrator`)

### `OpenMultiAgent`

```rust
pub struct OpenMultiAgent { /* private */ }

impl OpenMultiAgent {
    pub fn new(config: OrchestratorConfig) -> Self;

    /// Run a single agent against a prompt.
    pub async fn run_agent(
        &self,
        config: AgentConfig,
        prompt: &str,
    ) -> Result<AgentRunResult>;

    /// Run a team of agents toward a shared goal (goal injected as first task).
    pub async fn run_team(
        &self,
        team: &TeamConfig,
        goal: &str,
    ) -> Result<TeamRunResult>;

    /// Run an explicit set of tasks on a team, respecting dependencies.
    pub async fn run_tasks(
        &self,
        team: &TeamConfig,
        tasks: Vec<Task>,
    ) -> Result<TeamRunResult>;
}
```

### `TeamRunResult`

```rust
pub struct TeamRunResult {
    pub success: bool,
    pub agent_results: HashMap<String, AgentRunResult>,  // keyed by task ID
    pub token_usage: TokenUsage,
}
```

---

## Task Helpers (`task`)

### Free functions

```rust
/// Create a Task with a UUID id and Pending status.
pub fn create_task(
    title: impl Into<String>,
    description: impl Into<String>,
    assignee: Option<String>,
    depends_on: Vec<String>,  // Task IDs
) -> Task;

/// Return tasks in execution order, or Err if a cycle is detected.
pub fn topological_sort(tasks: &[Task]) -> Result<Vec<Task>, String>;

/// Return true if all of a task's dependencies are in the completed set.
pub fn is_task_ready(task: &Task, completed_ids: &HashSet<String>) -> bool;
```

### `TaskQueue`

```rust
pub struct TaskQueue { /* private */ }

impl TaskQueue {
    pub fn new() -> Self;

    pub fn add(&mut self, task: Task);
    pub fn add_batch(&mut self, tasks: Vec<Task>);

    /// Update arbitrary fields. Pass None to leave a field unchanged.
    pub fn update(
        &mut self,
        task_id: &str,
        status: Option<TaskStatus>,
        result: Option<String>,
        assignee: Option<Option<String>>,
    ) -> Result<Task>;

    pub fn complete(&mut self, task_id: &str, result: Option<String>) -> Result<Task>;
    pub fn fail(&mut self, task_id: &str, error: String) -> Result<Task>;
    pub fn skip(&mut self, task_id: &str, reason: String) -> Result<Task>;

    /// Mark all non-terminal tasks as Skipped.
    pub fn skip_remaining(&mut self, reason: &str);

    pub fn list(&self) -> Vec<Task>;
    pub fn get(&self, task_id: &str) -> Option<Task>;
    pub fn get_by_status(&self, status: TaskStatus) -> Vec<Task>;

    /// True when every task is in a terminal state.
    pub fn is_complete(&self) -> bool;
}
```

When a task is marked Completed, the queue automatically unblocks any downstream tasks
whose dependencies are now all satisfied. When a task fails, all tasks that directly or
transitively depend on it are marked Failed.

### `Scheduler`

```rust
pub struct Scheduler { /* private */ }

pub enum SchedulingStrategy {
    RoundRobin,         // Cycle through agents in order
    LeastBusy,          // Assign to the agent with fewest current tasks
    CapabilityMatch,    // Match task description keywords to agent name/role
    DependencyFirst,    // Prioritise tasks that unblock the most others
}

impl Scheduler {
    pub fn new(strategy: SchedulingStrategy) -> Self;

    /// Returns (task_id, agent_name) pairs for all assignable pending tasks.
    pub fn schedule(
        &mut self,
        tasks: &[Task],
        agents: &[AgentConfig],
    ) -> Vec<(String, String)>;
}
```

---

## Retry Helpers (`orchestrator`)

```rust
/// Compute the delay before retry attempt `attempt` (1-based).
/// Capped at 30 000 ms.
///
/// delay = base_delay * backoff^(attempt - 1)
pub fn compute_retry_delay(base_delay: u64, backoff: f64, attempt: u32) -> u64;

/// Run `run` up to task.max_retries + 1 times.
/// Returns the last result (success or failure) once retries are exhausted.
/// Token usage is accumulated across all attempts.
pub async fn execute_with_retry(
    run: impl Fn() -> BoxFuture<'static, Result<AgentRunResult>>,
    task: &Task,
    on_retry: Option<Arc<dyn Fn(
        attempt: u32,
        max: u32,
        error_msg: String,
        delay_ms: u64,
    ) + Send + Sync>>,
) -> AgentRunResult;
```

---

## Memory (`memory`)

### `MemoryStore` trait

```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn get(&self, key: &str) -> Option<MemoryEntry>;
    async fn set(
        &self,
        key: &str,
        value: &str,
        metadata: Option<HashMap<String, serde_json::Value>>,
    );
    async fn list(&self) -> Vec<MemoryEntry>;
    async fn delete(&self, key: &str);
    async fn clear(&self);
}
```

### `InMemoryStore`

```rust
pub struct InMemoryStore { /* private */ }

impl InMemoryStore {
    pub fn new() -> Self;
}

impl MemoryStore for InMemoryStore { ... }
```

### `SharedMemory`

A namespaced wrapper that keys values as `"{agent_name}/{key}"`, enabling agents
to read each other's results without collision.

```rust
pub struct SharedMemory { /* private */ }

impl SharedMemory {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self;

    pub async fn write(&self, agent_name: &str, key: &str, value: &str);
    pub async fn read(&self, agent_name: &str, key: &str) -> Option<MemoryEntry>;
    pub async fn read_all(&self) -> Vec<MemoryEntry>;

    /// Format all entries as a Markdown list (useful as context injection).
    pub async fn to_markdown(&self) -> String;
}
```

---

## Messaging (`messaging`)

### `MessageBus`

Clone-safe; all clones share the same underlying state.

```rust
pub struct MessageBus { /* private Arc */ }

impl MessageBus {
    pub fn new() -> Self;

    /// Send a point-to-point message.
    pub fn send(&self, from: &str, to: &str, content: &str) -> Message;

    /// Send to all agents except the sender.
    pub fn broadcast(&self, from: &str, content: &str) -> Message;

    /// Return messages addressed to `agent_name` that have not been marked read.
    pub fn get_unread(&self, agent_name: &str) -> Vec<Message>;

    /// Return all messages addressed to `agent_name` (read + unread).
    pub fn get_all(&self, agent_name: &str) -> Vec<Message>;

    /// Mark specific message IDs as read for `agent_name`.
    pub fn mark_read(&self, agent_name: &str, message_ids: &[String]);

    /// All messages exchanged between two agents, sorted by time.
    pub fn get_conversation(&self, agent1: &str, agent2: &str) -> Vec<Message>;

    /// Subscribe to incoming messages for `agent_name`.
    /// Returns an unsubscribe closure — call it to stop delivery.
    pub fn subscribe(
        &self,
        agent_name: &str,
        callback: impl Fn(Message) + Send + Sync + 'static,
    ) -> impl FnOnce() + 'static;

    /// Unsubscribe by numeric subscriber ID (returned inside the subscribe closure).
    pub fn unsubscribe(&self, agent_name: &str, subscriber_id: u64);
}
```

---

## Trace / Observability (`trace`)

```rust
/// Fire the trace callback if set. Panics inside the callback are caught and swallowed.
pub fn emit_trace(on_trace: &Option<OnTraceFn>, event: TraceEvent);

/// Generate a UUID v4 string for use as run_id.
pub fn generate_run_id() -> String;

/// Current Unix epoch time in milliseconds.
pub fn now_ms() -> u64;
```
