use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Image { source: ImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

impl ContentBlock {
    pub fn as_text(&self) -> Option<&str> {
        if let ContentBlock::Text { text } = self {
            Some(text.as_str())
        } else {
            None
        }
    }

    pub fn as_tool_use(&self) -> Option<&ToolUseBlock> {
        if let ContentBlock::ToolUse(b) = self {
            Some(b)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// LLM messages & responses
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsage {
    pub fn add(&self, other: &TokenUsage) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: TokenUsage,
}

// ---------------------------------------------------------------------------
// Stream events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Text(String),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Done(RunResult),
    Error(String),
}

// ---------------------------------------------------------------------------
// Trace events
// ---------------------------------------------------------------------------

/// Shared fields present on every trace event.
#[derive(Debug, Clone)]
pub struct TraceEventBase {
    /// Unique identifier for the entire run.
    pub run_id: String,
    /// Unix epoch ms when the span started.
    pub start_ms: u64,
    /// Unix epoch ms when the span ended.
    pub end_ms: u64,
    /// Wall-clock duration in ms.
    pub duration_ms: u64,
    /// Agent name associated with this span.
    pub agent: String,
    /// Task ID if associated with a task.
    pub task_id: Option<String>,
}

/// Emitted for each LLM API call (one per agent turn).
#[derive(Debug, Clone)]
pub struct LlmCallTrace {
    pub base: TraceEventBase,
    pub model: String,
    pub turn: usize,
    pub tokens: TokenUsage,
}

/// Emitted for each tool execution.
#[derive(Debug, Clone)]
pub struct ToolCallTrace {
    pub base: TraceEventBase,
    pub tool: String,
    pub is_error: bool,
}

/// Emitted when a task completes (wraps the full retry sequence).
#[derive(Debug, Clone)]
pub struct TaskTrace {
    pub base: TraceEventBase,
    pub task_id: String,
    pub task_title: String,
    pub success: bool,
    pub retries: u32,
}

/// Emitted when an agent run completes (wraps the full conversation loop).
#[derive(Debug, Clone)]
pub struct AgentTrace {
    pub base: TraceEventBase,
    pub turns: usize,
    pub tokens: TokenUsage,
    pub tool_calls: usize,
}

/// Discriminated union of all trace event types.
#[derive(Debug, Clone)]
pub enum TraceEvent {
    LlmCall(LlmCallTrace),
    ToolCall(ToolCallTrace),
    Task(TaskTrace),
    Agent(AgentTrace),
}

/// Callback type for trace events.
pub type OnTraceFn = Arc<dyn Fn(TraceEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// LLM streaming
// ---------------------------------------------------------------------------

/// An event emitted by `LLMAdapter::stream`.
///
/// `Text` events arrive in real time as the model generates tokens.
/// `Complete` arrives once — at the very end — carrying tool calls, stop
/// reason, and usage. Text content is NOT repeated inside `Complete.content`.
#[derive(Debug, Clone)]
pub enum LLMStreamDelta {
    /// A chunk of generated text (may be a single token or a few words).
    Text(String),
    /// Stream finished. Carries tool-use blocks and metadata.
    Complete(LLMResponse),
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/// Context passed to the `before_run` hook.
#[derive(Debug, Clone)]
pub struct BeforeRunHookContext {
    /// The user prompt text.
    pub prompt: String,
    /// The agent name.
    pub agent_name: String,
    /// The agent model.
    pub agent_model: String,
}

/// Sync before-run hook: receives context, may modify prompt.
pub type BeforeRunHook =
    Arc<dyn Fn(BeforeRunHookContext) -> BeforeRunHookContext + Send + Sync>;

/// Async before-run hook.
pub type BeforeRunHookAsync = Arc<
    dyn Fn(BeforeRunHookContext) -> BoxFuture<'static, Result<BeforeRunHookContext>>
        + Send
        + Sync,
>;

/// Sync after-run hook: receives result, may modify it.
pub type AfterRunHook = Arc<dyn Fn(AgentRunResult) -> AgentRunResult + Send + Sync>;

/// Async after-run hook.
pub type AfterRunHookAsync = Arc<
    dyn Fn(AgentRunResult) -> BoxFuture<'static, Result<AgentRunResult>> + Send + Sync,
>;

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct LLMToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolUseContext {
    pub agent: AgentInfo,
    pub cwd: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub role: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub data: String,
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<String>>,
    pub max_turns: Option<usize>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// JSON schema (as serde_json::Value) for structured output validation.
    pub output_schema: Option<serde_json::Value>,
    /// Called before each run; may modify the prompt.
    pub before_run: Option<BeforeRunHookAsync>,
    /// Called after each successful run; may modify the result.
    pub after_run: Option<AfterRunHookAsync>,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("provider", &self.provider)
            .field("system_prompt", &self.system_prompt)
            .field("max_turns", &self.max_turns)
            .field("output_schema", &self.output_schema)
            .field("before_run", &self.before_run.as_ref().map(|_| "<hook>"))
            .field("after_run", &self.after_run.as_ref().map(|_| "<hook>"))
            .finish()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            name: "agent".to_string(),
            model: "qwen/qwen3.6-plus:free".to_string(),
            provider: None,
            base_url: None,
            api_key: None,
            system_prompt: None,
            tools: None,
            max_turns: Some(10),
            max_tokens: None,
            temperature: None,
            output_schema: None,
            before_run: None,
            after_run: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    Idle,
    Running,
    Completed,
    Error,
}

#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub input: HashMap<String, serde_json::Value>,
    pub output: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct AgentRunResult {
    pub success: bool,
    pub output: String,
    pub messages: Vec<LLMMessage>,
    pub token_usage: TokenUsage,
    pub tool_calls: Vec<ToolCallRecord>,
    /// Total LLM turns in this run.
    pub turns: usize,
    /// Parsed structured output when `output_schema` is set.
    pub structured: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub messages: Vec<LLMMessage>,
    pub output: String,
    pub tool_calls: Vec<ToolCallRecord>,
    pub token_usage: TokenUsage,
    pub turns: usize,
}

// ---------------------------------------------------------------------------
// Team types
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TeamConfig {
    pub name: String,
    pub agents: Vec<AgentConfig>,
    pub shared_memory: Option<bool>,
    pub max_concurrency: Option<usize>,
}

// ---------------------------------------------------------------------------
// Task types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Blocked,
    Skipped,
}

impl TaskStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Skipped)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub assignee: Option<String>,
    pub depends_on: Vec<String>,
    pub result: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub max_retries: Option<u32>,
    pub retry_delay_ms: Option<u64>,
    pub retry_backoff: Option<f64>,
}

// ---------------------------------------------------------------------------
// Orchestrator types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum OrchestratorEventType {
    AgentStart,
    AgentComplete,
    TaskStart,
    TaskComplete,
    TaskSkipped,
    TaskRetry,
    Message,
    Error,
}

#[derive(Debug, Clone)]
pub struct OrchestratorEvent {
    pub event_type: OrchestratorEventType,
    pub agent: Option<String>,
    pub task: Option<String>,
    pub data: Option<String>,
}

// ---------------------------------------------------------------------------
// Memory types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// LLM chat options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LLMChatOptions {
    pub model: String,
    pub tools: Option<Vec<LLMToolDef>>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system_prompt: Option<String>,
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_add() {
        let a = TokenUsage { input_tokens: 10, output_tokens: 20 };
        let b = TokenUsage { input_tokens: 5,  output_tokens: 15 };
        let c = a.add(&b);
        assert_eq!(c.input_tokens,  15);
        assert_eq!(c.output_tokens, 35);
    }

    #[test]
    fn token_usage_add_zero() {
        let a = TokenUsage { input_tokens: 100, output_tokens: 200 };
        let z = TokenUsage::default();
        let c = a.add(&z);
        assert_eq!(c.input_tokens,  100);
        assert_eq!(c.output_tokens, 200);
    }

    #[test]
    fn content_block_as_text_some() {
        let b = ContentBlock::Text { text: "hello".to_string() };
        assert_eq!(b.as_text(), Some("hello"));
    }

    #[test]
    fn content_block_as_text_none_for_tool_result() {
        let b = ContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: "x".to_string(),
            content: "data".to_string(),
            is_error: None,
        });
        assert_eq!(b.as_text(), None);
    }

    #[test]
    fn content_block_as_tool_use_some() {
        let tu = ToolUseBlock { id: "1".to_string(), name: "bash".to_string(), input: HashMap::new() };
        let b = ContentBlock::ToolUse(tu.clone());
        assert!(b.as_tool_use().is_some());
        assert_eq!(b.as_tool_use().unwrap().name, "bash");
    }

    #[test]
    fn content_block_as_tool_use_none_for_text() {
        let b = ContentBlock::Text { text: "x".to_string() };
        assert!(b.as_tool_use().is_none());
    }

    #[test]
    fn task_status_is_terminal() {
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(TaskStatus::Skipped.is_terminal());
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::InProgress.is_terminal());
        assert!(!TaskStatus::Blocked.is_terminal());
    }

    #[test]
    fn role_serialization() {
        let msg = LLMMessage {
            role: Role::User,
            content: vec![],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"user\""));
    }

    #[test]
    fn agent_run_result_has_structured_field() {
        let r = AgentRunResult {
            success: true,
            output: "{}".to_string(),
            messages: vec![],
            token_usage: TokenUsage::default(),
            tool_calls: vec![],
            turns: 1,
            structured: Some(serde_json::json!({"key": "value"})),
        };
        assert!(r.structured.is_some());
    }

    #[test]
    fn trace_event_variants() {
        let base = TraceEventBase {
            run_id: "run-1".to_string(),
            start_ms: 0,
            end_ms: 100,
            duration_ms: 100,
            agent: "agent-a".to_string(),
            task_id: None,
        };
        let event = TraceEvent::LlmCall(LlmCallTrace {
            base: base.clone(),
            model: "test-model".to_string(),
            turn: 1,
            tokens: TokenUsage::default(),
        });
        assert!(matches!(event, TraceEvent::LlmCall(_)));

        let event2 = TraceEvent::ToolCall(ToolCallTrace {
            base,
            tool: "bash".to_string(),
            is_error: false,
        });
        assert!(matches!(event2, TraceEvent::ToolCall(_)));
    }
}
