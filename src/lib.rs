pub mod agent;
pub mod error;
pub mod llm;
pub mod memory;
pub mod messaging;
pub mod orchestrator;
pub mod task;
pub mod tool;
pub mod trace;
pub mod types;

// Orchestrator
pub use orchestrator::{
    compute_retry_delay, execute_with_retry, OpenMultiAgent, OrchestratorConfig, TeamRunResult,
};

// Types
pub use types::{
    AgentConfig, AgentRunResult, AgentStatus, AgentTrace, BeforeRunHookContext, LlmCallTrace,
    OnTraceFn, RunResult, StreamEvent, Task, TaskStatus, TaskTrace, TeamConfig, ToolCallTrace,
    TokenUsage, TraceEvent, TraceEventBase,
};

// Agent
pub use agent::{Agent, RunOptions};

// Task
pub use task::{create_task, topological_sort, queue::TaskQueue};
pub use task::scheduler::{Scheduler, SchedulingStrategy};

// Memory
pub use memory::{InMemoryStore, SharedMemory};

// Messaging
pub use messaging::{Message, MessageBus};

// Tool
pub use tool::{Tool, ToolRegistry, ToolExecutor};
pub use tool::built_in::{
    register_built_in_tools,
    BashTool, FileReadTool, FileWriteTool, FileUpdateTool, FileDeleteTool,
    FileListTool, FileMoveTool, DirCreateTool, DirDeleteTool, GrepTool,
    PythonWriteTool, PythonRunTool, PythonTestTool,
    RepoIngestTool,
};

// LLM adapters
pub use llm::{create_adapter, LLMAdapter};
pub use llm::anthropic::AnthropicAdapter;
pub use llm::openai::OpenAIAdapter;
pub use llm::openrouter::OpenRouterAdapter;

// Trace utilities
pub use trace::{emit_trace, generate_run_id, now_ms};

// Error
pub use error::{AgentError, Result};
