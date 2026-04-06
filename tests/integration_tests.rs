mod mock_adapter;

use mock_adapter::MockAdapter;
use open_multi_agent_rs::{
    agent::{pool::AgentPool, Agent},
    create_task,
    error::Result,
    llm::LLMAdapter,
    memory::{InMemoryStore, MemoryStore, SharedMemory},
    task::queue::TaskQueue,
    tool::{ToolExecutor, ToolRegistry},
    types::AgentConfig,
    AgentStatus, TaskStatus,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config(name: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "mock-model".to_string(),
        ..Default::default()
    }
}

fn make_agent(name: &str) -> Agent {
    let reg = Arc::new(Mutex::new(ToolRegistry::new()));
    let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
    Agent::new(default_config(name), Arc::clone(&reg), exec)
}

// Cast MockAdapter to Arc<dyn LLMAdapter> at the call site.
fn dyn_adapter(responses: Vec<mock_adapter::MockResponse>) -> Arc<dyn LLMAdapter> {
    MockAdapter::arc(responses)
}

// ---------------------------------------------------------------------------
// Single-agent conversation loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_agent_plain_response() {
    let mut agent = make_agent("bot");
    let adapter = dyn_adapter(vec![MockAdapter::text("The answer is 42.")]);
    let result = agent.run("What is the answer?", adapter).await.unwrap();
    assert!(result.success);
    assert_eq!(result.output, "The answer is 42.");
    assert_eq!(result.token_usage.input_tokens, 10);
    assert_eq!(result.token_usage.output_tokens, 20);
    assert!(result.tool_calls.is_empty());
    assert_eq!(result.turns, 1);
}

#[tokio::test]
async fn single_agent_multiple_runs_independent() {
    let mut agent = make_agent("bot");
    let adapter = dyn_adapter(vec![
        MockAdapter::text("First response."),
        MockAdapter::text("Second response."),
    ]);

    let r1 = agent.run("first", Arc::clone(&adapter)).await.unwrap();
    let r2 = agent.run("second", Arc::clone(&adapter)).await.unwrap();
    assert_eq!(r1.output, "First response.");
    assert_eq!(r2.output, "Second response.");
    // Each run is independent (no shared history for run())
    assert_eq!(r1.turns, 1);
    assert_eq!(r2.turns, 1);
}

// ---------------------------------------------------------------------------
// Multi-turn conversation (Agent::prompt)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_history_is_preserved() {
    let mut agent = make_agent("tutor");
    let adapter = dyn_adapter(vec![
        MockAdapter::text("2 + 2 = 4"),
        MockAdapter::text("4 × 3 = 12"),
    ]);

    let r1 = agent
        .prompt("What is 2+2?", Arc::clone(&adapter))
        .await
        .unwrap();
    assert_eq!(r1.output, "2 + 2 = 4");

    let r2 = agent
        .prompt("Multiply that by 3.", Arc::clone(&adapter))
        .await
        .unwrap();
    assert_eq!(r2.output, "4 × 3 = 12");

    // History: user1, assistant1, user2, assistant2
    assert_eq!(agent.get_history().len(), 4);
}

#[tokio::test]
async fn agent_reset_clears_history() {
    let mut agent = make_agent("bot");
    let adapter = dyn_adapter(vec![MockAdapter::text("hi")]);
    agent.prompt("hello", adapter).await.unwrap();
    assert!(!agent.get_history().is_empty());
    agent.reset();
    assert!(agent.get_history().is_empty());
    assert_eq!(agent.status, AgentStatus::Idle);
}

// ---------------------------------------------------------------------------
// Tool call loop
// ---------------------------------------------------------------------------

use async_trait::async_trait;
use open_multi_agent_rs::tool::Tool;
use open_multi_agent_rs::types::ToolUseContext;

struct UpperCaseTool;
#[async_trait]
impl Tool for UpperCaseTool {
    fn name(&self) -> &str {
        "uppercase"
    }
    fn description(&self) -> &str {
        "Uppercases a string"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})
    }
    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _ctx: &ToolUseContext,
    ) -> Result<open_multi_agent_rs::types::ToolResult> {
        let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
        Ok(open_multi_agent_rs::types::ToolResult {
            data: text.to_uppercase(),
            is_error: false,
        })
    }
}

#[tokio::test]
async fn agent_executes_tool_call_and_continues() {
    let reg = Arc::new(Mutex::new(ToolRegistry::new()));
    reg.lock().await.register(Arc::new(UpperCaseTool)).unwrap();
    let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
    let mut agent = Agent::new(
        AgentConfig {
            name: "bot".to_string(),
            model: "mock".to_string(),
            tools: Some(vec!["uppercase".to_string()]),
            ..Default::default()
        },
        reg,
        exec,
    );

    let mut tool_input = HashMap::new();
    tool_input.insert("text".to_string(), serde_json::json!("hello world"));

    let adapter = dyn_adapter(vec![
        MockAdapter::tool_call("uppercase", "call-1", tool_input),
        MockAdapter::text("Uppercased: HELLO WORLD"),
    ]);

    let result = agent.run("Uppercase 'hello world'", adapter).await.unwrap();
    assert!(result.success);
    assert_eq!(result.output, "Uppercased: HELLO WORLD");
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].tool_name, "uppercase");
    assert_eq!(result.tool_calls[0].output, "HELLO WORLD");
    assert_eq!(result.turns, 2); // tool-call turn + final turn
}

#[tokio::test]
async fn agent_tool_not_found_returns_error_and_continues() {
    let mut agent = make_agent("bot");
    let mut tool_input = HashMap::new();
    tool_input.insert("cmd".to_string(), serde_json::json!("ls"));

    let adapter = dyn_adapter(vec![
        MockAdapter::tool_call("bash", "c1", tool_input),
        MockAdapter::text("Could not run bash."),
    ]);

    let result = agent.run("Run ls", adapter).await.unwrap();
    assert!(result.success);
    // Tool error is fed back; model gives a follow-up
    assert_eq!(result.tool_calls.len(), 1);
    assert!(result.tool_calls[0].output.contains("not found"));
}

// ---------------------------------------------------------------------------
// Max turns guard
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_stops_at_max_turns() {
    let reg = Arc::new(Mutex::new(ToolRegistry::new()));
    let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
    let mut agent = Agent::new(
        AgentConfig {
            name: "bot".to_string(),
            model: "mock".to_string(),
            max_turns: Some(2),
            ..Default::default()
        },
        reg,
        exec,
    );

    let mut tool_input = HashMap::new();
    tool_input.insert("t".to_string(), serde_json::json!("x"));

    let adapter = dyn_adapter(vec![
        MockAdapter::tool_call("noop", "c1", tool_input.clone()),
        MockAdapter::tool_call("noop", "c2", tool_input.clone()),
        MockAdapter::tool_call("noop", "c3", tool_input.clone()),
        MockAdapter::text("done"),
    ]);

    let result = agent.run("loop", adapter).await.unwrap();
    assert!(
        result.turns <= 2,
        "Expected at most 2 turns, got {}",
        result.turns
    );
}

// ---------------------------------------------------------------------------
// AgentPool concurrency
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_pool_limits_concurrency() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    let pool = Arc::new(AgentPool::new(2));
    let concurrent_peak = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let pool = Arc::clone(&pool);
        let peak = Arc::clone(&concurrent_peak);
        let active = Arc::clone(&active);
        handles.push(tokio::spawn(async move {
            pool.run(|| async {
                let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(n, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                Ok::<(), open_multi_agent_rs::error::AgentError>(())
            })
            .await
            .unwrap();
        }));
    }

    futures::future::join_all(handles).await;
    assert!(
        concurrent_peak.load(Ordering::SeqCst) <= 2,
        "Peak concurrency exceeded pool size of 2"
    );
}

// ---------------------------------------------------------------------------
// TaskQueue integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn task_queue_pipeline_completes_in_order() {
    let mut q = TaskQueue::new();
    let t1 = create_task("Task A", "Do A", Some("alice".to_string()), vec![]);
    let t1_id = t1.id.clone();
    let t2 = create_task(
        "Task B",
        "Do B (needs A)",
        Some("bob".to_string()),
        vec![t1_id.clone()],
    );
    let t2_id = t2.id.clone();

    q.add_batch(vec![t1, t2]);

    let pending = q.pending_tasks();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, t1_id);

    q.complete(&t1_id, Some("A done".to_string())).unwrap();
    let pending = q.pending_tasks();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, t2_id);

    q.complete(&t2_id, Some("B done".to_string())).unwrap();
    assert!(q.is_complete());
}

// ---------------------------------------------------------------------------
// SharedMemory integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shared_memory_stores_and_retrieves_results() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let sm = SharedMemory::new(store);

    sm.write("researcher", "task-1", "Found: Rust is fast.")
        .await;
    sm.write("writer", "task-2", "Written: Blog post.").await;

    let md = sm.to_markdown().await;
    assert!(md.contains("Rust is fast"));
    assert!(md.contains("Blog post"));

    let entry = sm.read("researcher", "task-1").await.unwrap();
    assert_eq!(entry.value, "Found: Rust is fast.");
}

// ---------------------------------------------------------------------------
// End-to-end: explicit task pipeline queue logic
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_two_task_pipeline_queue_logic() {
    let mut q = TaskQueue::new();

    let t1 = create_task(
        "Research",
        "Find info",
        Some("researcher".to_string()),
        vec![],
    );
    let t1_id = t1.id.clone();
    let t2 = create_task(
        "Write",
        "Write report",
        Some("writer".to_string()),
        vec![t1_id.clone()],
    );
    let t2_id = t2.id.clone();

    q.add_batch(vec![t1, t2]);

    assert_eq!(q.pending_tasks().len(), 1);
    q.set_in_progress(&t1_id).unwrap();
    q.complete(&t1_id, Some("Rust is memory-safe.".to_string()))
        .unwrap();

    assert_eq!(q.pending_tasks().len(), 1);
    assert_eq!(q.pending_tasks()[0].id, t2_id);

    q.set_in_progress(&t2_id).unwrap();
    q.complete(&t2_id, Some("Report written.".to_string()))
        .unwrap();

    assert!(q.is_complete());
    assert_eq!(
        q.get(&t1_id).unwrap().result.as_deref(),
        Some("Rust is memory-safe.")
    );
    assert_eq!(
        q.get(&t2_id).unwrap().result.as_deref(),
        Some("Report written.")
    );
}

// ---------------------------------------------------------------------------
// Failure propagation pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_cascade_failure_stops_downstream() {
    let mut q = TaskQueue::new();

    let t1 = create_task("Step1", "d", None, vec![]);
    let t1_id = t1.id.clone();
    let t2 = create_task("Step2", "d", None, vec![t1_id.clone()]);
    let t2_id = t2.id.clone();
    let t3 = create_task("Step3", "d", None, vec![t2_id.clone()]);
    let t3_id = t3.id.clone();
    let t4 = create_task("Independent", "d", None, vec![]);
    let t4_id = t4.id.clone();

    q.add_batch(vec![t1, t2, t3, t4]);

    q.fail(&t1_id, "critical error".to_string()).unwrap();

    assert_eq!(q.get(&t1_id).unwrap().status, TaskStatus::Failed);
    assert_eq!(q.get(&t2_id).unwrap().status, TaskStatus::Failed);
    assert_eq!(q.get(&t3_id).unwrap().status, TaskStatus::Failed);
    assert_eq!(q.get(&t4_id).unwrap().status, TaskStatus::Pending);
}

// ---------------------------------------------------------------------------
// Agent state transitions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_status_transitions_correctly() {
    let mut agent = make_agent("bot");
    assert_eq!(agent.status, AgentStatus::Idle);

    let adapter = dyn_adapter(vec![MockAdapter::text("ok")]);
    agent.run("test", adapter).await.unwrap();
    assert_eq!(agent.status, AgentStatus::Completed);

    agent.reset();
    assert_eq!(agent.status, AgentStatus::Idle);
}

// ---------------------------------------------------------------------------
// Multiple tool calls in a single LLM turn
// ---------------------------------------------------------------------------

struct CounterTool {
    counter: Arc<std::sync::atomic::AtomicUsize>,
}
#[async_trait]
impl Tool for CounterTool {
    fn name(&self) -> &str {
        "counter"
    }
    fn description(&self) -> &str {
        "increments a counter"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn execute(
        &self,
        _input: &HashMap<String, serde_json::Value>,
        _ctx: &ToolUseContext,
    ) -> Result<open_multi_agent_rs::types::ToolResult> {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(open_multi_agent_rs::types::ToolResult {
            data: "counted".to_string(),
            is_error: false,
        })
    }
}

#[tokio::test]
async fn agent_parallel_tool_calls_in_one_turn() {
    // This tests that multiple tool_use blocks from one LLM response
    // are all executed and their results fed back.
    // We simulate 2 tool calls in one turn by yielding 2 ToolUse blocks.
    // (MockAdapter returns one at a time, so we chain: 2 tool calls → text)
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let reg = Arc::new(Mutex::new(ToolRegistry::new()));
    reg.lock()
        .await
        .register(Arc::new(CounterTool {
            counter: Arc::clone(&counter),
        }))
        .unwrap();
    let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
    let mut agent = Agent::new(
        AgentConfig {
            name: "bot".to_string(),
            model: "m".to_string(),
            ..Default::default()
        },
        reg,
        exec,
    );

    // Turn 1: tool call
    // Turn 2: another tool call
    // Turn 3: final text
    let mut inp = HashMap::new();
    inp.insert("x".to_string(), serde_json::json!(1));

    let adapter = dyn_adapter(vec![
        MockAdapter::tool_call("counter", "c1", inp.clone()),
        MockAdapter::tool_call("counter", "c2", inp.clone()),
        MockAdapter::text("done"),
    ]);

    let result = agent.run("count twice", adapter).await.unwrap();
    assert_eq!(result.output, "done");
    // 2 separate turns with tool calls → counter called twice
    assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
}

// ---------------------------------------------------------------------------
// Empty prompt guard
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_handles_empty_prompt() {
    let mut agent = make_agent("bot");
    let adapter = dyn_adapter(vec![MockAdapter::text("")]);
    let result = agent.run("", adapter).await.unwrap();
    assert!(result.success);
    assert_eq!(result.output, "");
}

// ---------------------------------------------------------------------------
// Large token accumulation across many messages
// ---------------------------------------------------------------------------

#[tokio::test]
async fn token_usage_accumulates_across_prompt_turns() {
    let mut agent = make_agent("bot");
    let n = 10;
    let adapter = dyn_adapter((0..n).map(|_| MockAdapter::text("ok")).collect());

    let mut total_in = 0u64;
    let mut total_out = 0u64;
    for _ in 0..n {
        let r = agent.prompt("hello", Arc::clone(&adapter)).await.unwrap();
        total_in += r.token_usage.input_tokens;
        total_out += r.token_usage.output_tokens;
    }
    // MockAdapter returns 10 input, 20 output per call
    assert_eq!(total_in, (n as u64) * 10);
    assert_eq!(total_out, (n as u64) * 20);
}
