/// Stress tests for open-multi-agent-rs.
///
/// These tests verify correctness and stability under high load:
/// - Many concurrent agent runs
/// - Large task queues with complex dependency graphs
/// - Concurrent memory store access
/// - Rapid queue operations
/// - Scheduler performance with large workloads
/// - New-feature stress: streaming, trace events, MessageBus, retry
mod mock_adapter;

use mock_adapter::MockAdapter;
use open_multi_agent_rs::{
    agent::{pool::AgentPool, Agent},
    create_task,
    memory::{InMemoryStore, MemoryStore, SharedMemory},
    messaging::MessageBus,
    orchestrator::{compute_retry_delay, execute_with_retry},
    task::{
        queue::TaskQueue,
        scheduler::{Scheduler, SchedulingStrategy},
    },
    tool::{ToolExecutor, ToolRegistry},
    types::{AgentConfig, AgentRunResult, Task, TaskStatus, TokenUsage, TraceEvent},
    RunOptions,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config(name: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "mock".to_string(),
        max_turns: Some(3),
        ..Default::default()
    }
}

fn make_agent(name: &str) -> Agent {
    let reg = Arc::new(Mutex::new(ToolRegistry::new()));
    let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
    Agent::new(default_config(name), reg, exec)
}

fn agent_config(name: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "mock".to_string(),
        system_prompt: Some(format!("{} system prompt", name)),
        ..Default::default()
    }
}

#[allow(dead_code)]
fn task_with_id(id: &str, deps: Vec<&str>) -> Task {
    let mut t = create_task(id, id, None, deps.iter().map(|s| s.to_string()).collect());
    t.id = id.to_string();
    t
}

// ---------------------------------------------------------------------------
// Stress 1: 50 concurrent agent runs via AgentPool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_concurrent_agent_pool_50_runs() {
    let pool = Arc::new(AgentPool::new(10));
    let success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..50)
        .map(|_| {
            let pool = Arc::clone(&pool);
            let count = Arc::clone(&success_count);
            tokio::spawn(async move {
                pool.run(|| async move {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    count.fetch_add(1, Ordering::Relaxed);
                    Ok::<(), open_multi_agent_rs::error::AgentError>(())
                })
                .await
                .unwrap();
            })
        })
        .collect();

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let n = success_count.load(Ordering::Relaxed);
    println!(
        "[stress] 50 agent runs via pool(10): {} succeeded in {:?}",
        n, elapsed
    );
    assert_eq!(n, 50);
    assert!(elapsed < Duration::from_secs(2));
}

// ---------------------------------------------------------------------------
// Stress 2: Large task queue (200 tasks, linear chain)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_large_linear_task_queue_200() {
    let n = 200;
    let mut q = TaskQueue::new();
    let ids: Vec<String> = (0..n).map(|i| format!("task-{}", i)).collect();

    let tasks: Vec<Task> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let deps = if i == 0 {
                vec![]
            } else {
                vec![ids[i - 1].clone()]
            };
            let mut t = create_task(id, id, None, deps);
            t.id = id.clone();
            t
        })
        .collect();

    let start = Instant::now();
    q.add_batch(tasks);

    for id in &ids {
        let pending = q.pending_tasks();
        assert_eq!(pending.len(), 1);
        assert_eq!(&pending[0].id, id);
        q.complete(id, Some("done".to_string())).unwrap();
    }

    let elapsed = start.elapsed();
    assert!(q.is_complete());
    println!("[stress] 200-task linear chain executed in {:?}", elapsed);
    assert!(elapsed < Duration::from_secs(2));
}

// ---------------------------------------------------------------------------
// Stress 3: Fan-out graph (1 root → 50 leaves → 1 sink)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_fan_out_task_graph() {
    let mut q = TaskQueue::new();

    let mut root = create_task("root", "root", None, vec![]);
    root.id = "root".to_string();

    let leaf_count = 50;
    let leaves: Vec<Task> = (0..leaf_count)
        .map(|i| {
            let mut t = create_task(
                &format!("leaf-{}", i),
                "leaf",
                None,
                vec!["root".to_string()],
            );
            t.id = format!("leaf-{}", i);
            t
        })
        .collect();

    let leaf_ids: Vec<String> = leaves.iter().map(|t| t.id.clone()).collect();

    let mut sink = create_task("sink", "sink", None, leaf_ids.clone());
    sink.id = "sink".to_string();

    let start = Instant::now();
    q.add(root.clone());
    q.add_batch(leaves);
    q.add(sink);

    assert_eq!(q.pending_tasks().len(), 1);
    q.complete("root", None).unwrap();
    assert_eq!(q.pending_tasks().len(), leaf_count);

    for lid in &leaf_ids {
        q.complete(lid, None).unwrap();
    }
    assert_eq!(q.pending_tasks().len(), 1);
    q.complete("sink", None).unwrap();
    assert!(q.is_complete());

    let elapsed = start.elapsed();
    println!("[stress] fan-out graph (1→50→1) in {:?}", elapsed);
    assert!(elapsed < Duration::from_secs(1));
}

// ---------------------------------------------------------------------------
// Stress 4: Cascade failure across 100 dependent tasks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_cascade_failure_100_dependents() {
    let n = 100;
    let mut q = TaskQueue::new();
    let ids: Vec<String> = (0..n).map(|i| format!("t{}", i)).collect();

    let root = {
        let mut t = create_task("t0", "root", None, vec![]);
        t.id = "t0".to_string();
        t
    };
    let dependents: Vec<Task> = (1..n)
        .map(|i| {
            let mut t = create_task(&ids[i], "dep", None, vec!["t0".to_string()]);
            t.id = ids[i].clone();
            t
        })
        .collect();

    q.add(root);
    q.add_batch(dependents);

    let start = Instant::now();
    q.fail("t0", "root failed".to_string()).unwrap();
    let elapsed = start.elapsed();

    let failed_count = q
        .list()
        .iter()
        .filter(|t| t.status == TaskStatus::Failed)
        .count();
    println!(
        "[stress] cascade fail for 100 tasks in {:?}: {} failed",
        elapsed, failed_count
    );
    assert_eq!(failed_count, n);
    assert!(elapsed < Duration::from_secs(1));
}

// ---------------------------------------------------------------------------
// Stress 5: 1000 rapid queue operations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_rapid_queue_operations_1000() {
    let mut q = TaskQueue::new();
    let start = Instant::now();

    let tasks: Vec<Task> = (0..1000)
        .map(|i| {
            let mut t = create_task(&format!("t{}", i), "task", None, vec![]);
            t.id = format!("t{}", i);
            t
        })
        .collect();

    q.add_batch(tasks);
    assert_eq!(q.pending_tasks().len(), 1000);

    for i in 0..1000 {
        q.complete(&format!("t{}", i), Some(format!("result-{}", i)))
            .unwrap();
    }

    let elapsed = start.elapsed();
    assert!(q.is_complete());
    println!("[stress] 1000 rapid queue ops in {:?}", elapsed);
    assert!(elapsed < Duration::from_secs(2));
}

// ---------------------------------------------------------------------------
// Stress 6: Concurrent InMemoryStore (100 concurrent)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_concurrent_memory_store_100() {
    let store = Arc::new(InMemoryStore::new());
    let write_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..100)
        .map(|i| {
            let store = Arc::clone(&store);
            let count = Arc::clone(&write_count);
            tokio::spawn(async move {
                let key = format!("key-{}", i);
                let value = format!("value-{}", i);
                store.set(&key, &value, None).await;
                let entry = store.get(&key).await;
                assert!(entry.is_some());
                assert_eq!(entry.unwrap().value, value);
                count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let n = write_count.load(Ordering::Relaxed);
    println!(
        "[stress] 100 concurrent memory store ops in {:?}: {} completed",
        elapsed, n
    );
    assert_eq!(n, 100);
    assert_eq!(store.list().await.len(), 100);
}

// ---------------------------------------------------------------------------
// Stress 7: SharedMemory under concurrent agent writes (50 agents)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_shared_memory_concurrent_writes_50_agents() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let sm = Arc::new(SharedMemory::new(store));
    let start = Instant::now();

    let handles: Vec<_> = (0..50)
        .map(|i| {
            let sm = Arc::clone(&sm);
            tokio::spawn(async move {
                let agent = format!("agent-{}", i);
                let key = format!("result-{}", i);
                let value = format!("output from agent {}", i);
                sm.write(&agent, &key, &value).await;
                let read = sm.read(&agent, &key).await;
                assert!(read.is_some());
                assert_eq!(read.unwrap().value, value);
            })
        })
        .collect();

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let all = sm.read_all().await;
    println!(
        "[stress] 50 concurrent shared memory writes in {:?}: {} entries",
        elapsed,
        all.len()
    );
    assert_eq!(all.len(), 50);
}

// ---------------------------------------------------------------------------
// Stress 8: Scheduler performance — 500 tasks, 10 agents
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_scheduler_500_tasks_10_agents() {
    let agents: Vec<AgentConfig> = (0..10)
        .map(|i| agent_config(&format!("agent-{}", i)))
        .collect();

    let tasks: Vec<Task> = (0..500)
        .map(|i| {
            let mut t = create_task(
                &format!("task-{}", i),
                &format!("Do task {}", i),
                None,
                vec![],
            );
            t.id = format!("task-{}", i);
            t
        })
        .collect();

    let start = Instant::now();
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let assignments = s.schedule(&tasks, &agents);
    let elapsed_rr = start.elapsed();
    assert_eq!(assignments.len(), 500);
    println!("[stress] RoundRobin schedule 500/10 in {:?}", elapsed_rr);

    let start = Instant::now();
    let mut s = Scheduler::new(SchedulingStrategy::LeastBusy);
    let assignments = s.schedule(&tasks, &agents);
    let elapsed_lb = start.elapsed();
    assert_eq!(assignments.len(), 500);
    println!("[stress] LeastBusy schedule 500/10 in {:?}", elapsed_lb);

    let start = Instant::now();
    let mut s = Scheduler::new(SchedulingStrategy::CapabilityMatch);
    let assignments = s.schedule(&tasks, &agents);
    let elapsed_cm = start.elapsed();
    assert_eq!(assignments.len(), 500);
    println!(
        "[stress] CapabilityMatch schedule 500/10 in {:?}",
        elapsed_cm
    );

    let start = Instant::now();
    let mut s = Scheduler::new(SchedulingStrategy::DependencyFirst);
    let assignments = s.schedule(&tasks, &agents);
    let elapsed_df = start.elapsed();
    assert_eq!(assignments.len(), 500);
    println!(
        "[stress] DependencyFirst schedule 500/10 in {:?}",
        elapsed_df
    );

    for (name, elapsed) in [
        ("RR", elapsed_rr),
        ("LB", elapsed_lb),
        ("CM", elapsed_cm),
        ("DF", elapsed_df),
    ] {
        assert!(
            elapsed < Duration::from_secs(1),
            "{} took too long: {:?}",
            name,
            elapsed
        );
    }
}

// ---------------------------------------------------------------------------
// Stress 9: Topological sort with complex layered DAG
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_topological_sort_complex_dag() {
    use open_multi_agent_rs::task::topological_sort;

    let layers = 20;
    let width = 3;
    let mut all_tasks: Vec<Task> = Vec::new();
    let mut prev_ids: Vec<String> = Vec::new();

    for layer in 0..layers {
        let mut cur_ids = Vec::new();
        for w in 0..width {
            let id = format!("l{}-w{}", layer, w);
            let deps = prev_ids.clone();
            let mut t = create_task(&id, &id, None, deps);
            t.id = id.clone();
            all_tasks.push(t);
            cur_ids.push(id);
        }
        prev_ids = cur_ids;
    }

    let start = Instant::now();
    let sorted = topological_sort(&all_tasks).unwrap();
    let elapsed = start.elapsed();

    println!(
        "[stress] topological sort of {}-node layered DAG in {:?}",
        all_tasks.len(),
        elapsed
    );
    assert_eq!(sorted.len(), all_tasks.len());
    assert!(elapsed < Duration::from_secs(1));
}

// ---------------------------------------------------------------------------
// Stress 10: TokenUsage accumulation (10_000 additions)
// ---------------------------------------------------------------------------

#[test]
fn stress_token_usage_accumulation_10k() {
    let start = Instant::now();
    let mut total = TokenUsage::default();
    for _ in 0..10_000 {
        let chunk = TokenUsage {
            input_tokens: 100,
            output_tokens: 200,
        };
        total = total.add(&chunk);
    }
    let elapsed = start.elapsed();
    println!("[stress] 10k TokenUsage.add() in {:?}", elapsed);
    assert_eq!(total.input_tokens, 1_000_000);
    assert_eq!(total.output_tokens, 2_000_000);
    assert!(elapsed < Duration::from_millis(50));
}

// ---------------------------------------------------------------------------
// Stress 11: Mock agent 20 sequential runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_agent_20_sequential_runs_mock() {
    let mut agent = make_agent("stress-bot");
    let responses: Vec<mock_adapter::MockResponse> = (0..20)
        .map(|i| MockAdapter::text(format!("Response number {}", i)))
        .collect();
    let adapter = MockAdapter::arc(responses);

    let start = Instant::now();
    let mut success = 0;
    for i in 0..20 {
        let r = agent
            .run(&format!("Query {}", i), Arc::clone(&adapter))
            .await
            .unwrap();
        if r.success {
            success += 1;
        }
    }
    let elapsed = start.elapsed();
    println!(
        "[stress] 20 sequential mock agent runs in {:?}: {}/20 succeeded",
        elapsed, success
    );
    assert_eq!(success, 20);
    assert!(elapsed < Duration::from_secs(2));
}

// ---------------------------------------------------------------------------
// Stress 12: Pool burst — 100 jobs, pool(5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_pool_burst_100_jobs() {
    let pool = Arc::new(AgentPool::new(5));
    let counter = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..100)
        .map(|_| {
            let pool = Arc::clone(&pool);
            let counter = Arc::clone(&counter);
            let peak = Arc::clone(&peak);
            let active = Arc::clone(&active);
            tokio::spawn(async move {
                pool.run(|| async {
                    let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok::<(), open_multi_agent_rs::error::AgentError>(())
                })
                .await
                .unwrap();
            })
        })
        .collect();

    let start = Instant::now();
    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();

    let n = counter.load(Ordering::Relaxed);
    let p = peak.load(Ordering::SeqCst);
    println!(
        "[stress] 100 burst jobs pool(5) in {:?}: peak={}, completed={}",
        elapsed, p, n
    );
    assert_eq!(n, 100);
    assert!(p <= 5, "Peak {} exceeded limit 5", p);
    assert!(elapsed < Duration::from_secs(3));
}

// ===========================================================================
// NEW FEATURE STRESS TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Stress 13: Streaming — 30 concurrent agent streams
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_30_concurrent_streams() {
    use futures::StreamExt;
    use open_multi_agent_rs::types::StreamEvent;

    let n = 30;
    let completed = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..n)
        .map(|i| {
            let completed = Arc::clone(&completed);
            tokio::spawn(async move {
                let reg = Arc::new(Mutex::new(ToolRegistry::new()));
                let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
                let mut agent = Agent::new(
                    AgentConfig {
                        name: format!("bot-{}", i),
                        model: "mock".to_string(),
                        ..Default::default()
                    },
                    reg,
                    exec,
                );
                let adapter = MockAdapter::arc(vec![MockAdapter::text(format!("result-{}", i))]);
                let mut done_seen = false;
                let mut s = agent.stream(&format!("query-{}", i), adapter);
                tokio::pin!(s);
                while let Some(ev) = s.next().await {
                    if matches!(ev, StreamEvent::Done(_)) {
                        done_seen = true;
                    }
                }
                assert!(done_seen, "stream-{} never emitted Done", i);
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let n_done = completed.load(Ordering::Relaxed);
    println!(
        "[stress] {} concurrent streams completed in {:?}",
        n_done, elapsed
    );
    assert_eq!(n_done, n);
    assert!(elapsed < Duration::from_secs(3));
}

// ---------------------------------------------------------------------------
// Stress 14: Trace events — high-volume event collection (50 agents, each 3 turns)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_trace_event_collection_50_agents() {
    use open_multi_agent_rs::error::Result as AgentResult;
    use open_multi_agent_rs::tool::Tool;
    use open_multi_agent_rs::types::ToolUseContext;

    struct NopTool;
    #[async_trait::async_trait]
    impl Tool for NopTool {
        fn name(&self) -> &str {
            "nop"
        }
        fn description(&self) -> &str {
            "noop"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn execute(
            &self,
            _: &std::collections::HashMap<String, serde_json::Value>,
            _: &ToolUseContext,
        ) -> AgentResult<open_multi_agent_rs::types::ToolResult> {
            Ok(open_multi_agent_rs::types::ToolResult {
                data: "ok".to_string(),
                is_error: false,
            })
        }
    }

    let total_events = Arc::new(AtomicUsize::new(0));
    let on_trace: open_multi_agent_rs::types::OnTraceFn = {
        let counter = Arc::clone(&total_events);
        Arc::new(move |_ev: TraceEvent| {
            counter.fetch_add(1, Ordering::Relaxed);
        })
    };

    let start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..50 {
        let on_trace = Arc::clone(&on_trace);
        handles.push(tokio::spawn(async move {
            let reg = Arc::new(Mutex::new(ToolRegistry::new()));
            reg.lock().await.register(Arc::new(NopTool)).unwrap();
            let exec = Arc::new(ToolExecutor::new(Arc::clone(&reg)));
            let mut agent = Agent::new(
                AgentConfig {
                    name: format!("tracer-{}", i),
                    model: "mock".to_string(),
                    ..Default::default()
                },
                reg,
                exec,
            );

            // Each run: tool-call turn + final turn → 2 llm_call + 1 tool_call + 1 agent = 4 events
            let mut inp = std::collections::HashMap::new();
            inp.insert("x".to_string(), serde_json::json!(1));
            let adapter = MockAdapter::arc(vec![
                MockAdapter::tool_call("nop", "c1", inp),
                MockAdapter::text("done"),
            ]);

            let opts = RunOptions {
                on_trace: Some(on_trace),
                ..Default::default()
            };
            agent.run_with_opts("go", adapter, opts).await.unwrap();
        }));
    }

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let count = total_events.load(Ordering::Relaxed);

    // Each agent: 2 llm_call + 1 tool_call + 1 agent = 4 events
    println!(
        "[stress] 50 agents × 4 trace events = {} events collected in {:?}",
        count, elapsed
    );
    assert_eq!(count, 50 * 4, "Expected {} events, got {}", 50 * 4, count);
    assert!(elapsed < Duration::from_secs(5));
}

// ---------------------------------------------------------------------------
// Stress 15: MessageBus — 100 concurrent senders, 10 subscribers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_message_bus_100_senders_10_receivers() {
    let bus = Arc::new(MessageBus::new());
    let msg_received = Arc::new(AtomicUsize::new(0));

    // Set up 10 subscribers
    let mut unsubs = Vec::new();
    for i in 0..10 {
        let r = Arc::clone(&msg_received);
        let agent = format!("receiver-{}", i);
        let unsub = bus.subscribe(&agent, move |_msg| {
            r.fetch_add(1, Ordering::Relaxed);
        });
        unsubs.push(unsub);
    }

    let start = Instant::now();

    // 100 concurrent senders, each sending to all 10 receivers
    let mut handles = Vec::new();
    for sender_i in 0..100 {
        let bus = Arc::clone(&bus);
        handles.push(tokio::spawn(async move {
            for receiver_i in 0..10 {
                let agent = format!("receiver-{}", receiver_i);
                bus.send(&format!("sender-{}", sender_i), &agent, "hello");
            }
        }));
    }
    futures::future::join_all(handles).await;

    let elapsed = start.elapsed();
    let total_msgs = msg_received.load(Ordering::Relaxed);

    // 100 senders × 10 receivers = 1000 messages, each triggering 1 subscriber callback
    println!(
        "[stress] MessageBus: 100 senders × 10 receivers = {} callbacks in {:?}",
        total_msgs, elapsed
    );
    assert_eq!(
        total_msgs,
        100 * 10,
        "Expected {} callbacks, got {}",
        100 * 10,
        total_msgs
    );
    assert!(elapsed < Duration::from_secs(5));
}

// ---------------------------------------------------------------------------
// Stress 16: MessageBus broadcast — 50 broadcasts received by 20 subscribers each
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_message_bus_broadcasts_50x20() {
    let bus = Arc::new(MessageBus::new());
    let total_received = Arc::new(AtomicUsize::new(0));

    // 20 subscribers on different agents
    let mut _unsubs = Vec::new();
    for i in 0..20 {
        let r = Arc::clone(&total_received);
        let unsub = bus.subscribe(&format!("agent-{}", i), move |_| {
            r.fetch_add(1, Ordering::Relaxed);
        });
        _unsubs.push(unsub);
    }

    let start = Instant::now();

    // 50 broadcasts from different senders
    let mut handles = Vec::new();
    for sender_i in 0..50 {
        let bus = Arc::clone(&bus);
        handles.push(tokio::spawn(async move {
            bus.broadcast(&format!("sender-{}", sender_i), "broadcast message");
        }));
    }
    futures::future::join_all(handles).await;

    let elapsed = start.elapsed();
    let count = total_received.load(Ordering::Relaxed);

    // Each broadcast reaches all 20 subscribers
    println!(
        "[stress] 50 broadcasts × 20 subscribers = {} callbacks in {:?}",
        count, elapsed
    );
    assert_eq!(count, 50 * 20);
    assert!(elapsed < Duration::from_secs(5));
}

// ---------------------------------------------------------------------------
// Stress 17: Retry — 100 tasks, each with 2 retries, all succeed on 3rd attempt
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_retry_100_tasks_each_with_2_retries() {
    let total_calls = Arc::new(AtomicUsize::new(0));
    let successful = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..100 {
        let tc = Arc::clone(&total_calls);
        let sc = Arc::clone(&successful);
        handles.push(tokio::spawn(async move {
            let mut task = create_task("t".to_string(), "d".to_string(), None, vec![]);
            task.max_retries = Some(2);
            task.retry_delay_ms = Some(1); // tiny delay for fast tests
            task.retry_backoff = Some(1.0);

            let call_n = Arc::new(AtomicUsize::new(0));
            let call_n_clone = Arc::clone(&call_n);
            let tc_clone = Arc::clone(&tc);

            let result = execute_with_retry(
                move || {
                    let n = call_n_clone.fetch_add(1, Ordering::SeqCst);
                    tc_clone.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async move {
                        if n < 2 {
                            Err(open_multi_agent_rs::error::AgentError::Other(
                                "fail".to_string(),
                            ))
                        } else {
                            Ok(AgentRunResult {
                                success: true,
                                output: "ok".to_string(),
                                messages: vec![],
                                token_usage: TokenUsage::default(),
                                tool_calls: vec![],
                                turns: 1,
                                structured: None,
                            })
                        }
                    })
                },
                &task,
                None::<Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>>,
            )
            .await;

            if result.success {
                sc.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let calls = total_calls.load(Ordering::Relaxed);
    let success = successful.load(Ordering::Relaxed);

    // Each task: 3 attempts (fail, fail, succeed)
    println!(
        "[stress] 100 tasks × 3 attempts = {} calls, {} succeeded in {:?}",
        calls, success, elapsed
    );
    assert_eq!(
        calls, 300,
        "Expected 300 total calls (100 tasks × 3), got {}",
        calls
    );
    assert_eq!(success, 100, "All 100 tasks should succeed on 3rd attempt");
    assert!(elapsed < Duration::from_secs(5));
}

// ---------------------------------------------------------------------------
// Stress 18: compute_retry_delay — 10k computations stay under 1ms
// ---------------------------------------------------------------------------

#[test]
fn stress_compute_retry_delay_10k() {
    let start = Instant::now();
    let mut total: u64 = 0;
    for attempt in 1..=10_000u32 {
        let delay = compute_retry_delay(100, 2.0, (attempt % 20) + 1);
        total = total.wrapping_add(delay);
    }
    let elapsed = start.elapsed();
    println!(
        "[stress] 10k compute_retry_delay calls in {:?} (total={})",
        elapsed, total
    );
    assert!(
        elapsed < Duration::from_millis(10),
        "compute_retry_delay is too slow"
    );
}

// ---------------------------------------------------------------------------
// Stress 19: MessageBus concurrent subscribe/unsubscribe while sending
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stress_message_bus_concurrent_subscribe_unsubscribe() {
    let bus = Arc::new(MessageBus::new());
    let received = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = Vec::new();

    // 20 concurrent workers that subscribe, receive, then unsubscribe
    for i in 0..20 {
        let bus = Arc::clone(&bus);
        let r = Arc::clone(&received);
        handles.push(tokio::spawn(async move {
            let agent = format!("worker-{}", i);
            let r2 = Arc::clone(&r);
            let unsub = bus.subscribe(&agent, move |_| {
                r2.fetch_add(1, Ordering::Relaxed);
            });
            // Send some messages while subscribed
            for j in 0..5 {
                bus.send("external", &agent, &format!("msg-{}", j));
            }
            // Unsubscribe
            unsub();
            // Messages after unsubscription should NOT be received
            for j in 0..5 {
                bus.send("external", &agent, &format!("after-{}", j));
            }
        }));
    }

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let count = received.load(Ordering::Relaxed);

    println!(
        "[stress] concurrent subscribe/unsubscribe: {} callbacks in {:?}",
        count, elapsed
    );
    // Each of 20 workers receives exactly 5 messages while subscribed
    assert_eq!(
        count,
        20 * 5,
        "Expected {} callbacks, got {}",
        20 * 5,
        count
    );
    assert!(elapsed < Duration::from_secs(5));
}

// ---------------------------------------------------------------------------
// Stress 20: Skip remaining — 500 tasks, skip after first batch
// ---------------------------------------------------------------------------

#[test]
fn stress_skip_remaining_500_tasks() {
    let mut q = TaskQueue::new();
    let n = 500;

    let tasks: Vec<Task> = (0..n)
        .map(|i| {
            let mut t = create_task(&format!("t{}", i), "desc", None, vec![]);
            t.id = format!("t{}", i);
            t
        })
        .collect();
    q.add_batch(tasks);

    let start = Instant::now();
    q.skip_remaining("stress test");
    let elapsed = start.elapsed();

    let skipped = q
        .list()
        .iter()
        .filter(|t| t.status == TaskStatus::Skipped)
        .count();
    println!(
        "[stress] skip_remaining 500 tasks in {:?}: {} skipped",
        elapsed, skipped
    );
    assert_eq!(skipped, n);
    assert!(elapsed < Duration::from_millis(100));
}
