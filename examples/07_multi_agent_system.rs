/// Example 07 — Full Multi-Agent System
///
/// A complete multi-agent system for software planning and code review:
///
///   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
///   │  Architect  │────▶│  Developer  │────▶│  Reviewer   │
///   └─────────────┘     └─────────────┘     └─────────────┘
///         │                                        │
///         └────────── MessageBus ─────────────────▶│
///
/// Features demonstrated:
///   • Multi-agent team with shared memory
///   • Explicit dependency graph across 3 agents
///   • MessageBus for direct agent-to-agent communication
///   • Trace events (every LLM call and tool call logged)
///   • Approval gate between pipeline stages
///   • Retry with exponential backoff on the review task
///   • RunOptions callbacks for real-time tool observability
///
/// Run:
///   cargo run --example 07_multi_agent_system
use futures::future::BoxFuture;
use open_multi_agent_rs::{
    create_task,
    messaging::MessageBus,
    types::{AgentConfig, Task, TraceEvent},
    AgentStatus, OpenMultiAgent, OrchestratorConfig, TeamConfig,
};
use std::sync::{Arc, Mutex};

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

fn agent(name: &str, role: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "qwen/qwen3.6-plus:free".to_string(),
        provider: Some("openrouter".to_string()),
        base_url: Some("https://openrouter.ai/api/v1".to_string()),
        api_key: Some(api_key()),
        system_prompt: Some(role.to_string()),
        max_turns: Some(3),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Trace printer — logs every span to stdout
// ---------------------------------------------------------------------------

fn trace_printer() -> Arc<dyn Fn(TraceEvent) + Send + Sync> {
    Arc::new(|event: TraceEvent| match event {
        TraceEvent::LlmCall(t) => {
            println!(
                "  [trace] llm_call  agent={:<12} turn={} tokens={}+{}  {}ms",
                t.base.agent,
                t.turn,
                t.tokens.input_tokens,
                t.tokens.output_tokens,
                t.base.duration_ms
            );
        }
        TraceEvent::ToolCall(t) => {
            println!(
                "  [trace] tool_call agent={:<12} tool={} error={}",
                t.base.agent, t.tool, t.is_error
            );
        }
        TraceEvent::Task(t) => {
            println!(
                "  [trace] task      title={:<30} success={} retries={}  {}ms",
                t.task_title, t.success, t.retries, t.base.duration_ms
            );
        }
        TraceEvent::Agent(t) => {
            println!(
                "  [trace] agent     name={:<12} turns={} tool_calls={} tokens={}",
                t.base.agent,
                t.turns,
                t.tool_calls,
                t.tokens.input_tokens + t.tokens.output_tokens
            );
        }
    })
}

// ---------------------------------------------------------------------------
// Approval gate — asks (auto-approves here; change to false to test rejection)
// ---------------------------------------------------------------------------

fn approval_gate() -> Arc<dyn Fn(Vec<Task>, Vec<Task>) -> BoxFuture<'static, bool> + Send + Sync> {
    Arc::new(
        |completed: Vec<Task>, pending: Vec<Task>| -> BoxFuture<'static, bool> {
            Box::pin(async move {
                println!(
                    "\n[approval gate] {} task(s) done. {} task(s) pending.",
                    completed.len(),
                    pending.len()
                );
                for t in &pending {
                    println!("  → pending: {}", t.title);
                }
                println!("[approval gate] Auto-approving.\n");
                true // return false here to abort the remaining tasks
            })
        },
    )
}

// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // ── MessageBus ────────────────────────────────────────────────────────────
    // Agents can send messages to each other via the bus.
    // We subscribe before tasks run so we can log inter-agent messages.
    let bus = MessageBus::new();
    let bus_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let log_clone = Arc::clone(&bus_log);
    let _unsub = bus.subscribe("reviewer", move |msg| {
        log_clone
            .lock()
            .unwrap()
            .push(format!("[{}→reviewer] {}", msg.from, msg.content));
    });

    // ── Orchestrator ──────────────────────────────────────────────────────────
    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "qwen/qwen3.6-plus:free".to_string(),
        default_provider: "openrouter".to_string(),
        default_base_url: Some("https://openrouter.ai/api/v1".to_string()),
        default_api_key: Some(api_key()),
        max_concurrency: 2,
        on_progress: None,
        on_trace: Some(trace_printer()),
        on_approval: Some(approval_gate()),
    });

    // ── Team ──────────────────────────────────────────────────────────────────
    let team = TeamConfig {
        name: "software-team".to_string(),
        agents: vec![
            agent(
                "architect",
                "You are a senior software architect. \
                 Design clear, minimal API surfaces with concise explanations. \
                 Output numbered bullet points.",
            ),
            agent(
                "developer",
                "You are an expert Rust developer. \
                 Implement what the architect designs using idiomatic Rust. \
                 Write actual code snippets, not pseudocode.",
            ),
            agent(
                "reviewer",
                "You are a code reviewer focused on correctness, safety, and performance. \
                 Point out specific issues and suggest concrete improvements. \
                 Be concise.",
            ),
        ],
        shared_memory: Some(true),
        max_concurrency: Some(2),
    };

    // ── Task graph ────────────────────────────────────────────────────────────
    //
    //   task_design ──┐
    //                 ├──▶ task_implement ──▶ task_review
    //   (no deps)    ─┘

    let task_design = create_task(
        "Design a rate-limiter API",
        "Design a simple token-bucket rate limiter API in Rust. \
         Specify: the public struct, its fields, and 3 key methods with signatures. \
         Be concise — no implementation, just the design.",
        Some("architect".to_string()),
        vec![],
    );
    let design_id = task_design.id.clone();

    let task_implement = create_task(
        "Implement the rate limiter",
        "Based on the architect's design in shared memory, implement the rate limiter in Rust. \
         Write complete, compiling code with doc-comments. Keep it under 60 lines.",
        Some("developer".to_string()),
        vec![design_id],
    );
    let implement_id = task_implement.id.clone();

    let task_review = {
        let mut t = create_task(
            "Review the implementation",
            "Review the Rust rate limiter implementation from shared memory. \
             Check for: correctness, thread safety, edge cases, and idiomatic Rust. \
             List at most 3 issues with suggested fixes.",
            Some("reviewer".to_string()),
            vec![implement_id],
        );
        // Reviewer task: retry up to 2 times if it fails (e.g. empty output).
        t.max_retries = Some(2);
        t.retry_delay_ms = Some(500);
        t.retry_backoff = Some(2.0);
        t
    };

    // ── Send a message from "system" to "reviewer" via the bus ───────────────
    // In a real system agents would send these; here we demonstrate the bus API.
    bus.send("system", "reviewer", "Focus especially on thread safety.");

    // ── Run ───────────────────────────────────────────────────────────────────
    println!("Multi-Agent Software Team");
    println!("{}", "=".repeat(60));
    println!("Goal: Design, implement, and review a Rust rate limiter.\n");

    match orchestrator
        .run_tasks(&team, vec![task_design, task_implement, task_review])
        .await
    {
        Ok(result) => {
            println!("\n{}", "=".repeat(60));
            println!("Pipeline complete — success: {}", result.success);
            println!(
                "Total tokens — input: {}, output: {}\n",
                result.total_token_usage.input_tokens, result.total_token_usage.output_tokens
            );

            // Print each agent's output with a header.
            let labels = [
                ("architect", "DESIGN"),
                ("developer", "IMPLEMENTATION"),
                ("reviewer", "REVIEW"),
            ];
            for (agent_name, label) in labels {
                // Find the result that belongs to this agent.
                if let Some(r) = result.agent_results.values().find(|r| r.messages.len() > 0) {
                    // Fallback: just print in order.
                    let _ = (agent_name, label, r);
                }
            }
            // Simpler: print in insertion order (HashMap ordering is not guaranteed,
            // so we just print all results clearly).
            for (task_id, r) in &result.agent_results {
                println!("── Task {} ──────────────────────────", &task_id[..8]);
                println!("{}\n", r.output);
            }

            // Show any inter-agent bus messages that were received.
            let messages = bus_log.lock().unwrap();
            if !messages.is_empty() {
                println!("── MessageBus log ───────────────────");
                for msg in messages.iter() {
                    println!("  {}", msg);
                }
            }
        }
        Err(e) => eprintln!("System error: {}", e),
    }
}
