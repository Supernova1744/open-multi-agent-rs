/// Example 10 — Retry with Backoff and Approval Gate
///
/// Demonstrates:
///   • Per-task retry configuration (max_retries, retry_delay_ms, retry_backoff)
///   • `execute_with_retry` used directly for custom retry logic
///   • An approval gate that inspects pending tasks and can abort the pipeline
///
/// The approval gate in this example approves the first round but rejects
/// the second, so only the first batch of tasks completes.
///
/// Run:
///   cargo run --example 10_retry_and_approval
use futures::future::BoxFuture;
use open_multi_agent::{
    compute_retry_delay, create_task,
    types::{AgentRunResult, Task, TokenUsage},
    AgentConfig, OpenMultiAgent, OrchestratorConfig, TeamConfig,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

fn agent(name: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "qwen/qwen3.6-plus:free".to_string(),
        provider: Some("openrouter".to_string()),
        base_url: Some("https://openrouter.ai/api/v1".to_string()),
        api_key: Some(api_key()),
        system_prompt: Some("You are a concise assistant. Answer in one sentence.".to_string()),
        max_turns: Some(2),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() {
    // ── Part 1: show compute_retry_delay math ────────────────────────────────
    println!("Retry delay schedule (base=500ms, backoff=2.0):");
    for attempt in 1..=6u32 {
        let delay = compute_retry_delay(500, 2.0, attempt);
        println!("  attempt {} → {}ms", attempt, delay);
    }
    println!();

    // ── Part 2: execute_with_retry directly ─────────────────────────────────
    println!("Simulating a flaky task that fails twice then succeeds...");

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let mut task = create_task("flaky-task", "some description", None, vec![]);
    task.max_retries = Some(3);
    task.retry_delay_ms = Some(50); // short for demo
    task.retry_backoff = Some(2.0);

    let result = open_multi_agent::execute_with_retry(
        move || {
            let cc = Arc::clone(&cc);
            Box::pin(async move {
                let n = cc.fetch_add(1, Ordering::SeqCst);
                println!("  attempt {} ...", n + 1);
                if n < 2 {
                    // Fail first two attempts.
                    Ok(AgentRunResult {
                        success: false,
                        output: format!("Attempt {} failed.", n + 1),
                        messages: vec![],
                        token_usage: TokenUsage {
                            input_tokens: 10,
                            output_tokens: 5,
                        },
                        tool_calls: vec![],
                        turns: 1,
                        structured: None,
                    })
                } else {
                    Ok(AgentRunResult {
                        success: true,
                        output: "Finally succeeded!".to_string(),
                        messages: vec![],
                        token_usage: TokenUsage {
                            input_tokens: 10,
                            output_tokens: 5,
                        },
                        tool_calls: vec![],
                        turns: 1,
                        structured: None,
                    })
                }
            })
        },
        &task,
        Some(Arc::new(|attempt: u32, max: u32, err: String, delay: u64| {
            println!(
                "  [retry {}/{}] \"{}\" — waiting {}ms",
                attempt, max, err, delay
            );
        })
            as Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>),
    )
    .await;

    println!(
        "Result: success={} output=\"{}\" total_tokens={}+{}\n",
        result.success,
        result.output,
        result.token_usage.input_tokens,
        result.token_usage.output_tokens
    );

    // ── Part 3: approval gate in an orchestrator pipeline ───────────────────
    println!("Running a two-stage pipeline with an approval gate...\n");

    let round = Arc::new(AtomicUsize::new(0));

    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "qwen/qwen3.6-plus:free".to_string(),
        default_provider: "openrouter".to_string(),
        default_base_url: Some("https://openrouter.ai/api/v1".to_string()),
        default_api_key: Some(api_key()),
        max_concurrency: 2,
        on_progress: None,
        on_trace: None,
        on_approval: Some(Arc::new(
            move |completed: Vec<Task>, pending: Vec<Task>| -> BoxFuture<'static, bool> {
                let round_n = round.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    println!(
                        "[approval gate] round={} completed={} pending={}",
                        round_n,
                        completed.len(),
                        pending.len()
                    );
                    // Approve round 0, reject round 1+.
                    let approved = round_n == 0;
                    println!(
                        "[approval gate] decision: {}",
                        if approved { "APPROVE" } else { "REJECT" }
                    );
                    approved
                })
            },
        )),
    });

    let team = TeamConfig {
        name: "demo-team".to_string(),
        agents: vec![agent("worker-a"), agent("worker-b")],
        shared_memory: Some(false),
        max_concurrency: Some(2),
    };

    let t1 = create_task(
        "Stage 1",
        "Say: 'Stage 1 complete.'",
        Some("worker-a".to_string()),
        vec![],
    );
    let t1_id = t1.id.clone();

    let t2 = create_task(
        "Stage 2",
        "Say: 'Stage 2 complete.'",
        Some("worker-b".to_string()),
        vec![t1_id],
    );

    match orchestrator.run_tasks(&team, vec![t1, t2]).await {
        Ok(result) => {
            println!("\nPipeline done. success={}", result.success);
            for (id, r) in &result.agent_results {
                println!(
                    "  task {}: {} (success={})",
                    &id[..8],
                    r.output.trim(),
                    r.success
                );
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
