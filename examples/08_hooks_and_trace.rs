/// Example 08 — Lifecycle Hooks and Trace Events
///
/// Shows how to use `before_run` and `after_run` hooks on an agent config,
/// and how to attach a trace callback to observe every LLM call and agent run.
///
/// The before_run hook adds a word limit to every prompt.
/// The after_run hook uppercases the output for demonstration.
///
/// Run:
///   cargo run --example 08_hooks_and_trace
use futures::future::BoxFuture;
use open_multi_agent::{
    error::Result,
    types::{AgentConfig, AgentRunResult, BeforeRunHookContext, TraceEvent},
    AgentConfig as _, OrchestratorConfig, OpenMultiAgent,
};

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

#[tokio::main]
async fn main() {
    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "qwen/qwen3.6-plus:free".to_string(),
        default_provider: "openrouter".to_string(),
        default_base_url: Some("https://openrouter.ai/api/v1".to_string()),
        default_api_key: Some(api_key()),
        max_concurrency: 1,
        on_progress: None,
        on_trace: Some(std::sync::Arc::new(|event: TraceEvent| {
            // Trace callback: fired for every LLM call and agent run.
            match event {
                TraceEvent::LlmCall(t) => println!(
                    "[trace] LLM call  turn={} tokens={}in+{}out  {}ms",
                    t.turn,
                    t.tokens.input_tokens,
                    t.tokens.output_tokens,
                    t.base.duration_ms
                ),
                TraceEvent::Agent(t) => println!(
                    "[trace] Agent run turns={} total_tokens={}",
                    t.turns,
                    t.tokens.input_tokens + t.tokens.output_tokens
                ),
                _ => {}
            }
        })),
        on_approval: None,
    });

    let agent = AgentConfig {
        name: "hooked-agent".to_string(),
        system_prompt: Some("You are a helpful assistant.".to_string()),

        // before_run: append a word limit instruction to every prompt.
        before_run: Some(std::sync::Arc::new(
            |mut ctx: BeforeRunHookContext| -> BoxFuture<'static, Result<BeforeRunHookContext>> {
                Box::pin(async move {
                    println!("[before_run] original prompt: \"{}\"", ctx.prompt);
                    ctx.prompt = format!("{} (Answer in 10 words or fewer.)", ctx.prompt);
                    println!("[before_run] modified prompt: \"{}\"", ctx.prompt);
                    Ok(ctx)
                })
            },
        )),

        // after_run: uppercase the output.
        after_run: Some(std::sync::Arc::new(
            |mut result: AgentRunResult| -> BoxFuture<'static, Result<AgentRunResult>> {
                Box::pin(async move {
                    println!("[after_run]  raw output:      \"{}\"", result.output);
                    result.output = result.output.to_uppercase();
                    println!("[after_run]  modified output: \"{}\"", result.output);
                    Ok(result)
                })
            },
        )),

        ..Default::default()
    };

    println!("Running agent with before_run + after_run hooks...\n");

    match orchestrator
        .run_agent(agent, "What is the capital of Japan?")
        .await
    {
        Ok(result) => {
            println!("\nFinal output: {}", result.output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
