/// Example 01 — Single Agent
///
/// The simplest possible usage: one agent, one question, one answer.
/// Shows how to construct an orchestrator and run a single agent.
///
/// Run:
///   cargo run --example 01_single_agent
use open_multi_agent::{AgentConfig, OrchestratorConfig, OpenMultiAgent};

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
        on_trace: None,
        on_approval: None,
    });

    let agent = AgentConfig {
        name: "assistant".to_string(),
        system_prompt: Some("You are a concise, factual assistant. Keep answers under 3 sentences.".to_string()),
        ..Default::default()
    };

    println!("Asking: What is Rust's ownership model in one sentence?\n");

    match orchestrator
        .run_agent(agent, "What is Rust's ownership model in one sentence?")
        .await
    {
        Ok(result) => {
            println!("Answer: {}", result.output);
            println!(
                "\nTokens — input: {}, output: {}",
                result.token_usage.input_tokens, result.token_usage.output_tokens
            );
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
