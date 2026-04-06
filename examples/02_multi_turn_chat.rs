/// Example 02 — Multi-Turn Chat
///
/// Demonstrates stateful conversation using `agent.prompt()`.
/// Each call appends to the conversation history so the agent remembers
/// what was said in previous turns.
///
/// Run:
///   cargo run --example 02_multi_turn_chat
use open_multi_agent_rs::{agent::Agent, create_adapter, AgentConfig, ToolExecutor, ToolRegistry};
use std::sync::Arc;
use tokio::sync::Mutex;

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

#[tokio::main]
async fn main() {
    // Build a shared adapter and tool infrastructure.
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    let config = AgentConfig {
        name: "tutor".to_string(),
        model: "qwen/qwen3.6-plus:free".to_string(),
        system_prompt: Some("You are a patient math tutor. Give short, clear answers.".to_string()),
        ..Default::default()
    };

    let mut agent = Agent::new(config, registry, executor);

    // Each prompt() call retains history — the agent remembers all previous turns.
    let turns = [
        "What is 12 multiplied by 8?",
        "Now divide that result by 4.",
        "What percentage of 100 is that final number?",
    ];

    for (i, question) in turns.iter().enumerate() {
        println!("Turn {}: {}", i + 1, question);
        match agent.prompt(question, Arc::clone(&adapter)).await {
            Ok(r) => println!("  → {}\n", r.output),
            Err(e) => eprintln!("  → Error: {}\n", e),
        }
    }

    // Show accumulated token usage.
    println!(
        "Total tokens used — input: {}, output: {}",
        agent.total_token_usage.input_tokens, agent.total_token_usage.output_tokens
    );
}
