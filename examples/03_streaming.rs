/// Example 03 — Streaming Output
///
/// Uses `agent.stream()` to receive tokens as they are generated, printing
/// each text chunk immediately instead of waiting for the full response.
/// Also shows how to read the final `Done` event for usage stats.
///
/// Run:
///   cargo run --example 03_streaming
use futures::StreamExt;
use open_multi_agent::{
    agent::Agent, create_adapter, types::StreamEvent, AgentConfig, ToolExecutor, ToolRegistry,
};
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
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    let config = AgentConfig {
        name: "storyteller".to_string(),
        model: "qwen/qwen3.6-plus:free".to_string(),
        system_prompt: Some("You are a creative writer. Write engaging short stories.".to_string()),
        max_turns: Some(1),
        ..Default::default()
    };

    let mut agent = Agent::new(config, registry, executor);

    println!("Streaming a short story about a Rust programmer...\n");
    println!("{}", "-".repeat(60));

    let stream = agent.stream(
        "Write a 4-sentence story about a programmer who discovers Rust.",
        Arc::clone(&adapter),
    );
    tokio::pin!(stream);

    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Text(text) => {
                // Print each chunk immediately, no newline — they stream in.
                print!("{}", text);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            StreamEvent::Done(result) => {
                println!("\n{}", "-".repeat(60));
                println!(
                    "\nDone. Turns: {}, Tokens: in={} out={}",
                    result.turns,
                    result.token_usage.input_tokens,
                    result.token_usage.output_tokens
                );
            }
            StreamEvent::Error(msg) => {
                eprintln!("\nStream error: {}", msg);
                break;
            }
            _ => {} // ToolUse / ToolResult not expected without tools registered
        }
    }
}
