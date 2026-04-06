/// Example 16 — Web Research & Search Agent
///
/// Demonstrates the web research tools:
///   • web_fetch      — fetch a URL and return clean Markdown (HTML stripped)
///   • tavily_search  — real-time web search via the Tavily API
///   • schema_validate— validate tool output against a JSON schema
///
/// The agent:
///   1. Uses web_fetch to retrieve a public documentation page as Markdown
///   2. Optionally uses tavily_search for real-time information
///      (requires TAVILY_API_KEY — skipped gracefully if not set)
///   3. Uses schema_validate to force structured output
///
/// Run:
///   cargo run --example 16_web_search
///
/// For Tavily search:
///   TAVILY_API_KEY=tvly-... cargo run --example 16_web_search
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry},
    types::AgentConfig,
    AgentRunResult,
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
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    let has_tavily = std::env::var("TAVILY_API_KEY").is_ok();

    let config = AgentConfig {
        name: "web-researcher".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a web research agent. Use web_fetch to retrieve web pages as clean Markdown. \
             Use tavily_search for real-time web search when available. \
             Use schema_validate to output structured summaries. Be concise."
            .to_string(),
        ),
        max_turns: Some(8),
        tools: Some({
            let mut tools = vec![
                "web_fetch".to_string(),
                "json_parse".to_string(),
                "schema_validate".to_string(),
            ];
            if has_tavily {
                tools.push("tavily_search".to_string());
            }
            tools
        }),
        ..Default::default()
    };

    let mut agent = Agent::new(config, Arc::clone(&registry), Arc::clone(&executor));

    let task = if has_tavily {
        "1. Use tavily_search to find the latest information about 'Rust programming language 2024'.\n\
         2. Use schema_validate to extract a structured summary with schema:\n\
            {\"type\":\"object\",\"required\":[\"topic\",\"key_points\"],\"properties\":\
            {\"topic\":{\"type\":\"string\"},\"key_points\":{\"type\":\"array\"}}}\n\
         Report the validated result."
            .to_string()
    } else {
        "1. Use web_fetch to retrieve https://httpbin.org/html (a sample HTML page).\n\
         2. Use schema_validate to produce a structured summary of the page with schema:\n\
            {\"type\":\"object\",\"required\":[\"title\",\"summary\"],\"properties\":\
            {\"title\":{\"type\":\"string\"},\"summary\":{\"type\":\"string\"}}}\n\
         Report the validated result."
            .to_string()
    };

    println!("Task:\n{}\n{}", task, "─".repeat(60));
    if !has_tavily {
        println!("[TAVILY_API_KEY not set — using web_fetch instead of tavily_search]");
    }

    match agent.run(&task, adapter).await {
        Ok(AgentRunResult { output, turns, token_usage, tool_calls, .. }) => {
            println!("{}", "─".repeat(60));
            println!("Done in {} turns", turns);
            println!("Tokens — input: {}, output: {}", token_usage.input_tokens, token_usage.output_tokens);
            println!("Tool calls:");
            for tc in &tool_calls {
                println!("  • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
            println!("\nAgent response:\n{}", output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
