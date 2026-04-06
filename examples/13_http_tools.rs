/// Example 13 — HTTP Tools Agent
///
/// Demonstrates the built-in HTTP tools:
///   • http_get  — fetch any URL and return the response body + status
///   • http_post — send a POST request with a JSON body
///
/// Also exercises the data tools in the same workflow:
///   • json_parse     — validate and extract fields from JSON responses
///   • json_transform — reshape JSON (keys, map, pointer extraction)
///
/// The agent is asked to:
///   1. Fetch a public JSON API endpoint (httpbin.org)
///   2. Extract specific fields from the response using json_parse
///   3. Post a small JSON payload and inspect the echo response
///
/// Run:
///   cargo run --example 13_http_tools
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
    // ── Tool registry ─────────────────────────────────────────────────────────
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

    // ── Adapter ───────────────────────────────────────────────────────────────
    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    // ── Agent ─────────────────────────────────────────────────────────────────
    let config = AgentConfig {
        name: "http-agent".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are an agent that can make HTTP requests and process JSON responses. \
             Use the available tools to fetch data, parse JSON, and report what you find. \
             Be concise in your final summary."
            .to_string(),
        ),
        max_turns: Some(8),
        tools: Some(vec![
            "http_get".to_string(),
            "http_post".to_string(),
            "json_parse".to_string(),
            "json_transform".to_string(),
        ]),
        ..Default::default()
    };

    let mut agent = Agent::new(config, Arc::clone(&registry), Arc::clone(&executor));

    // ── Task ──────────────────────────────────────────────────────────────────
    let task = "\
        Complete the following HTTP tasks and report the results:\n\
        \n\
        1. Fetch https://httpbin.org/json using http_get\n\
        2. Use json_parse to extract the value at JSON pointer /slideshow/title\n\
        3. Use json_transform with operation 'keys' on the top-level JSON object \
           to list its keys\n\
        4. POST to https://httpbin.org/post with body '{\"agent\":\"open-multi-agent-rs\",\"test\":true}' \
           using http_post, then use json_parse to extract /json/agent from the response\n\
        \n\
        Summarise all results clearly.";

    println!("Task:\n{}\n{}", task, "─".repeat(60));

    match agent.run(task, adapter).await {
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
