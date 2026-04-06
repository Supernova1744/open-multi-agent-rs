/// Example 14 — Data Processing Tools Agent
///
/// Demonstrates the built-in data processing and math tools:
///   • csv_read       — read a CSV file and return rows as JSON
///   • csv_write      — write a JSON array to a CSV file
///   • json_parse     — parse and extract from JSON strings
///   • json_transform — reshape JSON (keys, values, length, map, pointer)
///   • math_eval      — evaluate mathematical expressions with variables
///   • datetime       — get current time, format timestamps, compute diffs
///   • text_regex     — find matches, replace, or split text
///   • text_chunk     — split large text into processing chunks
///
/// The agent:
///   1. Creates a CSV file with sample sales data
///   2. Reads it back and extracts stats using math_eval
///   3. Demonstrates regex extraction on the data
///   4. Shows datetime operations
///   5. Chunks a larger text block
///
/// Run:
///   cargo run --example 14_data_tools
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
    // ── Sandbox ───────────────────────────────────────────────────────────────
    let sandbox = std::env::temp_dir().join("open_multi_agent_data_demo");
    tokio::fs::create_dir_all(&sandbox).await.unwrap();
    std::env::set_current_dir(&sandbox).unwrap();
    println!("Sandbox: {}\n", sandbox.display());

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
        name: "data-agent".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a data processing agent. Use the available tools to create, \
             read, and analyse data files, evaluate math expressions, work with dates, \
             and process text. Be systematic and show your work."
            .to_string(),
        ),
        max_turns: Some(14),
        tools: Some(vec![
            "csv_write".to_string(),
            "csv_read".to_string(),
            "json_parse".to_string(),
            "json_transform".to_string(),
            "math_eval".to_string(),
            "datetime".to_string(),
            "text_regex".to_string(),
            "text_chunk".to_string(),
            "file_write".to_string(),
            "file_read".to_string(),
        ]),
        ..Default::default()
    };

    let mut agent = Agent::new(config, Arc::clone(&registry), Arc::clone(&executor));

    // ── Task ──────────────────────────────────────────────────────────────────
    let task = "\
        Complete the following data processing tasks:\n\
        \n\
        1. CSV WRITE: Use csv_write to create 'sales.csv' with this data JSON:\n\
           '[{\"product\":\"Widget A\",\"units\":\"120\",\"price\":\"9.99\"},\
             {\"product\":\"Gadget B\",\"units\":\"45\",\"price\":\"24.50\"},\
             {\"product\":\"Doohickey C\",\"units\":\"200\",\"price\":\"4.75\"}]'\n\
        \n\
        2. CSV READ: Read 'sales.csv' back as JSON, then use json_transform \
           with operation '[/units]' to extract all unit counts.\n\
        \n\
        3. MATH: Use math_eval to calculate:\n\
           a) Total revenue for Widget A: 120 * 9.99\n\
           b) Total revenue for Gadget B: 45 * 24.50\n\
           c) Average price: (9.99 + 24.50 + 4.75) / 3\n\
        \n\
        4. DATETIME: Use the datetime tool to:\n\
           a) Get the current UTC time (operation='now')\n\
           b) Compute the diff between timestamps 1700000000 and 1710000000\n\
        \n\
        5. REGEX: Use text_regex on the string \
           'Order #1001 placed on 2024-01-15, Order #1002 on 2024-02-20' \
           with pattern '\\\\d{4}-\\\\d{2}-\\\\d{2}' and mode 'find_all' \
           to extract all dates.\n\
        \n\
        Report all results clearly.";

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

    // ── Cleanup ───────────────────────────────────────────────────────────────
    let _ = tokio::fs::remove_dir_all(&sandbox).await;
    println!("\nSandbox cleaned up.");
}
