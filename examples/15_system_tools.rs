/// Example 15 — System, Encoding & Cache Tools Agent
///
/// Demonstrates the built-in system and utility tools:
///   • system_info  — return OS, arch, CPU count, and working directory
///   • env_get      — safely read an environment variable
///   • base64       — encode and decode Base64 strings
///   • hash_file    — compute the FNV-1a hash of a file
///   • cache_set    — store a value in the in-process cache with optional TTL
///   • cache_get    — retrieve a cached value
///
/// The agent:
///   1. Inspects the current system environment
///   2. Reads a safe environment variable
///   3. Encodes and decodes a message with Base64
///   4. Writes a small file then hashes it
///   5. Caches a computed result and retrieves it
///
/// Run:
///   cargo run --example 15_system_tools
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
    let sandbox = std::env::temp_dir().join("open_multi_agent_system_demo");
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
        name: "system-agent".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a system utility agent. Use the available tools to inspect \
             the system, work with encodings, manage a key-value cache, and verify \
             file integrity. Report your findings clearly and concisely."
            .to_string(),
        ),
        max_turns: Some(12),
        tools: Some(vec![
            "system_info".to_string(),
            "env_get".to_string(),
            "base64".to_string(),
            "hash_file".to_string(),
            "cache_set".to_string(),
            "cache_get".to_string(),
            "file_write".to_string(),
            "file_read".to_string(),
        ]),
        ..Default::default()
    };

    let mut agent = Agent::new(config, Arc::clone(&registry), Arc::clone(&executor));

    // ── Task ──────────────────────────────────────────────────────────────────
    let task = "\
        Complete the following system utility tasks:\n\
        \n\
        1. SYSTEM INFO: Call system_info to get OS, architecture, CPU count, \
           and current working directory.\n\
        \n\
        2. ENV: Use env_get to read the 'HOME' (or 'USERNAME' on Windows) \
           environment variable.\n\
        \n\
        3. BASE64 ENCODE: Use base64 with mode='encode' to encode the string \
           'Hello, open-multi-agent-rs!'.\n\
        \n\
        4. BASE64 DECODE: Decode the result back with mode='decode' and verify \
           it matches the original.\n\
        \n\
        5. HASH: Write a file 'checksum_test.txt' with content \
           'The quick brown fox jumps over the lazy dog', then call hash_file \
           on it to get its hash.\n\
        \n\
        6. CACHE: Use cache_set to store key='last_hash' with the hash value \
           from step 5 (ttl_seconds=300). Then immediately retrieve it with \
           cache_get to confirm it was stored.\n\
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
