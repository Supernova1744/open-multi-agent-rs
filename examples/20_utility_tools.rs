/// Example 20 — Utility Tools
///
/// Demonstrates the seven general-purpose utility tools:
///
///   sleep    — pause execution (rate-limiting, polling)
///   random   — UUIDs, integers, floats, choices, strings
///   template — {{variable}} substitution in text
///   diff     — unified diff between two texts or files
///   zip      — create / list / extract ZIP archives
///   git      — safe read-heavy git operations
///   url      — parse / build / encode / decode / join URLs
///
/// Run:
///   cargo run --example 20_utility_tools
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry},
    types::AgentConfig,
};
use std::sync::Arc;
use tokio::sync::Mutex;

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set")
        .trim_matches('"')
        .to_string()
}

#[tokio::main]
async fn main() {
    // ── Shared infrastructure ─────────────────────────────────────────────────
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    let adapter: Arc<dyn open_multi_agent_rs::LLMAdapter> = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    // ── Agent config ──────────────────────────────────────────────────────────
    let config = AgentConfig {
        name: "utility-demo".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a demo agent for utility tools. \
             Complete each task precisely and report the result."
                .to_string(),
        ),
        max_turns: Some(20),
        tools: Some(vec![
            "sleep".to_string(),
            "random".to_string(),
            "template".to_string(),
            "diff".to_string(),
            "zip".to_string(),
            "git".to_string(),
            "url".to_string(),
            "file_write".to_string(),
            "file_list".to_string(),
        ]),
        ..Default::default()
    };

    let task = r#"
Please demonstrate each utility tool in order:

1. **sleep** — sleep for 200ms and confirm it worked.

2. **random** — generate:
   a. A UUID
   b. A random integer between 1 and 6 (dice roll)
   c. A random choice from ["rock", "paper", "scissors"]
   d. A 12-character alphanumeric string

3. **template** — render this template:
   "Dear {{name}}, your order #{{order_id}} totalling ${{amount}} has shipped."
   with vars: name="Alice", order_id="ORD-9821", amount="49.99"

4. **diff** — compute the unified diff between:
   Text A: "The quick brown fox\njumps over the lazy dog\nThe end"
   Text B: "The quick brown fox\nleaps over the lazy cat\nThe end\nFin"

5. **zip** — in the current directory:
   a. Write two files: notes.txt ("Hello zip!") and data.txt ("1,2,3")
   b. Create archive.zip containing both files
   c. List the contents of archive.zip
   d. Extract archive.zip into an "extracted/" subdirectory

6. **git** — run `git status` in the current directory.
   (It may not be a git repo — that's fine, just report the output.)

7. **url** — demonstrate:
   a. Parse "https://api.example.com/v1/search?q=rust+lang&page=2#results"
   b. Build a URL: scheme=https, host=openai.com, path=/v1/chat/completions,
      query={model: "gpt-4", stream: "true"}
   c. Encode the string "hello world & foo=bar"
   d. Join base="https://docs.rs/tokio/latest/tokio/index.html" with "../time/index.html"

Report each result clearly.
"#;

    let mut agent = Agent::new(config, registry, executor);
    match agent.run(task, adapter).await {
        Ok(result) => {
            println!("\n{}", "═".repeat(60));
            println!("UTILITY TOOLS DEMO COMPLETE");
            println!("{}", "═".repeat(60));
            println!("\n{}", result.output);
            println!(
                "\nFinished in {} turns with {} tool calls.",
                result.turns,
                result.tool_calls.len()
            );
            for tc in &result.tool_calls {
                println!("  • {} → {}ms", tc.tool_name, tc.duration_ms);
            }
        }
        Err(e) => eprintln!("Agent error: {}", e),
    }
}
