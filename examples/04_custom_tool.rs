/// Example 04 — Custom Tool
///
/// Defines a custom tool (`word_count`) that the agent can call during its
/// reasoning loop. Demonstrates the Tool trait and ToolRegistry.
///
/// The agent is given a passage of text and asked to count the words. It
/// will naturally call the word_count tool and then report back.
///
/// Run:
///   cargo run --example 04_custom_tool
use async_trait::async_trait;
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    error::Result,
    tool::Tool,
    types::{LLMToolDef, ToolResult, ToolUseContext},
    AgentConfig, ToolExecutor, ToolRegistry,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

// ---------------------------------------------------------------------------
// Custom tool definition
// ---------------------------------------------------------------------------

struct WordCountTool;

#[async_trait]
impl Tool for WordCountTool {
    fn name(&self) -> &str {
        "word_count"
    }

    fn description(&self) -> &str {
        "Count the number of words in a given text string."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in."
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(
        &self,
        input: &HashMap<String, serde_json::Value>,
        _context: &ToolUseContext,
    ) -> Result<ToolResult> {
        let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");

        let count = text.split_whitespace().count();
        println!("  [tool: word_count called → {} words]", count);

        Ok(ToolResult {
            data: count.to_string(),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    registry
        .lock()
        .await
        .register(Arc::new(WordCountTool))
        .expect("failed to register tool");

    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    let config = AgentConfig {
        name: "analyst".to_string(),
        model: "mistralai/mistral-nemo".to_string(),
        system_prompt: Some(
            "You are a text analyst. Use the word_count tool whenever you need to count words."
                .to_string(),
        ),
        max_turns: Some(5),
        ..Default::default()
    };

    let mut agent = Agent::new(config, registry, executor);

    let passage = "The quick brown fox jumps over the lazy dog. \
                   Rust is a systems programming language that runs blazingly fast.";

    println!("Passage: \"{}\"\n", passage);
    println!("Asking the agent to count words using the tool...\n");

    match agent
        .run(
            &format!("How many words are in this text: \"{}\"", passage),
            Arc::clone(&adapter),
        )
        .await
    {
        Ok(result) => {
            println!("\nAgent answer: {}", result.output);
            println!("Tool calls made: {}", result.tool_calls.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
