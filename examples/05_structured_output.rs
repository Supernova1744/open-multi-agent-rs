/// Example 05 — Structured Output
///
/// Demonstrates the `output_schema` field on AgentConfig.
/// The agent is asked to extract information from a sentence and return it
/// as a validated JSON object. If the first response doesn't match the schema
/// the framework retries automatically with error feedback.
///
/// Run:
///   cargo run --example 05_structured_output
use open_multi_agent_rs::{AgentConfig, OpenMultiAgent, OrchestratorConfig};

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

    // Define the JSON schema the response must conform to.
    let schema = serde_json::json!({
        "type": "object",
        "required": ["name", "age", "city"],
        "properties": {
            "name": { "type": "string" },
            "age":  { "type": "number" },
            "city": { "type": "string" }
        }
    });

    let agent = AgentConfig {
        name: "extractor".to_string(),
        system_prompt: Some(
            "You are a data extractor. Always respond with only a JSON object.".to_string(),
        ),
        output_schema: Some(schema),
        ..Default::default()
    };

    let sentences = [
        "Alice is 28 years old and lives in Berlin.",
        "Bob, who is 35, moved to Tokyo last year.",
        "Carol (age 22) is based in São Paulo.",
    ];

    for sentence in sentences {
        println!("Input: \"{}\"", sentence);

        match orchestrator.run_agent(agent.clone(), sentence).await {
            Ok(result) => {
                if let Some(structured) = &result.structured {
                    println!(
                        "  name={}, age={}, city={}",
                        structured["name"], structured["age"], structured["city"]
                    );
                } else {
                    println!("  (no structured output) raw: {}", result.output);
                }
            }
            Err(e) => eprintln!("  Error: {}", e),
        }
        println!();
    }
}
