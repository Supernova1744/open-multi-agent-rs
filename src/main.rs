/// Demo: open-multi-agent-rs
///
/// Tests the Rust port against OpenRouter with model qwen/qwen3-6-plus:free.
///
/// Run with:
///   cargo run --bin demo
use open_multi_agent::{
    AgentConfig, OrchestratorConfig, OpenMultiAgent, TeamConfig, create_task,
};

fn load_env() {
    dotenvy::dotenv().ok(); // silently ignore if .env is absent
}

fn openrouter_api_key() -> String {
    let key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY environment variable must be set");
    // Strip surrounding quotes that Windows cmd users often include accidentally.
    key.trim_matches('"').to_string()
}

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const MODEL: &str = "qwen/qwen3.6-plus:free";

// ---------------------------------------------------------------------------
// Helper: make a default AgentConfig for OpenRouter
// ---------------------------------------------------------------------------

fn make_agent(name: &str, system_prompt: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: MODEL.to_string(),
        provider: Some("openrouter".to_string()),
        base_url: Some(OPENROUTER_BASE_URL.to_string()),
        api_key: Some(openrouter_api_key()),
        system_prompt: Some(system_prompt.to_string()),
        max_turns: Some(5),
        ..Default::default()
    }
}

fn make_orchestrator() -> OpenMultiAgent {
    OpenMultiAgent::new(OrchestratorConfig {
        default_model: MODEL.to_string(),
        default_provider: "openrouter".to_string(),
        default_base_url: Some(OPENROUTER_BASE_URL.to_string()),
        default_api_key: Some(openrouter_api_key()),
        max_concurrency: 3,
        on_progress: None,
        on_trace: None,
        on_approval: None,
    })
}

// ---------------------------------------------------------------------------
// Test 1: Single agent
// ---------------------------------------------------------------------------

async fn test_single_agent() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 1: Single Agent");
    println!("{}", "=".repeat(60));

    let orchestrator = make_orchestrator();
    let config = make_agent("assistant", "You are a concise, helpful assistant.");

    println!("Prompt: What is the capital of France? Answer in one word.");

    match orchestrator.run_agent(config, "What is the capital of France? Answer in one word.").await {
        Ok(result) => {
            println!("Success: {}", result.success);
            println!("Output: {}", result.output);
            println!("Tokens: input={}, output={}", result.token_usage.input_tokens, result.token_usage.output_tokens);
        }
        Err(e) => println!("Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Test 2: Multi-turn conversation (Agent::prompt)
// ---------------------------------------------------------------------------

async fn test_multi_turn() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 2: Multi-Turn Conversation");
    println!("{}", "=".repeat(60));

    let _orchestrator = make_orchestrator();
    let config = make_agent("tutor", "You are a concise math tutor.");

    let registry = open_multi_agent::ToolRegistry::new();
    let registry = std::sync::Arc::new(tokio::sync::Mutex::new(registry));
    let executor = std::sync::Arc::new(open_multi_agent::ToolExecutor::new(std::sync::Arc::clone(&registry)));
    let adapter = std::sync::Arc::from(open_multi_agent::create_adapter(
        "openrouter",
        Some(openrouter_api_key()),
        Some(OPENROUTER_BASE_URL.to_string()),
    ));

    let mut agent = open_multi_agent::agent::Agent::new(config, registry, executor);

    println!("Turn 1: What is 2 + 2?");
    match agent.prompt("What is 2 + 2?", std::sync::Arc::clone(&adapter)).await {
        Ok(r) => println!("  → {}", r.output),
        Err(e) => println!("  → Error: {}", e),
    }

    println!("Turn 2: Multiply that result by 3.");
    match agent.prompt("Multiply that result by 3.", std::sync::Arc::clone(&adapter)).await {
        Ok(r) => println!("  → {}", r.output),
        Err(e) => println!("  → Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Test 3: Explicit task pipeline (runTasks)
// ---------------------------------------------------------------------------

async fn test_task_pipeline() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 3: Explicit Task Pipeline (runTasks)");
    println!("{}", "=".repeat(60));

    let orchestrator = make_orchestrator();

    let team = TeamConfig {
        name: "pipeline-team".to_string(),
        agents: vec![
            make_agent("researcher", "You research topics and summarize key points concisely."),
            make_agent("writer", "You write clear, well-structured content based on provided research."),
        ],
        shared_memory: Some(true),
        max_concurrency: Some(2),
    };

    let task1 = {
        create_task(
            "Research Rust async ecosystem",
            "Research and summarize the key libraries in the Rust async ecosystem (tokio, async-std, etc.) in 2-3 bullet points.",
            Some("researcher".to_string()),
            vec![],
        )
    };

    let task1_id = task1.id.clone();

    let task2 = {
        create_task(
            "Write summary article",
            "Based on the research about the Rust async ecosystem, write a 2-paragraph introductory summary suitable for beginners.",
            Some("writer".to_string()),
            vec![task1_id],
        )
    };

    println!("Running 2-task pipeline: Research → Write");

    match orchestrator.run_tasks(&team, vec![task1, task2]).await {
        Ok(result) => {
            println!("Pipeline success: {}", result.success);
            println!("Total tokens: input={}, output={}",
                result.total_token_usage.input_tokens,
                result.total_token_usage.output_tokens
            );
            for (id, agent_result) in &result.agent_results {
                println!("\n--- Task {} ---", id);
                println!("{}", agent_result.output);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Test 4: Team with coordinator (runTeam)
// ---------------------------------------------------------------------------

async fn test_team_coordinator() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 4: Team with Coordinator Pattern (runTeam)");
    println!("{}", "=".repeat(60));

    let orchestrator = make_orchestrator();

    let team = TeamConfig {
        name: "content-team".to_string(),
        agents: vec![
            make_agent(
                "planner",
                "You break down content creation tasks into clear, actionable steps.",
            ),
            make_agent(
                "writer",
                "You write concise, engaging content based on given topics and outlines.",
            ),
        ],
        shared_memory: Some(true),
        max_concurrency: Some(2),
    };

    let goal = "Create a short blog post (2 paragraphs) explaining what Rust ownership is and why it matters.";
    println!("Goal: {}", goal);

    match orchestrator.run_team(&team, goal).await {
        Ok(result) => {
            println!("Team run success: {}", result.success);
            println!("Total tokens: input={}, output={}",
                result.total_token_usage.input_tokens,
                result.total_token_usage.output_tokens,
            );
            if let Some(coord) = result.agent_results.get("coordinator") {
                println!("\n=== FINAL OUTPUT (Coordinator) ===");
                println!("{}", coord.output);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    load_env();
    println!("open-multi-agent-rs — Rust port demo");
    println!("Using model: {}", MODEL);
    println!("Using API: OpenRouter");

    test_single_agent().await;
    test_multi_turn().await;
    test_task_pipeline().await;
    test_team_coordinator().await;

    println!("\n{}", "=".repeat(60));
    println!("All tests complete.");
}
