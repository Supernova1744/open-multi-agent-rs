/// Example 06 — Task Pipeline with Dependencies
///
/// Defines an explicit three-stage pipeline:
///   Researcher → Analyst → Writer
///
/// Each task depends on the previous one. The orchestrator resolves the
/// dependency graph, executes tasks in topological order, and stores results
/// in SharedMemory so downstream agents can read prior output.
///
/// Run:
///   cargo run --example 06_task_pipeline
use open_multi_agent_rs::{
    create_task, AgentConfig, OpenMultiAgent, OrchestratorConfig, TeamConfig,
};

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

fn make_agent(name: &str, role: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "qwen/qwen3.6-plus:free".to_string(),
        provider: Some("openrouter".to_string()),
        base_url: Some("https://openrouter.ai/api/v1".to_string()),
        api_key: Some(api_key()),
        system_prompt: Some(role.to_string()),
        max_turns: Some(3),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() {
    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "qwen/qwen3.6-plus:free".to_string(),
        default_provider: "openrouter".to_string(),
        default_base_url: Some("https://openrouter.ai/api/v1".to_string()),
        default_api_key: Some(api_key()),
        max_concurrency: 2,
        on_progress: None,
        on_trace: None,
        on_approval: None,
    });

    let team = TeamConfig {
        name: "content-pipeline".to_string(),
        agents: vec![
            make_agent(
                "researcher",
                "You are a researcher. Gather key facts and present them as 3 concise bullet points.",
            ),
            make_agent(
                "analyst",
                "You are an analyst. Take the research bullets and add one insight per point.",
            ),
            make_agent(
                "writer",
                "You are a technical writer. Turn the analysis into a single polished paragraph.",
            ),
        ],
        shared_memory: Some(true),
        max_concurrency: Some(1), // pipeline must be sequential
    };

    // Build the dependency chain.
    let task1 = create_task(
        "Research: async runtimes",
        "Research the top 3 Rust async runtimes (Tokio, async-std, smol). List key facts about each.",
        Some("researcher".to_string()),
        vec![],
    );
    let task1_id = task1.id.clone();

    let task2 = create_task(
        "Analyse the research",
        "Review the research about Rust async runtimes and add one real-world insight to each bullet.",
        Some("analyst".to_string()),
        vec![task1_id],
    );
    let task2_id = task2.id.clone();

    let task3 = create_task(
        "Write the summary",
        "Using the analysed bullets, write a single polished 3-sentence paragraph for a developer blog.",
        Some("writer".to_string()),
        vec![task2_id],
    );

    println!("Running 3-stage pipeline: Research → Analyse → Write\n");

    match orchestrator
        .run_tasks(&team, vec![task1, task2, task3])
        .await
    {
        Ok(result) => {
            println!("Pipeline success: {}\n", result.success);
            println!(
                "Total tokens — input: {}, output: {}\n",
                result.total_token_usage.input_tokens, result.total_token_usage.output_tokens
            );

            // Print each stage's output in order.
            for (id, agent_result) in &result.agent_results {
                println!("── Task {} ──", &id[..8]);
                println!("{}\n", agent_result.output);
            }
        }
        Err(e) => eprintln!("Pipeline error: {}", e),
    }
}
