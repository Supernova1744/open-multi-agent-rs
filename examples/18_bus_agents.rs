/// Example 18 — MessageBus Multi-Agent Communication
///
/// Demonstrates the MessageBus broadcaster tools:
///   • bus_publish — publish a message to another agent or broadcast to all
///   • bus_read    — read messages addressed to the current agent
///
/// Two agents share a MessageBus and coordinate a task:
///   • "researcher" — searches/synthesizes information, publishes findings
///   • "writer"     — waits for the researcher's findings, writes a summary
///
/// The bus tools are injected via `register_bus_tools()` with a shared
/// MessageBus instance.
///
/// Run:
///   cargo run --example 18_bus_agents
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    messaging::MessageBus,
    tool::{
        built_in::{register_built_in_tools, register_bus_tools},
        ToolExecutor, ToolRegistry,
    },
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
    // ── Shared MessageBus ─────────────────────────────────────────────────────
    let bus = Arc::new(MessageBus::new());

    // ── Shared tool registry ─────────────────────────────────────────────────
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
        // Inject the MessageBus so agents can publish/read during tool calls.
        register_bus_tools(&mut reg, Arc::clone(&bus)).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    println!("{}", "─".repeat(60));
    println!("Step 1: Researcher agent gathers facts and publishes to writer");
    println!("{}", "─".repeat(60));

    // ── Step 1: Researcher agent ──────────────────────────────────────────────
    let researcher_config = AgentConfig {
        name: "researcher".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a research agent. Gather the requested information, \
             then use bus_publish to send your findings to the 'writer' agent \
             before finishing. Be concise and factual."
            .to_string(),
        ),
        max_turns: Some(4),
        tools: Some(vec![
            "bus_publish".to_string(),
            "math_eval".to_string(),
            "datetime".to_string(),
        ]),
        ..Default::default()
    };

    let mut researcher = Agent::new(researcher_config, Arc::clone(&registry), Arc::clone(&executor));

    let research_task = "\
        Research the following and then publish your findings to the 'writer' agent:\n\
        1. Use math_eval to calculate: compound interest for $10000 at 5% for 10 years \
           (formula: 10000 * 1.05^10)\n\
        2. Use datetime to get the current UTC time.\n\
        3. Use bus_publish (to='writer') to send a message with your findings, \
           formatted as: 'FINDINGS: compound_result=<result>, timestamp=<time>'";

    match researcher.run(research_task, Arc::clone(&adapter)).await {
        Ok(AgentRunResult { turns, tool_calls, .. }) => {
            println!("Researcher done in {} turns.", turns);
            for tc in &tool_calls {
                println!("  • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
        }
        Err(e) => eprintln!("Researcher error: {}", e),
    }

    // Check what was published on the bus
    let messages = bus.get_unread("writer");
    println!("\nMessages on bus for 'writer': {}", messages.len());
    for msg in &messages {
        println!("  [{}→{}] {}", msg.from, msg.to, msg.content);
    }

    println!("\n{}", "─".repeat(60));
    println!("Step 2: Writer agent reads the bus and produces a report");
    println!("{}", "─".repeat(60));

    // ── Step 2: Writer agent ──────────────────────────────────────────────────
    let writer_config = AgentConfig {
        name: "writer".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a report writer agent. First use bus_read to retrieve messages \
             addressed to you, then write a clear summary based on those findings. \
             After writing, broadcast a completion notice with bus_publish (to='*')."
            .to_string(),
        ),
        max_turns: Some(4),
        tools: Some(vec![
            "bus_read".to_string(),
            "bus_publish".to_string(),
        ]),
        ..Default::default()
    };

    let mut writer = Agent::new(writer_config, Arc::clone(&registry), Arc::clone(&executor));

    let write_task = "\
        1. Use bus_read to get messages addressed to 'writer'.\n\
        2. Write a concise financial report based on the researcher's findings.\n\
        3. Use bus_publish with to='*' to broadcast: 'REPORT COMPLETE: <one-line summary>'";

    match writer.run(write_task, Arc::clone(&adapter)).await {
        Ok(AgentRunResult { output, turns, tool_calls, .. }) => {
            println!("Writer done in {} turns.", turns);
            for tc in &tool_calls {
                println!("  • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
            println!("\nWriter report:\n{}", output);
        }
        Err(e) => eprintln!("Writer error: {}", e),
    }

    // Show the broadcast message
    let broadcasts = bus.get_unread("researcher");
    println!("\nBroadcast received by researcher:");
    for msg in &broadcasts {
        println!("  [{}] {}", msg.from, msg.content);
    }

    println!("\nTotal messages on bus: {}", {
        let all: Vec<_> = bus.get_all("writer");
        all.len()
    });
}
