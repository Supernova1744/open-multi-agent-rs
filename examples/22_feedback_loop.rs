/// Example 22 — Feedback Loop: A → (B ↔ C) → D
///
/// Demonstrates a four-agent system where two agents (writer B and editor C)
/// form a feedback loop inside a larger pipeline:
///
///   A (Researcher)  →  brief
///   B (Writer)  ↔  C (Editor)   ← feedback loop, up to 3 rounds
///   D (Publisher)  →  final post
///
/// The FeedbackLoop handles all iteration automatically:
///   - Round 1: B writes from A's brief; C reviews
///   - Round 2+: B revises using A's brief + C's feedback; C re-reviews
///   - Exits when C says APPROVED or max_rounds (3) is reached
///
/// Run:
///   cargo run --example 22_feedback_loop
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    FeedbackLoop,
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

    let model = "qwen/qwen3-32b:nitro".to_string();

    // ── Agent A — Researcher ──────────────────────────────────────────────────
    let agent_a = AgentConfig {
        name: "researcher".to_string(),
        model: model.clone(),
        system_prompt: Some(
            "You are a research analyst. Given a topic, produce a concise structured brief \
             with: key facts, main arguments, and 3 suggested angles for a blog post. \
             Keep it under 200 words."
                .to_string(),
        ),
        max_turns: Some(2),
        ..Default::default()
    };

    // ── Agent B — Writer (worker in the loop) ─────────────────────────────────
    let agent_b = AgentConfig {
        name: "writer".to_string(),
        model: model.clone(),
        system_prompt: Some(
            "You are a technical blog writer. Write clear, engaging blog posts based on \
             the research brief provided. Aim for 150-200 words with an intro, body, and \
             conclusion. Address all reviewer feedback precisely."
                .to_string(),
        ),
        max_turns: Some(3),
        ..Default::default()
    };

    // ── Agent C — Editor (critic in the loop) ────────────────────────────────
    let agent_c = AgentConfig {
        name: "editor".to_string(),
        model: model.clone(),
        system_prompt: Some(
            "You are a strict technical editor. Review blog post drafts for clarity, \
             accuracy, structure, and engagement. \
             If the post is ready to publish, respond with APPROVED followed by a \
             brief reason. \
             Otherwise provide numbered, specific feedback on what must be improved. \
             Be concise."
                .to_string(),
        ),
        max_turns: Some(2),
        ..Default::default()
    };

    // ── Agent D — Publisher ───────────────────────────────────────────────────
    let agent_d = AgentConfig {
        name: "publisher".to_string(),
        model: model.clone(),
        system_prompt: Some(
            "You are a content publisher. Given an approved blog post, format it for \
             publication: add a compelling title, a one-sentence meta description, \
             and suggest 3 relevant tags. Output clean Markdown."
                .to_string(),
        ),
        max_turns: Some(2),
        ..Default::default()
    };

    // ── Topic ─────────────────────────────────────────────────────────────────
    let topic = "The impact of Rust's ownership model on systems programming safety";

    println!("{}", "═".repeat(60));
    println!("PIPELINE: A → (B ↔ C) → D");
    println!("Topic: {}", topic);
    println!("{}", "═".repeat(60));

    // ── STEP 1: Agent A — Research ────────────────────────────────────────────
    println!("\n🔬 Agent A (Researcher) running...");
    let mut agent_a_runner = Agent::new(
        agent_a,
        Arc::clone(&registry),
        Arc::clone(&executor),
    );
    let result_a = agent_a_runner
        .run(
            &format!("Research this topic and produce a structured brief: {}", topic),
            Arc::clone(&adapter),
        )
        .await
        .expect("Agent A failed");

    println!("✓ Research brief ready ({} chars)", result_a.output.len());
    println!("\n── Research Brief ───────────────────────────────────────");
    println!("{}", result_a.output);

    // ── STEP 2: B ↔ C Feedback Loop ──────────────────────────────────────────
    println!("\n{}", "═".repeat(60));
    println!("✍️  B ↔ C Feedback Loop (max 3 rounds)");
    println!("{}", "═".repeat(60));

    let bc_loop = FeedbackLoop::new(agent_b, agent_c)
        .max_rounds(3)
        .approval_signal("APPROVED")
        .on_round(|round, worker_out, critic_out, approved| {
            println!("\n── Round {} ──────────────────────────────────────────────", round);
            println!("  Writer ({} chars): {}...",
                worker_out.len(),
                worker_out.chars().take(120).collect::<String>().replace('\n', " ")
            );
            println!("  Editor: {}",
                critic_out.chars().take(120).collect::<String>().replace('\n', " ")
            );
            println!("  Approved: {}", if approved { "✅ YES" } else { "❌ NO — revising" });
        });

    // The loop receives A's research brief as the task
    let task_for_bc = format!(
        "Write a blog post about: {}\n\nUse this research brief:\n\n{}",
        topic, result_a.output
    );

    let result_bc = bc_loop
        .run(
            &task_for_bc,
            Arc::clone(&registry),
            Arc::clone(&executor),
            Arc::clone(&adapter),
        )
        .await
        .expect("Feedback loop failed");

    println!("\n{}", "═".repeat(60));
    println!(
        "✓ Loop complete — {} round(s), {}",
        result_bc.rounds,
        if result_bc.approved { "APPROVED ✅" } else { "max rounds reached ⚠️" }
    );

    // ── STEP 3: Agent D — Publish ─────────────────────────────────────────────
    println!("\n📤 Agent D (Publisher) running...");
    let mut agent_d_runner = Agent::new(
        agent_d,
        Arc::clone(&registry),
        Arc::clone(&executor),
    );
    let result_d = agent_d_runner
        .run(
            &format!(
                "Format this approved blog post for publication:\n\n{}",
                result_bc.final_output
            ),
            Arc::clone(&adapter),
        )
        .await
        .expect("Agent D failed");

    // ── Final output ──────────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(60));
    println!("🎉  PIPELINE COMPLETE");
    println!("{}", "═".repeat(60));
    println!("\n{}", result_d.output);

    println!("\n── Pipeline Stats ───────────────────────────────────────");
    println!("  A (research):  {} turns", result_a.turns);
    println!("  B↔C (loop):    {} round(s), approved={}", result_bc.rounds, result_bc.approved);
    for r in &result_bc.history {
        println!("    round {}: worker={} chars, critic approved={}",
            r.round, r.worker_output.len(), r.approved);
    }
    println!("  D (publish):   {} turns", result_d.turns);
}
