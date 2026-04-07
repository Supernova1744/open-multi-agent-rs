/// Integration tests for FeedbackLoop using the deterministic MockAdapter.
///
/// Each test scripts the exact responses the mock LLM will return so results
/// are fully deterministic with no network calls.
mod mock_adapter;

use mock_adapter::MockAdapter;
use open_multi_agent_rs::{
    FeedbackLoop,
    tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry},
    types::AgentConfig,
};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn worker_cfg() -> AgentConfig {
    AgentConfig {
        name:          "worker".to_string(),
        model:         "mock".to_string(),
        system_prompt: Some("You are a writer.".to_string()),
        max_turns:     Some(1),
        ..Default::default()
    }
}

fn critic_cfg() -> AgentConfig {
    AgentConfig {
        name:          "critic".to_string(),
        model:         "mock".to_string(),
        system_prompt: Some("You are an editor.".to_string()),
        max_turns:     Some(1),
        ..Default::default()
    }
}

async fn registry_and_executor() -> (Arc<Mutex<ToolRegistry>>, Arc<ToolExecutor>) {
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    (registry, executor)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn approves_on_first_round() {
    // Worker produces a draft; critic immediately approves.
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Draft: Great post about Rust."),   // worker round 1
        MockAdapter::text("APPROVED — excellent work."),      // critic round 1
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(3)
        .run("Write a blog post about Rust.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(result.approved);
    assert_eq!(result.rounds, 1);
    assert_eq!(result.final_output, "Draft: Great post about Rust.");
    assert!(result.history[0].approved);
}

#[tokio::test]
async fn approves_on_second_round() {
    // Critic rejects round 1, approves round 2.
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Draft v1: intro needs work."),         // worker round 1
        MockAdapter::text("Intro is too short. Expand it."),      // critic round 1 (reject)
        MockAdapter::text("Draft v2: expanded intro..."),         // worker round 2
        MockAdapter::text("APPROVED — much better."),             // critic round 2
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(3)
        .run("Write a blog post.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(result.approved);
    assert_eq!(result.rounds, 2);
    assert_eq!(result.final_output, "Draft v2: expanded intro...");
    assert!(!result.history[0].approved);
    assert!(result.history[1].approved);
}

#[tokio::test]
async fn exhausts_max_rounds_without_approval() {
    // Critic never approves; loop runs exactly max_rounds times.
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Draft 1"),
        MockAdapter::text("Not good enough."),   // reject
        MockAdapter::text("Draft 2"),
        MockAdapter::text("Still not great."),   // reject
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(2)
        .run("Write something.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(!result.approved);
    assert_eq!(result.rounds, 2);
    assert_eq!(result.final_output, "Draft 2");
}

#[tokio::test]
async fn max_rounds_one_runs_exactly_once() {
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Single draft."),
        MockAdapter::text("Needs work but no more rounds."),
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(1)
        .run("Task.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(!result.approved);
    assert_eq!(result.rounds, 1);
}

#[tokio::test]
async fn custom_approval_signal() {
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Draft content."),
        MockAdapter::text("LGTM — ship it!"),
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .approval_signal("LGTM")
        .run("Write.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(result.approved);
    assert_eq!(result.rounds, 1);
}

#[tokio::test]
async fn custom_approve_when_closure() {
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("First attempt."),
        MockAdapter::text("score: 6/10"),    // reject
        MockAdapter::text("Second attempt."),
        MockAdapter::text("score: 9/10"),    // approve
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(3)
        .approve_when(|out| out.contains("score: 9") || out.contains("score: 10"))
        .run("Write.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(result.approved);
    assert_eq!(result.rounds, 2);
}

#[tokio::test]
async fn approval_signal_case_insensitive() {
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Draft."),
        MockAdapter::text("approved — looks good"),   // lowercase
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .approval_signal("APPROVED")
        .run("Write.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(result.approved);
}

#[tokio::test]
async fn on_round_callback_fires_each_round() {
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("Draft 1."),
        MockAdapter::text("Needs work."),
        MockAdapter::text("Draft 2."),
        MockAdapter::text("APPROVED."),
    ]);
    let (registry, executor) = registry_and_executor().await;

    let fired_rounds = Arc::new(Mutex::new(Vec::<usize>::new()));
    let fired_clone  = Arc::clone(&fired_rounds);

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(3)
        .on_round(move |round, _w, _c, _ok| {
            let clone = Arc::clone(&fired_clone);
            // Blocking lock is fine inside a sync closure in tests
            futures::executor::block_on(async {
                clone.lock().await.push(round);
            });
        })
        .run("Write.", registry, executor, adapter)
        .await
        .unwrap();

    let rounds_fired = fired_rounds.lock().await.clone();
    assert_eq!(rounds_fired, vec![1, 2]);
    assert_eq!(result.rounds, 2);
}

#[tokio::test]
async fn history_records_all_rounds() {
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("v1"),
        MockAdapter::text("fix this"),
        MockAdapter::text("v2"),
        MockAdapter::text("APPROVED"),
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(3)
        .run("Write.", registry, executor, adapter)
        .await
        .unwrap();

    assert_eq!(result.history.len(), 2);
    assert_eq!(result.history[0].round, 1);
    assert_eq!(result.history[0].worker_output, "v1");
    assert_eq!(result.history[0].critic_output, "fix this");
    assert!(!result.history[0].approved);

    assert_eq!(result.history[1].round, 2);
    assert_eq!(result.history[1].worker_output, "v2");
    assert!(result.history[1].approved);
}

#[tokio::test]
async fn final_output_is_last_worker_output() {
    // Even without approval, final_output = last draft
    let adapter = MockAdapter::arc(vec![
        MockAdapter::text("draft A"),
        MockAdapter::text("not great"),
        MockAdapter::text("draft B"),
        MockAdapter::text("still no"),
        MockAdapter::text("draft C"),
        MockAdapter::text("nope"),
    ]);
    let (registry, executor) = registry_and_executor().await;

    let result = FeedbackLoop::new(worker_cfg(), critic_cfg())
        .max_rounds(3)
        .run("Write.", registry, executor, adapter)
        .await
        .unwrap();

    assert!(!result.approved);
    assert_eq!(result.final_output, "draft C");
}
