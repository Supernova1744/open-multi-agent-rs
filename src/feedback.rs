//! Feedback loop — iterative worker → critic cycle.
//!
//! A [`FeedbackLoop`] pairs a *worker* agent with a *critic* agent and runs
//! them in alternating turns until the critic approves the output or
//! `max_rounds` is exhausted.
//!
//! ## How a round works
//!
//! ```text
//! Round 1:  worker ← original task
//!           critic ← worker's output
//!
//! Round 2+: worker ← original task + previous draft + critic feedback
//!           critic ← worker's revised output
//!
//! Exit when critic output satisfies the approval predicate, or max_rounds hit.
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use open_multi_agent_rs::feedback::FeedbackLoop;
//!
//! let result = FeedbackLoop::new(writer_config, editor_config)
//!     .max_rounds(3)
//!     .approval_signal("APPROVED")
//!     .run(task, registry, executor, adapter)
//!     .await?;
//!
//! println!("approved={} rounds={}", result.approved, result.rounds);
//! println!("{}", result.final_output);
//! ```

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    agent::Agent,
    error::Result,
    tool::{ToolExecutor, ToolRegistry},
    types::AgentConfig,
    LLMAdapter,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// One worker → critic exchange.
#[derive(Debug, Clone)]
pub struct Round {
    /// Iteration number (1-based).
    pub round: usize,
    /// Worker agent's output for this round.
    pub worker_output: String,
    /// Critic agent's evaluation.
    pub critic_output: String,
    /// Whether the critic approved this round.
    pub approved: bool,
}

/// Returned by [`FeedbackLoop::run`].
#[derive(Debug, Clone)]
pub struct FeedbackLoopResult {
    /// Worker's final output — pass this to downstream agents.
    pub final_output: String,
    /// `true` if the critic approved before `max_rounds` was reached.
    pub approved: bool,
    /// Number of rounds that ran.
    pub rounds: usize,
    /// Full transcript, one entry per round.
    pub history: Vec<Round>,
}

// ── Internal type aliases ─────────────────────────────────────────────────────

type ApprovalFn = Arc<dyn Fn(&str) -> bool + Send + Sync>;
type OnRoundFn  = Arc<dyn Fn(usize, &str, &str, bool) + Send + Sync>;

// ── FeedbackLoop ──────────────────────────────────────────────────────────────

/// Iterative worker → critic feedback loop.
///
/// Build with [`FeedbackLoop::new`], configure with the builder methods, then
/// call [`FeedbackLoop::run`].
pub struct FeedbackLoop {
    worker_config: AgentConfig,
    critic_config: AgentConfig,
    max_rounds:    usize,
    approve_fn:    ApprovalFn,
    on_round:      Option<OnRoundFn>,
}

impl FeedbackLoop {
    /// Create a feedback loop.
    ///
    /// Defaults:
    /// - `max_rounds = 3`
    /// - approves when critic output contains `"APPROVED"` (case-insensitive)
    pub fn new(worker: AgentConfig, critic: AgentConfig) -> Self {
        Self {
            worker_config: worker,
            critic_config: critic,
            max_rounds:    3,
            approve_fn:    Arc::new(|out: &str| out.to_uppercase().contains("APPROVED")),
            on_round:      None,
        }
    }

    /// Set the maximum number of worker → critic iterations (default: 3, minimum: 1).
    pub fn max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n.max(1);
        self
    }

    /// Approve when the critic output contains `signal` (case-insensitive).
    /// Replaces the default `"APPROVED"` check.
    pub fn approval_signal(mut self, signal: &str) -> Self {
        let s = signal.to_uppercase();
        self.approve_fn = Arc::new(move |out: &str| out.to_uppercase().contains(&s));
        self
    }

    /// Approve using a custom closure. Replaces `approval_signal`.
    ///
    /// ```rust,no_run
    /// .approve_when(|output| output.contains("score: 9") || output.contains("score: 10"))
    /// ```
    pub fn approve_when<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.approve_fn = Arc::new(f);
        self
    }

    /// Register a callback invoked after every round.
    ///
    /// Signature: `fn(round: usize, worker_output: &str, critic_output: &str, approved: bool)`
    pub fn on_round<F>(mut self, f: F) -> Self
    where
        F: Fn(usize, &str, &str, bool) + Send + Sync + 'static,
    {
        self.on_round = Some(Arc::new(f));
        self
    }

    /// Run the feedback loop.
    ///
    /// `task` is the original instruction for the worker. On round 1 the worker
    /// receives exactly `task`. On round 2+ it receives:
    ///
    /// ```text
    /// <original task>
    ///
    /// ---
    /// Your previous draft:
    /// <worker round N-1 output>
    ///
    /// Reviewer feedback (address all points):
    /// <critic round N-1 output>
    /// ```
    pub async fn run(
        self,
        task: &str,
        registry: Arc<Mutex<ToolRegistry>>,
        executor: Arc<ToolExecutor>,
        adapter: Arc<dyn LLMAdapter>,
    ) -> Result<FeedbackLoopResult> {
        let mut history:      Vec<Round> = Vec::new();
        let mut worker_input: String     = task.to_string();
        let mut final_output: String     = String::new();
        let mut approved:     bool       = false;

        for round in 1..=self.max_rounds {
            // ── Worker ────────────────────────────────────────────────────────
            let mut worker = Agent::new(
                self.worker_config.clone(),
                Arc::clone(&registry),
                Arc::clone(&executor),
            );
            let worker_result = worker.run(&worker_input, Arc::clone(&adapter)).await?;
            let worker_output = worker_result.output.clone();

            // ── Critic ────────────────────────────────────────────────────────
            let critic_task = format!(
                "Review the following output.\n\
                 If it is ready, respond with APPROVED.\n\
                 Otherwise provide specific, actionable feedback.\n\n\
                 Output to review:\n\n{}",
                worker_output
            );
            let mut critic = Agent::new(
                self.critic_config.clone(),
                Arc::clone(&registry),
                Arc::clone(&executor),
            );
            let critic_result = critic.run(&critic_task, Arc::clone(&adapter)).await?;
            let critic_output = critic_result.output.clone();

            let round_approved = (self.approve_fn)(&critic_output);

            // ── Callback ──────────────────────────────────────────────────────
            if let Some(ref cb) = self.on_round {
                cb(round, &worker_output, &critic_output, round_approved);
            }

            history.push(Round {
                round,
                worker_output: worker_output.clone(),
                critic_output: critic_output.clone(),
                approved: round_approved,
            });

            final_output = worker_output;

            if round_approved {
                approved = true;
                break;
            }

            // ── Build next round context ───────────────────────────────────────
            worker_input = format!(
                "{}\n\n---\nYour previous draft:\n{}\n\nReviewer feedback (address all points):\n{}",
                task, final_output, critic_output
            );
        }

        Ok(FeedbackLoopResult {
            final_output,
            approved,
            rounds: history.len(),
            history,
        })
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentConfig;

    fn worker_cfg() -> AgentConfig {
        AgentConfig {
            name:          "worker".to_string(),
            model:         "mock".to_string(),
            system_prompt: Some("You are a writer.".to_string()),
            ..Default::default()
        }
    }

    fn critic_cfg() -> AgentConfig {
        AgentConfig {
            name:          "critic".to_string(),
            model:         "mock".to_string(),
            system_prompt: Some("You are an editor.".to_string()),
            ..Default::default()
        }
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    #[test]
    fn default_max_rounds_is_3() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg());
        assert_eq!(fl.max_rounds, 3);
    }

    #[test]
    fn max_rounds_setter() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg()).max_rounds(5);
        assert_eq!(fl.max_rounds, 5);
    }

    #[test]
    fn max_rounds_minimum_is_1() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg()).max_rounds(0);
        assert_eq!(fl.max_rounds, 1);
    }

    // ── Approval predicate ────────────────────────────────────────────────────

    #[test]
    fn default_approval_signal_case_insensitive() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg());
        assert!((fl.approve_fn)("APPROVED great work"));
        assert!((fl.approve_fn)("approved!"));
        assert!((fl.approve_fn)("Approved."));
        assert!(!(fl.approve_fn)("Needs more work."));
        assert!(!(fl.approve_fn)(""));
    }

    #[test]
    fn custom_approval_signal() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg())
            .approval_signal("LGTM");
        assert!((fl.approve_fn)("LGTM, ship it"));
        assert!((fl.approve_fn)("lgtm"));
        assert!(!(fl.approve_fn)("APPROVED"));
    }

    #[test]
    fn approve_when_custom_closure() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg())
            .approve_when(|out| out.contains("score: 9") || out.contains("score: 10"));
        assert!((fl.approve_fn)("score: 9/10, well done"));
        assert!((fl.approve_fn)("score: 10 — perfect"));
        assert!(!(fl.approve_fn)("score: 7, needs work"));
        assert!(!(fl.approve_fn)("APPROVED"));
    }

    #[test]
    fn approval_signal_overrides_default() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg())
            .approval_signal("SHIP IT");
        // Default "APPROVED" should no longer trigger
        assert!(!(fl.approve_fn)("APPROVED"));
        assert!((fl.approve_fn)("SHIP IT"));
    }

    #[test]
    fn approve_when_overrides_approval_signal() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg())
            .approval_signal("APPROVED")
            .approve_when(|out| out.starts_with("YES"));
        assert!((fl.approve_fn)("YES this is great"));
        assert!(!(fl.approve_fn)("APPROVED"));
    }

    // ── on_round callback registration ───────────────────────────────────────

    #[test]
    fn on_round_callback_is_stored() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg())
            .on_round(|_round, _w, _c, _ok| {});
        assert!(fl.on_round.is_some());
    }

    #[test]
    fn no_callback_by_default() {
        let fl = FeedbackLoop::new(worker_cfg(), critic_cfg());
        assert!(fl.on_round.is_none());
    }

    // ── FeedbackLoopResult fields ─────────────────────────────────────────────

    #[test]
    fn result_fields_accessible() {
        let r = FeedbackLoopResult {
            final_output: "done".to_string(),
            approved:     true,
            rounds:       2,
            history:      vec![
                Round { round: 1, worker_output: "v1".into(), critic_output: "fix it".into(), approved: false },
                Round { round: 2, worker_output: "v2".into(), critic_output: "APPROVED".into(), approved: true  },
            ],
        };
        assert_eq!(r.final_output, "done");
        assert!(r.approved);
        assert_eq!(r.rounds, 2);
        assert_eq!(r.history[0].round, 1);
        assert!(!r.history[0].approved);
        assert!(r.history[1].approved);
    }

    #[test]
    fn round_fields_accessible() {
        let r = Round {
            round:          3,
            worker_output:  "draft".to_string(),
            critic_output:  "APPROVED".to_string(),
            approved:       true,
        };
        assert_eq!(r.round, 3);
        assert!(r.approved);
    }
}
