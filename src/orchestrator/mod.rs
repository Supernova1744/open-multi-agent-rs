/// OpenMultiAgent — top-level orchestrator.
///
/// Provides three execution modes:
/// 1. run_agent  — single agent, one-shot
/// 2. run_team   — coordinator pattern: LLM decomposes goal into tasks
/// 3. run_tasks  — explicit task pipeline with user-defined dependencies
///
/// Features: retry with exponential backoff, approval gates, trace events.
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::{pool::AgentPool, Agent, RunOptions};
use crate::error::{AgentError, Result};
use crate::llm::{create_adapter, LLMAdapter};
use crate::memory::{InMemoryStore, MemoryStore, SharedMemory};
use crate::task::{
    create_task,
    queue::TaskQueue,
    scheduler::{Scheduler, SchedulingStrategy},
};
use crate::tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry};
use crate::trace::{emit_trace, generate_run_id, now_ms};
use crate::types::{
    AgentConfig, AgentRunResult, ContentBlock, LLMChatOptions, LLMMessage, OnTraceFn, Role, Task,
    TaskTrace, TeamConfig, TokenUsage, TraceEventBase,
};

// ---------------------------------------------------------------------------
// Orchestrator config
// ---------------------------------------------------------------------------

pub struct OrchestratorConfig {
    pub max_concurrency: usize,
    pub default_model: String,
    pub default_provider: String,
    pub default_base_url: Option<String>,
    pub default_api_key: Option<String>,
    /// Progress callback (defaults to printing to stdout).
    pub on_progress: Option<Box<dyn Fn(String) + Send + Sync>>,
    /// Trace callback — receives observability spans for all LLM calls, tool
    /// calls, task completions, and agent runs.
    pub on_trace: Option<OnTraceFn>,
    /// Approval gate called between task execution rounds.
    ///
    /// Receives (`completed_tasks`, `next_tasks`). Return `true` to continue,
    /// `false` to abort (remaining tasks are skipped).
    pub on_approval:
        Option<Arc<dyn Fn(Vec<Task>, Vec<Task>) -> BoxFuture<'static, bool> + Send + Sync>>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        OrchestratorConfig {
            max_concurrency: 5,
            default_model: "qwen/qwen3.6-plus:free".to_string(),
            default_provider: "openrouter".to_string(),
            default_base_url: None,
            default_api_key: None,
            on_progress: None,
            on_trace: None,
            on_approval: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Team run result
// ---------------------------------------------------------------------------

pub struct TeamRunResult {
    pub success: bool,
    pub agent_results: HashMap<String, AgentRunResult>,
    pub total_token_usage: TokenUsage,
}

// ---------------------------------------------------------------------------
// Retry with exponential backoff
// ---------------------------------------------------------------------------

/// Maximum delay cap to prevent runaway backoff (30 seconds).
const MAX_RETRY_DELAY_MS: u64 = 30_000;

/// Compute retry delay for a given attempt, capped at `MAX_RETRY_DELAY_MS`.
pub fn compute_retry_delay(base_delay: u64, backoff: f64, attempt: u32) -> u64 {
    let delay = (base_delay as f64) * backoff.powi(attempt as i32 - 1);
    (delay as u64).min(MAX_RETRY_DELAY_MS)
}

/// Execute an agent task with optional retry and exponential backoff.
///
/// Reads `max_retries`, `retry_delay_ms`, `retry_backoff` from the task.
/// Accumulates token usage across all attempts.
pub async fn execute_with_retry(
    run: impl Fn() -> BoxFuture<'static, Result<AgentRunResult>>,
    task: &Task,
    on_retry: Option<Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>>,
) -> AgentRunResult {
    let max_retries = task.max_retries.unwrap_or(0);
    let max_attempts = max_retries + 1;
    let base_delay = task.retry_delay_ms.unwrap_or(1000);
    let backoff = task.retry_backoff.unwrap_or(2.0).max(1.0);

    let mut total_usage = TokenUsage::default();
    let mut last_error = String::new();

    for attempt in 1..=max_attempts {
        match run().await {
            Ok(result) => {
                let merged_usage = total_usage.add(&result.token_usage);
                let merged = AgentRunResult {
                    token_usage: merged_usage,
                    ..result.clone()
                };
                if result.success {
                    return merged;
                }
                last_error = result.output.clone();
                total_usage = total_usage.add(&result.token_usage);
                if attempt < max_attempts {
                    let delay = compute_retry_delay(base_delay, backoff, attempt);
                    if let Some(ref cb) = on_retry {
                        cb(attempt, max_attempts, last_error.clone(), delay);
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                } else {
                    return AgentRunResult {
                        token_usage: total_usage.clone(),
                        ..merged
                    };
                }
            }
            Err(e) => {
                last_error = e.to_string();
                if attempt < max_attempts {
                    let delay = compute_retry_delay(base_delay, backoff, attempt);
                    if let Some(ref cb) = on_retry {
                        cb(attempt, max_attempts, last_error.clone(), delay);
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                } else {
                    return AgentRunResult {
                        success: false,
                        output: last_error,
                        messages: Vec::new(),
                        token_usage: total_usage,
                        tool_calls: Vec::new(),
                        turns: 0,
                        structured: None,
                    };
                }
            }
        }
    }

    // Unreachable but required by the type system.
    AgentRunResult {
        success: false,
        output: last_error,
        messages: Vec::new(),
        token_usage: total_usage,
        tool_calls: Vec::new(),
        turns: 0,
        structured: None,
    }
}

fn orchestrator_retry_logger(attempt: u32, max: u32, err: String, delay: u64) {
    println!(
        "[orchestrator] Task retry {}/{}: {} — retrying in {}ms",
        attempt, max, err, delay
    );
}

// ---------------------------------------------------------------------------
// Coordinator task format (parsed from LLM JSON output)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct CoordinatorTask {
    title: String,
    description: String,
    #[serde(default)]
    assignee: Option<String>,
    #[serde(default)]
    depends_on: Vec<String>,
}

// ---------------------------------------------------------------------------
// OpenMultiAgent
// ---------------------------------------------------------------------------

pub struct OpenMultiAgent {
    pub config: OrchestratorConfig,
}

impl OpenMultiAgent {
    pub fn new(config: OrchestratorConfig) -> Self {
        OpenMultiAgent { config }
    }

    /// Build an LLM adapter for a given agent config, falling back to orchestrator defaults.
    fn build_adapter(&self, agent_config: &AgentConfig) -> Arc<dyn LLMAdapter> {
        let provider = agent_config
            .provider
            .as_deref()
            .unwrap_or(&self.config.default_provider)
            .to_string();
        let api_key = agent_config
            .api_key
            .clone()
            .or_else(|| self.config.default_api_key.clone());
        let base_url = agent_config
            .base_url
            .clone()
            .or_else(|| self.config.default_base_url.clone());
        Arc::from(create_adapter(&provider, api_key, base_url))
    }

    /// Build a fresh agent with its own registry/executor and built-in tools.
    async fn build_agent(&self, config: AgentConfig) -> (Agent, Arc<dyn LLMAdapter>) {
        let mut registry = ToolRegistry::new();
        register_built_in_tools(&mut registry).await;
        let registry = Arc::new(Mutex::new(registry));
        let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
        let adapter = self.build_adapter(&config);
        (Agent::new(config, registry, executor), adapter)
    }

    // -------------------------------------------------------------------------
    // Mode 1: Single agent run
    // -------------------------------------------------------------------------

    pub async fn run_agent(&self, config: AgentConfig, prompt: &str) -> Result<AgentRunResult> {
        let (mut agent, adapter) = self.build_agent(config).await;
        let opts = self.build_run_opts(None);
        agent.run_with_opts(prompt, adapter, opts).await
    }

    // -------------------------------------------------------------------------
    // Mode 2: Team with coordinator pattern
    // -------------------------------------------------------------------------

    pub async fn run_team(&self, team: &TeamConfig, goal: &str) -> Result<TeamRunResult> {
        let shared_store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let shared_memory = Arc::new(SharedMemory::new(Arc::clone(&shared_store)));
        let run_id = generate_run_id();

        // Step 1: Coordinator decomposes goal into tasks.
        self.emit_progress("Coordinator decomposing goal into tasks...");

        let coordinator_config = AgentConfig {
            name: "coordinator".to_string(),
            model: team
                .agents
                .first()
                .map(|a| a.model.clone())
                .unwrap_or_else(|| self.config.default_model.clone()),
            system_prompt: Some(self.build_coordinator_prompt(team)),
            ..Default::default()
        };

        let coordinator_adapter = self.build_adapter(&coordinator_config);

        let coordinator_prompt = format!(
            "Goal: {}\n\nAvailable agents: {}\n\nReturn a JSON array of tasks. Each task: {{\"title\": string, \"description\": string, \"assignee\": agent_name_or_null, \"depends_on\": [task_id_array]}}.\n\nIMPORTANT: task IDs for depends_on should use 0-based index strings like \"0\", \"1\", \"2\". Return ONLY the JSON array, no other text.",
            goal,
            team.agents
                .iter()
                .map(|a| format!("{}: {}", a.name, a.system_prompt.as_deref().unwrap_or("general assistant")))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let coord_messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: coordinator_prompt,
            }],
        }];

        let coord_chat_opts = LLMChatOptions {
            model: coordinator_config.model.clone(),
            tools: None,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            system_prompt: coordinator_config.system_prompt.clone(),
        };

        let coord_response = coordinator_adapter
            .chat(&coord_messages, &coord_chat_opts)
            .await?;

        let coord_text: String = coord_response
            .content
            .iter()
            .filter_map(|b| b.as_text())
            .collect();

        // Step 2: Parse task list from coordinator output.
        let coord_tasks = self.parse_coordinator_tasks(&coord_text)?;
        self.emit_progress(&format!("Coordinator created {} tasks.", coord_tasks.len()));

        // Step 3: Build queue with correct dependency IDs.
        let mut queue = TaskQueue::new();
        let mut task_id_map: HashMap<String, String> = HashMap::new();

        let mut tasks: Vec<Task> = coord_tasks
            .iter()
            .enumerate()
            .map(|(i, ct)| {
                let t = create_task(
                    ct.title.clone(),
                    ct.description.clone(),
                    ct.assignee.clone(),
                    Vec::new(),
                );
                task_id_map.insert(i.to_string(), t.id.clone());
                t
            })
            .collect();

        for (i, coord_task) in coord_tasks.iter().enumerate() {
            let real_deps: Vec<String> = coord_task
                .depends_on
                .iter()
                .filter_map(|dep_idx| task_id_map.get(dep_idx).cloned())
                .collect();
            tasks[i].depends_on = real_deps;
        }

        queue.add_batch(tasks);

        // Step 4: Execute queue.
        let mut agent_results: HashMap<String, AgentRunResult> = HashMap::new();
        let mut total_usage = TokenUsage::default();
        let pool = AgentPool::new(self.config.max_concurrency);

        let result = self
            .execute_queue(
                &mut queue,
                team,
                &shared_memory,
                &pool,
                &mut agent_results,
                &mut total_usage,
                &run_id,
            )
            .await;

        // Step 5: Coordinator synthesizes final answer.
        self.emit_progress("Coordinator synthesizing final answer...");

        let memory_summary = shared_memory.to_markdown().await;
        let synthesis_prompt = format!(
            "Goal: {}\n\n{}\n\nBased on the above task results, write a final, coherent, polished answer to the original goal. Do NOT return JSON. Write a natural language response.",
            goal, memory_summary
        );

        let synthesis_opts = LLMChatOptions {
            model: coordinator_config.model.clone(),
            tools: None,
            max_tokens: Some(2048),
            temperature: Some(0.5),
            system_prompt: Some(
                "You are a helpful assistant that synthesizes information into clear, well-written final answers."
                    .to_string(),
            ),
        };

        let synthesis_messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: synthesis_prompt,
            }],
        }];

        let synthesis_response = coordinator_adapter
            .chat(&synthesis_messages, &synthesis_opts)
            .await
            .unwrap_or_else(|_| coord_response.clone());

        let final_output: String = synthesis_response
            .content
            .iter()
            .filter_map(|b| b.as_text())
            .collect();

        total_usage = total_usage.add(&synthesis_response.usage);

        agent_results.insert(
            "coordinator".to_string(),
            AgentRunResult {
                success: true,
                output: final_output,
                messages: Vec::new(),
                token_usage: synthesis_response.usage,
                tool_calls: Vec::new(),
                turns: 1,
                structured: None,
            },
        );

        Ok(TeamRunResult {
            success: result.is_ok(),
            agent_results,
            total_token_usage: total_usage,
        })
    }

    // -------------------------------------------------------------------------
    // Mode 3: Explicit task pipeline
    // -------------------------------------------------------------------------

    pub async fn run_tasks(&self, team: &TeamConfig, tasks: Vec<Task>) -> Result<TeamRunResult> {
        let shared_store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let shared_memory = Arc::new(SharedMemory::new(Arc::clone(&shared_store)));
        let run_id = generate_run_id();

        let mut queue = TaskQueue::new();
        queue.add_batch(tasks);

        let mut agent_results: HashMap<String, AgentRunResult> = HashMap::new();
        let mut total_usage = TokenUsage::default();
        let pool = AgentPool::new(self.config.max_concurrency);

        let success = self
            .execute_queue(
                &mut queue,
                team,
                &shared_memory,
                &pool,
                &mut agent_results,
                &mut total_usage,
                &run_id,
            )
            .await
            .is_ok();

        Ok(TeamRunResult {
            success,
            agent_results,
            total_token_usage: total_usage,
        })
    }

    // -------------------------------------------------------------------------
    // Internal: execute queue until all tasks are done
    // -------------------------------------------------------------------------

    async fn execute_queue(
        &self,
        queue: &mut TaskQueue,
        team: &TeamConfig,
        shared_memory: &Arc<SharedMemory>,
        _pool: &AgentPool,
        agent_results: &mut HashMap<String, AgentRunResult>,
        total_usage: &mut TokenUsage,
        run_id: &str,
    ) -> Result<()> {
        let mut scheduler = Scheduler::new(SchedulingStrategy::DependencyFirst);

        loop {
            if queue.is_complete() {
                break;
            }

            // Auto-assign unassigned pending tasks.
            let all_tasks = queue.list();
            let assignments = scheduler.schedule(&all_tasks, &team.agents);
            for (task_id, agent_name) in &assignments {
                let _ = queue.set_assignee(task_id, agent_name);
            }

            // Collect this round's ready tasks.
            let pending: Vec<Task> = queue.pending_tasks();
            if pending.is_empty() {
                break;
            }

            // --- Approval gate ---
            if let Some(ref on_approval) = self.config.on_approval {
                let completed: Vec<Task> = queue
                    .list()
                    .into_iter()
                    .filter(|t| t.status == crate::types::TaskStatus::Completed)
                    .collect();
                let approved = on_approval(completed, pending.clone()).await;
                if !approved {
                    self.emit_progress("Approval gate rejected — skipping remaining tasks.");
                    queue.skip_remaining("approval gate rejected");
                    break;
                }
            }

            // Execute all pending tasks concurrently.
            let mut join_handles = Vec::new();

            for task in &pending {
                let _ = queue.set_in_progress(&task.id);
            }

            for task in pending {
                let task_id = task.id.clone();
                let task_title = task.title.clone();
                let agent_name = task.assignee.clone().unwrap_or_else(|| {
                    team.agents
                        .first()
                        .map(|a| a.name.clone())
                        .unwrap_or_default()
                });

                let agent_config = team
                    .agents
                    .iter()
                    .find(|a| a.name == agent_name)
                    .cloned()
                    .unwrap_or_else(|| team.agents.first().cloned().unwrap_or_default());

                let memory_md = shared_memory.to_markdown().await;
                let task_prompt = format!(
                    "{}\n\n{}\n\n{}",
                    task.description,
                    if memory_md.is_empty() {
                        String::new()
                    } else {
                        format!("Context from previous tasks:\n{}", memory_md)
                    },
                    "Complete the task above and provide your result."
                );

                let adapter = self.build_adapter(&agent_config);

                let mut registry = ToolRegistry::new();
                register_built_in_tools(&mut registry).await;
                let registry = Arc::new(Mutex::new(registry));
                let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

                let agent = Agent::new(agent_config.clone(), registry, executor);

                self.emit_progress(&format!(
                    "Starting task: {} (agent: {})",
                    task_title, agent_name
                ));

                let opts = self.build_run_opts(Some(task_id.clone()));
                let task_start_ms = now_ms();
                let on_trace = self.config.on_trace.clone();
                let run_id_owned = run_id.to_string();
                let task_title_owned = task_title.clone();
                let task_id_owned = task_id.clone();
                let agent_name_owned = agent_name.clone();

                // Wrap in Arc so we can use it in a boxed async closure for retry.
                let task_arc = Arc::new(task.clone());
                let agent = Arc::new(Mutex::new(agent));
                let adapter = Arc::clone(&adapter);

                let result = {
                    let task_arc = Arc::clone(&task_arc);
                    let agent = Arc::clone(&agent);
                    let adapter = Arc::clone(&adapter);
                    let task_prompt = task_prompt.clone();
                    let opts_clone = opts.clone();

                    execute_with_retry(
                        move || {
                            let agent = Arc::clone(&agent);
                            let adapter = Arc::clone(&adapter);
                            let prompt = task_prompt.clone();
                            let opts = opts_clone.clone();
                            Box::pin(async move {
                                let mut ag = agent.lock().await;
                                ag.run_with_opts(&prompt, adapter, opts).await
                            })
                        },
                        &task_arc,
                        Some(Arc::new(orchestrator_retry_logger)
                            as Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>),
                    )
                    .await
                };

                let task_end_ms = now_ms();
                let success = result.success;

                // Emit task trace.
                emit_trace(
                    &on_trace,
                    crate::types::TraceEvent::Task(TaskTrace {
                        base: TraceEventBase {
                            run_id: run_id_owned,
                            start_ms: task_start_ms,
                            end_ms: task_end_ms,
                            duration_ms: task_end_ms - task_start_ms,
                            agent: agent_name_owned.clone(),
                            task_id: Some(task_id_owned.clone()),
                        },
                        task_id: task_id_owned.clone(),
                        task_title: task_title_owned.clone(),
                        success,
                        retries: task_arc.max_retries.unwrap_or(0),
                    }),
                );

                join_handles.push((
                    task_id,
                    task_title,
                    agent_name_owned,
                    Ok::<AgentRunResult, crate::error::AgentError>(result),
                ));
            }

            // Process results.
            for (task_id, task_title, agent_name, result) in join_handles {
                match result {
                    Ok(run_result) => {
                        *total_usage = total_usage.add(&run_result.token_usage);
                        if run_result.success {
                            self.emit_progress(&format!("Completed task: {}", task_title));
                            shared_memory
                                .write(&agent_name, &task_id, &run_result.output)
                                .await;
                            let _ = queue.complete(&task_id, Some(run_result.output.clone()));
                        } else {
                            self.emit_progress(&format!(
                                "Failed task: {} — {}",
                                task_title, run_result.output
                            ));
                            let _ = queue.fail(&task_id, run_result.output.clone());
                        }
                        agent_results.insert(task_id.clone(), run_result);
                    }
                    Err(e) => {
                        self.emit_progress(&format!("Failed task: {} — {}", task_title, e));
                        let _ = queue.fail(&task_id, e.to_string());
                    }
                }
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Build RunOptions with trace forwarding from orchestrator config.
    fn build_run_opts(&self, task_id: Option<String>) -> RunOptions {
        RunOptions {
            on_trace: self.config.on_trace.clone(),
            task_id,
            ..Default::default()
        }
    }

    fn build_coordinator_prompt(&self, team: &TeamConfig) -> String {
        let agent_list = team
            .agents
            .iter()
            .map(|a| {
                format!(
                    "- {}: {}",
                    a.name,
                    a.system_prompt.as_deref().unwrap_or("general assistant")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "You are a coordinator for a team of AI agents. Your role is to decompose goals into tasks and assign them to the right agents.\n\nAvailable agents:\n{}\n\nWhen given a goal, respond with a JSON array of task objects. Each object must have:\n- title: string\n- description: string (detailed instructions for the agent)\n- assignee: agent name or null\n- depends_on: array of task indices (0-based) that must complete first\n\nReturn ONLY valid JSON, no other text.",
            agent_list
        )
    }

    fn parse_coordinator_tasks(&self, text: &str) -> Result<Vec<CoordinatorTask>> {
        let json_str = if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                &text[start..=end]
            } else {
                text
            }
        } else {
            text
        };

        serde_json::from_str(json_str).map_err(|e| {
            AgentError::Other(format!(
                "Failed to parse coordinator output as JSON tasks: {}. Output was: {}",
                e, text
            ))
        })
    }

    fn emit_progress(&self, message: &str) {
        if let Some(cb) = &self.config.on_progress {
            cb(message.to_string());
        } else {
            println!("[orchestrator] {}", message);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_retry_delay_first_attempt() {
        // attempt=1: base_delay * backoff^0 = base_delay
        assert_eq!(compute_retry_delay(1000, 2.0, 1), 1000);
    }

    #[test]
    fn compute_retry_delay_second_attempt() {
        assert_eq!(compute_retry_delay(1000, 2.0, 2), 2000);
    }

    #[test]
    fn compute_retry_delay_third_attempt() {
        assert_eq!(compute_retry_delay(1000, 2.0, 3), 4000);
    }

    #[test]
    fn compute_retry_delay_caps_at_max() {
        // Very large attempt — should cap at 30_000
        assert_eq!(compute_retry_delay(1000, 2.0, 100), MAX_RETRY_DELAY_MS);
    }

    #[test]
    fn compute_retry_delay_backoff_one_is_constant() {
        assert_eq!(compute_retry_delay(500, 1.0, 5), 500);
    }

    #[tokio::test]
    async fn execute_with_retry_succeeds_first_attempt() {
        use crate::task::create_task;
        let task = create_task("t".to_string(), "d".to_string(), None, vec![]);
        let called = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let called_clone = Arc::clone(&called);

        let result = execute_with_retry(
            move || {
                let c = Arc::clone(&called_clone);
                Box::pin(async move {
                    c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(AgentRunResult {
                        success: true,
                        output: "done".to_string(),
                        messages: vec![],
                        token_usage: TokenUsage::default(),
                        tool_calls: vec![],
                        turns: 1,
                        structured: None,
                    })
                })
            },
            &task,
            None::<Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>>,
        )
        .await;

        assert!(result.success);
        assert_eq!(called.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn execute_with_retry_retries_on_failure() {
        use crate::task::create_task;
        let mut task = create_task("t".to_string(), "d".to_string(), None, vec![]);
        task.max_retries = Some(2);
        task.retry_delay_ms = Some(1); // near-zero for fast tests

        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let result = execute_with_retry(
            move || {
                let c = Arc::clone(&call_count_clone);
                Box::pin(async move {
                    let n = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if n < 2 {
                        Ok(AgentRunResult {
                            success: false,
                            output: "fail".to_string(),
                            messages: vec![],
                            token_usage: TokenUsage::default(),
                            tool_calls: vec![],
                            turns: 0,
                            structured: None,
                        })
                    } else {
                        Ok(AgentRunResult {
                            success: true,
                            output: "ok".to_string(),
                            messages: vec![],
                            token_usage: TokenUsage::default(),
                            tool_calls: vec![],
                            turns: 1,
                            structured: None,
                        })
                    }
                })
            },
            &task,
            None::<Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>>,
        )
        .await;

        assert!(result.success);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn execute_with_retry_exhausts_and_returns_failure() {
        use crate::task::create_task;
        let mut task = create_task("t".to_string(), "d".to_string(), None, vec![]);
        task.max_retries = Some(1);
        task.retry_delay_ms = Some(1);

        let result = execute_with_retry(
            move || {
                Box::pin(
                    async move { Err(crate::error::AgentError::Other("always fails".to_string())) },
                )
            },
            &task,
            None::<Arc<dyn Fn(u32, u32, String, u64) + Send + Sync>>,
        )
        .await;

        assert!(!result.success);
        assert!(result.output.contains("always fails"));
    }
}
