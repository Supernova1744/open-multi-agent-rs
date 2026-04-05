/// Core agentic conversation loop.
///
/// Handles the while-loop: LLM call → extract tool_use → execute in parallel → append results → loop.
/// Supports both eager (`run`) and streaming (`stream`) execution.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use async_stream::stream;
use futures::Stream;

use crate::error::Result;
use crate::llm::LLMAdapter;
use crate::tool::{ToolExecutor, ToolRegistry};
use crate::trace::{emit_trace, now_ms};
use crate::types::{
    AgentConfig, AgentInfo, AgentTrace, ContentBlock, LlmCallTrace, LLMChatOptions, LLMMessage,
    OnTraceFn, Role, RunResult, StreamEvent, ToolCallRecord, ToolCallTrace, ToolResultBlock,
    ToolUseBlock, ToolUseContext, TokenUsage, TraceEventBase,
};

// ---------------------------------------------------------------------------
// RunOptions — per-call callbacks for observing execution
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct RunOptions {
    /// Fired just before each tool is dispatched.
    pub on_tool_call: Option<Arc<dyn Fn(&str, &HashMap<String, serde_json::Value>) + Send + Sync>>,
    /// Fired after each tool result is received.
    pub on_tool_result: Option<Arc<dyn Fn(&str, bool) + Send + Sync>>,
    /// Fired after each complete LLMMessage is appended.
    pub on_message: Option<Arc<dyn Fn(&LLMMessage) + Send + Sync>>,
    /// Trace callback for observability spans.
    pub on_trace: Option<OnTraceFn>,
    /// Run ID for trace correlation.
    pub run_id: Option<String>,
    /// Task ID for trace correlation.
    pub task_id: Option<String>,
    /// Agent name override for trace (defaults to config.name).
    pub trace_agent: Option<String>,
}

// ---------------------------------------------------------------------------
// AgentRunner
// ---------------------------------------------------------------------------

pub struct AgentRunner {
    pub adapter: Arc<dyn LLMAdapter>,
    pub registry: Arc<Mutex<ToolRegistry>>,
    pub executor: Arc<ToolExecutor>,
    pub config: AgentConfig,
}

impl AgentRunner {
    // -----------------------------------------------------------------------
    // Eager execution
    // -----------------------------------------------------------------------

    pub async fn run(
        &self,
        initial_messages: Vec<LLMMessage>,
        opts: &RunOptions,
    ) -> Result<RunResult> {
        let mut result = RunResult {
            messages: Vec::new(),
            output: String::new(),
            tool_calls: Vec::new(),
            token_usage: TokenUsage::default(),
            turns: 0,
        };

        // Collect from the stream.
        use futures::StreamExt;
        let s = self.stream_internal(initial_messages, opts);
        tokio::pin!(s);
        while let Some(event) = s.next().await {
            if let StreamEvent::Done(r) = event {
                result = r;
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Streaming execution
    // -----------------------------------------------------------------------

    pub fn stream_run<'a>(
        &'a self,
        initial_messages: Vec<LLMMessage>,
        opts: &'a RunOptions,
    ) -> impl Stream<Item = StreamEvent> + 'a {
        self.stream_internal(initial_messages, opts)
    }

    // -----------------------------------------------------------------------
    // Internal streaming implementation
    // -----------------------------------------------------------------------

    fn stream_internal<'a>(
        &'a self,
        initial_messages: Vec<LLMMessage>,
        opts: &'a RunOptions,
    ) -> impl Stream<Item = StreamEvent> + 'a {
        let adapter = Arc::clone(&self.adapter);
        let registry = Arc::clone(&self.registry);
        let executor = Arc::clone(&self.executor);
        let config = self.config.clone();
        let opts_on_trace = opts.on_trace.clone();
        let opts_run_id = opts.run_id.clone().unwrap_or_default();
        let opts_task_id = opts.task_id.clone();
        let opts_on_tool_call = opts.on_tool_call.clone();
        let opts_on_tool_result = opts.on_tool_result.clone();
        let opts_on_message = opts.on_message.clone();
        let trace_agent = opts.trace_agent.clone().unwrap_or_else(|| config.name.clone());

        stream! {
            let max_turns = config.max_turns.unwrap_or(10);
            let mut conversation = initial_messages.clone();
            let mut total_usage = TokenUsage::default();
            let mut all_tool_calls: Vec<ToolCallRecord> = Vec::new();
            let mut final_output = String::new();
            let mut turns = 0;

            // Build tool defs once.
            let tool_defs = {
                let reg = registry.lock().await;
                reg.to_tool_defs(config.tools.as_deref())
            };

            let chat_options = LLMChatOptions {
                model: config.model.clone(),
                tools: if tool_defs.is_empty() { None } else { Some(tool_defs) },
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                system_prompt: config.system_prompt.clone(),
            };

            'outer: loop {
                if turns >= max_turns {
                    break 'outer;
                }
                turns += 1;

                // Step 1: Call LLM.
                let llm_start = now_ms();
                let response = match adapter.chat(&conversation, &chat_options).await {
                    Ok(r) => r,
                    Err(e) => {
                        yield StreamEvent::Error(e.to_string());
                        return;
                    }
                };
                let llm_end = now_ms();

                // Emit llm_call trace.
                emit_trace(&opts_on_trace, crate::types::TraceEvent::LlmCall(LlmCallTrace {
                    base: TraceEventBase {
                        run_id: opts_run_id.clone(),
                        start_ms: llm_start,
                        end_ms: llm_end,
                        duration_ms: llm_end - llm_start,
                        agent: trace_agent.clone(),
                        task_id: opts_task_id.clone(),
                    },
                    model: config.model.clone(),
                    turn: turns,
                    tokens: response.usage.clone(),
                }));

                total_usage = total_usage.add(&response.usage);

                // Step 2: Build assistant message.
                let assistant_msg = LLMMessage {
                    role: Role::Assistant,
                    content: response.content.clone(),
                };
                conversation.push(assistant_msg.clone());
                if let Some(ref cb) = opts_on_message {
                    cb(&assistant_msg);
                }

                // Extract text and yield it.
                let text: String = response.content.iter()
                    .filter_map(|b| b.as_text())
                    .collect::<Vec<_>>()
                    .join("");

                if !text.is_empty() {
                    yield StreamEvent::Text(text.clone());
                }

                // Extract tool_use blocks and yield each.
                let tool_use_blocks: Vec<ToolUseBlock> = response.content.iter()
                    .filter_map(|b| b.as_tool_use())
                    .cloned()
                    .collect();

                for block in &tool_use_blocks {
                    yield StreamEvent::ToolUse(block.clone());
                }

                // Step 3: If no tool calls, done.
                if tool_use_blocks.is_empty() {
                    final_output = text;
                    break 'outer;
                }

                // Step 4: Execute all tool calls in parallel.
                let context = ToolUseContext {
                    agent: AgentInfo {
                        name: config.name.clone(),
                        role: config.system_prompt.as_deref().unwrap_or("assistant").to_string(),
                        model: config.model.clone(),
                    },
                    cwd: None,
                };
                let ctx_arc = Arc::new(context);

                let tool_futures: Vec<_> = tool_use_blocks
                    .iter()
                    .map(|block| {
                        let executor = Arc::clone(&executor);
                        let ctx = Arc::clone(&ctx_arc);
                        let name = block.name.clone();
                        let input = block.input.clone();
                        let id = block.id.clone();
                        let on_call = opts_on_tool_call.clone();
                        async move {
                            if let Some(ref cb) = on_call {
                                cb(&name, &input);
                            }
                            let start = Instant::now();
                            let result = executor.execute(&name, &input, &ctx).await;
                            let duration_ms = start.elapsed().as_millis() as u64;
                            (id, name, input, result, duration_ms)
                        }
                    })
                    .collect();

                let executions = futures::future::join_all(tool_futures).await;

                // Step 5: Collect results.
                let mut result_blocks: Vec<ContentBlock> = Vec::new();
                for (id, tool_name, input, tool_result, duration_ms) in &executions {
                    let is_err = tool_result.is_error;
                    if let Some(ref cb) = opts_on_tool_result {
                        cb(tool_name, is_err);
                    }

                    // Emit tool_call trace.
                    let t_start = now_ms() - duration_ms;
                    let t_end = t_start + duration_ms;
                    emit_trace(&opts_on_trace, crate::types::TraceEvent::ToolCall(ToolCallTrace {
                        base: TraceEventBase {
                            run_id: opts_run_id.clone(),
                            start_ms: t_start,
                            end_ms: t_end,
                            duration_ms: *duration_ms,
                            agent: trace_agent.clone(),
                            task_id: opts_task_id.clone(),
                        },
                        tool: tool_name.clone(),
                        is_error: is_err,
                    }));

                    all_tool_calls.push(ToolCallRecord {
                        tool_name: tool_name.clone(),
                        input: input.clone(),
                        output: tool_result.data.clone(),
                        duration_ms: *duration_ms,
                    });

                    let result_block = ToolResultBlock {
                        tool_use_id: id.clone(),
                        content: tool_result.data.clone(),
                        is_error: if is_err { Some(true) } else { None },
                    };
                    yield StreamEvent::ToolResult(result_block.clone());
                    result_blocks.push(ContentBlock::ToolResult(result_block));
                }

                let tool_result_msg = LLMMessage {
                    role: Role::User,
                    content: result_blocks,
                };
                if let Some(ref cb) = opts_on_message {
                    cb(&tool_result_msg);
                }
                conversation.push(tool_result_msg);
            }

            // If we hit maxTurns without a clean end_turn, grab last assistant text.
            if final_output.is_empty() {
                if let Some(last_assistant) = conversation.iter().rev().find(|m| m.role == Role::Assistant) {
                    final_output = last_assistant.content.iter()
                        .filter_map(|b| b.as_text())
                        .collect::<Vec<_>>()
                        .join("");
                }
            }

            // Emit agent trace.
            let run_result = RunResult {
                messages: conversation[initial_messages.len()..].to_vec(),
                output: final_output,
                tool_calls: all_tool_calls,
                token_usage: total_usage,
                turns,
            };

            emit_trace(&opts_on_trace, crate::types::TraceEvent::Agent(AgentTrace {
                base: TraceEventBase {
                    run_id: opts_run_id.clone(),
                    start_ms: 0,
                    end_ms: 0,
                    duration_ms: 0,
                    agent: trace_agent.clone(),
                    task_id: opts_task_id.clone(),
                },
                turns: run_result.turns,
                tokens: run_result.token_usage.clone(),
                tool_calls: run_result.tool_calls.len(),
            }));

            yield StreamEvent::Done(run_result);
        }
    }
}
