/// Shared mock LLM adapter for integration and stress tests.
///
/// Responses are fed from a pre-defined queue, making tests deterministic
/// without any real network calls.
use async_trait::async_trait;
use open_multi_agent::llm::LLMAdapter;
use open_multi_agent::types::{
    ContentBlock, LLMChatOptions, LLMMessage, LLMResponse, TokenUsage, ToolUseBlock,
};
use open_multi_agent::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// MockResponse variants
// ---------------------------------------------------------------------------

pub enum MockResponse {
    /// Plain text response (end_turn).
    Text(String),
    /// Response with custom token counts.
    TextWithUsage(String, u64, u64),
    /// Single tool_use block (model requests one tool).
    ToolCall {
        tool_name: String,
        tool_input: HashMap<String, serde_json::Value>,
        tool_id: String,
    },
    /// Multiple tool_use blocks in one response (parallel tool calls).
    MultiToolCall(Vec<(String, String, HashMap<String, serde_json::Value>)>), // (name, id, input)
    /// Text + tool calls together (e.g. reasoning text then tools).
    TextAndTools {
        text: String,
        tools: Vec<(String, String, HashMap<String, serde_json::Value>)>,
    },
    /// Simulate an LLM API error.
    Error(String),
}

// ---------------------------------------------------------------------------
// Recorded call — what the adapter saw
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub messages_count: usize,
    pub model: String,
    pub system_prompt: Option<String>,
    pub tools_count: usize,
}

// ---------------------------------------------------------------------------
// MockAdapter
// ---------------------------------------------------------------------------

pub struct MockAdapter {
    responses: Arc<Mutex<Vec<MockResponse>>>,
    /// All calls recorded for verification.
    pub recorded_calls: Arc<Mutex<Vec<RecordedCall>>>,
    /// Simple call counter.
    pub call_count: Arc<Mutex<usize>>,
}

impl MockAdapter {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        MockAdapter {
            responses: Arc::new(Mutex::new(responses)),
            recorded_calls: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Build already cast to `Arc<dyn LLMAdapter>`.
    pub fn arc(responses: Vec<MockResponse>) -> Arc<dyn LLMAdapter> {
        Arc::new(Self::new(responses))
    }

    // -----------------------------------------------------------------------
    // Convenience constructors
    // -----------------------------------------------------------------------

    pub fn text(s: impl Into<String>) -> MockResponse {
        MockResponse::Text(s.into())
    }

    pub fn text_with_usage(s: impl Into<String>, input: u64, output: u64) -> MockResponse {
        MockResponse::TextWithUsage(s.into(), input, output)
    }

    pub fn tool_call(
        name: &str,
        id: &str,
        input: HashMap<String, serde_json::Value>,
    ) -> MockResponse {
        MockResponse::ToolCall {
            tool_name: name.to_string(),
            tool_input: input,
            tool_id: id.to_string(),
        }
    }

    pub fn multi_tool_call(
        calls: Vec<(&str, &str, HashMap<String, serde_json::Value>)>,
    ) -> MockResponse {
        MockResponse::MultiToolCall(
            calls
                .into_iter()
                .map(|(n, id, inp)| (n.to_string(), id.to_string(), inp))
                .collect(),
        )
    }

    pub fn error(msg: impl Into<String>) -> MockResponse {
        MockResponse::Error(msg.into())
    }

    /// Total call count snapshot.
    pub fn calls(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait]
impl LLMAdapter for MockAdapter {
    fn name(&self) -> &str {
        "mock"
    }

    async fn chat(
        &self,
        messages: &[LLMMessage],
        options: &LLMChatOptions,
    ) -> Result<LLMResponse> {
        // Record the call.
        {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
        }
        {
            let mut calls = self.recorded_calls.lock().unwrap();
            calls.push(RecordedCall {
                messages_count: messages.len(),
                model: options.model.clone(),
                system_prompt: options.system_prompt.clone(),
                tools_count: options.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            });
        }

        let response = {
            let mut queue = self.responses.lock().unwrap();
            if queue.is_empty() {
                MockResponse::Text("(mock: no more responses)".to_string())
            } else {
                queue.remove(0)
            }
        };

        match response {
            MockResponse::Text(t) => Ok(LLMResponse {
                id: "mock-id".to_string(),
                content: vec![ContentBlock::Text { text: t }],
                model: options.model.clone(),
                stop_reason: "end_turn".to_string(),
                usage: TokenUsage { input_tokens: 10, output_tokens: 20 },
            }),
            MockResponse::TextWithUsage(t, input, output) => Ok(LLMResponse {
                id: "mock-id".to_string(),
                content: vec![ContentBlock::Text { text: t }],
                model: options.model.clone(),
                stop_reason: "end_turn".to_string(),
                usage: TokenUsage { input_tokens: input, output_tokens: output },
            }),
            MockResponse::ToolCall { tool_name, tool_input, tool_id } => Ok(LLMResponse {
                id: "mock-id".to_string(),
                content: vec![ContentBlock::ToolUse(ToolUseBlock {
                    id: tool_id,
                    name: tool_name,
                    input: tool_input,
                })],
                model: options.model.clone(),
                stop_reason: "tool_use".to_string(),
                usage: TokenUsage { input_tokens: 10, output_tokens: 20 },
            }),
            MockResponse::MultiToolCall(calls) => {
                let content = calls
                    .into_iter()
                    .map(|(name, id, input)| {
                        ContentBlock::ToolUse(ToolUseBlock { id, name, input })
                    })
                    .collect();
                Ok(LLMResponse {
                    id: "mock-id".to_string(),
                    content,
                    model: options.model.clone(),
                    stop_reason: "tool_use".to_string(),
                    usage: TokenUsage { input_tokens: 10, output_tokens: 20 },
                })
            }
            MockResponse::TextAndTools { text, tools } => {
                let mut content = vec![ContentBlock::Text { text }];
                for (name, id, input) in tools {
                    content.push(ContentBlock::ToolUse(ToolUseBlock { id, name, input }));
                }
                Ok(LLMResponse {
                    id: "mock-id".to_string(),
                    content,
                    model: options.model.clone(),
                    stop_reason: "tool_use".to_string(),
                    usage: TokenUsage { input_tokens: 10, output_tokens: 20 },
                })
            }
            MockResponse::Error(msg) => Err(open_multi_agent::error::AgentError::LlmError(msg)),
        }
    }
}
