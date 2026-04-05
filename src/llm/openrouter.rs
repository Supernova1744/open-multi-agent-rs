/// OpenAI-compatible adapter that works with OpenRouter, OpenAI, Ollama, vLLM, etc.
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{AgentError, Result};
use crate::types::{
    ContentBlock, LLMChatOptions, LLMMessage, LLMResponse, LLMToolDef, Role, TokenUsage,
    ToolUseBlock,
};

use super::LLMAdapter;

// ---------------------------------------------------------------------------
// OpenAI-compatible wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<OAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OAIToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OAIToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OAIToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OAIToolFunction,
}

#[derive(Debug, Serialize)]
struct OAIToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    id: String,
    model: Option<String>,
    choices: Vec<OAIChoice>,
    usage: Option<OAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OAIChoice {
    message: OAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAIUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

pub struct OpenRouterAdapter {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenRouterAdapter {
    pub fn new(api_key: String, base_url: String) -> Self {
        OpenRouterAdapter {
            api_key,
            base_url,
            client: reqwest::Client::new(),
        }
    }

    // Convert our LLMMessage format to OpenAI wire format.
    fn to_oai_messages(&self, messages: &[LLMMessage], system_prompt: Option<&str>) -> Vec<OAIMessage> {
        let mut result = Vec::new();

        // Inject system prompt as first message if provided.
        if let Some(sys) = system_prompt {
            if !sys.is_empty() {
                result.push(OAIMessage {
                    role: "system".to_string(),
                    content: Some(serde_json::Value::String(sys.to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
        }

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            // Separate text content from tool use and tool result blocks.
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<OAIToolCall> = Vec::new();
            let mut tool_results: Vec<(String, String, Option<bool>)> = Vec::new(); // (id, content, is_error)

            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        text_parts.push(text.clone());
                    }
                    ContentBlock::ToolUse(tu) => {
                        let args = serde_json::to_string(&tu.input).unwrap_or_default();
                        tool_calls.push(OAIToolCall {
                            id: tu.id.clone(),
                            tool_type: "function".to_string(),
                            function: OAIToolCallFunction {
                                name: tu.name.clone(),
                                arguments: args,
                            },
                        });
                    }
                    ContentBlock::ToolResult(tr) => {
                        tool_results.push((tr.tool_use_id.clone(), tr.content.clone(), tr.is_error));
                    }
                    ContentBlock::Image { .. } => {}
                }
            }

            // Tool results become separate "tool" role messages.
            for (tool_use_id, content, _is_error) in tool_results {
                result.push(OAIMessage {
                    role: "tool".to_string(),
                    content: Some(serde_json::Value::String(content)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id),
                    name: None,
                });
            }

            // Build the main message.
            if !tool_calls.is_empty() {
                // Assistant message with tool calls.
                let text_content = if text_parts.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::String(text_parts.join("")))
                };
                result.push(OAIMessage {
                    role: role.to_string(),
                    content: text_content,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                    name: None,
                });
            } else if !text_parts.is_empty() {
                result.push(OAIMessage {
                    role: role.to_string(),
                    content: Some(serde_json::Value::String(text_parts.join(""))),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
        }

        result
    }

    fn to_oai_tools(tools: &[LLMToolDef]) -> Vec<OAITool> {
        tools
            .iter()
            .map(|t| OAITool {
                tool_type: "function".to_string(),
                function: OAIToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect()
    }

    fn parse_response(&self, resp: ChatResponse, model: &str) -> Result<LLMResponse> {
        let choice = resp.choices.into_iter().next().ok_or_else(|| {
            AgentError::LlmError("Empty choices in response".to_string())
        })?;

        let finish_reason = choice.finish_reason.unwrap_or_else(|| "end_turn".to_string());
        let stop_reason = if finish_reason == "tool_calls" {
            "tool_use".to_string()
        } else {
            "end_turn".to_string()
        };

        let mut content: Vec<ContentBlock> = Vec::new();

        // Extract text content.
        if let Some(text_val) = &choice.message.content {
            let text = match text_val {
                serde_json::Value::String(s) => s.clone(),
                _ => text_val.to_string(),
            };
            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }
        }

        // Extract tool calls.
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                content.push(ContentBlock::ToolUse(ToolUseBlock {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                }));
            }
        }

        let usage = resp.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens.unwrap_or(0),
            output_tokens: u.completion_tokens.unwrap_or(0),
        }).unwrap_or_default();

        Ok(LLMResponse {
            id: resp.id,
            content,
            model: resp.model.unwrap_or_else(|| model.to_string()),
            stop_reason,
            usage,
        })
    }
}

#[async_trait]
impl LLMAdapter for OpenRouterAdapter {
    fn name(&self) -> &str {
        "openrouter"
    }

    async fn chat(&self, messages: &[LLMMessage], options: &LLMChatOptions) -> Result<LLMResponse> {
        let oai_messages = self.to_oai_messages(messages, options.system_prompt.as_deref());

        let oai_tools = options.tools.as_ref().map(|t| Self::to_oai_tools(t));

        let request = ChatRequest {
            model: options.model.clone(),
            messages: oai_messages,
            tools: if oai_tools.as_ref().map(|t| t.is_empty()).unwrap_or(true) {
                None
            } else {
                oai_tools
            },
            max_tokens: options.max_tokens,
            temperature: options.temperature,
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::LlmError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let chat_resp: ChatResponse = response.json().await?;
        self.parse_response(chat_resp, &options.model)
    }
}
