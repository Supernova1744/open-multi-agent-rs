/// Anthropic Messages API adapter.
///
/// Uses the native Anthropic wire format (different from OpenAI-compatible):
/// - Endpoint: https://api.anthropic.com/v1/messages
/// - Auth: x-api-key header
/// - System prompt is a top-level field, not a message
/// - Tool use blocks are part of content array
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
// Anthropic wire types (request)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Anthropic wire types (response)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    stop_reason: Option<String>,
    content: Vec<AnthropicResponseBlock>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

pub struct AnthropicAdapter {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicAdapter {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        AnthropicAdapter {
            api_key,
            base_url: base_url
                .unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            client: reqwest::Client::new(),
        }
    }

    fn to_anthropic_messages(&self, messages: &[LLMMessage]) -> Vec<AnthropicMessage> {
        let mut result = Vec::new();

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let mut blocks: Vec<AnthropicContentBlock> = Vec::new();

            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        if !text.is_empty() {
                            blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                        }
                    }
                    ContentBlock::ToolUse(tu) => {
                        blocks.push(AnthropicContentBlock::ToolUse {
                            id: tu.id.clone(),
                            name: tu.name.clone(),
                            input: serde_json::to_value(&tu.input).unwrap_or(serde_json::Value::Object(Default::default())),
                        });
                    }
                    ContentBlock::ToolResult(tr) => {
                        blocks.push(AnthropicContentBlock::ToolResult {
                            tool_use_id: tr.tool_use_id.clone(),
                            content: tr.content.clone(),
                            is_error: tr.is_error,
                        });
                    }
                    ContentBlock::Image { .. } => {} // Images not yet supported
                }
            }

            if !blocks.is_empty() {
                result.push(AnthropicMessage {
                    role: role.to_string(),
                    content: blocks,
                });
            }
        }

        result
    }

    fn to_anthropic_tools(tools: &[LLMToolDef]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
            })
            .collect()
    }
}

#[async_trait]
impl LLMAdapter for AnthropicAdapter {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn chat(&self, messages: &[LLMMessage], options: &LLMChatOptions) -> Result<LLMResponse> {
        let anthropic_messages = self.to_anthropic_messages(messages);
        let anthropic_tools = options
            .tools
            .as_ref()
            .map(|t| Self::to_anthropic_tools(t))
            .filter(|t| !t.is_empty());

        let request = AnthropicRequest {
            model: options.model.clone(),
            max_tokens: options.max_tokens.unwrap_or(8192),
            messages: anthropic_messages,
            system: options.system_prompt.clone(),
            tools: anthropic_tools,
            temperature: options.temperature,
        };

        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::LlmError(format!("HTTP {}: {}", status, body)));
        }

        let resp: AnthropicResponse = response.json().await?;

        let stop_reason = resp
            .stop_reason
            .as_deref()
            .map(|r| if r == "tool_use" { "tool_use" } else { "end_turn" })
            .unwrap_or("end_turn")
            .to_string();

        let mut content: Vec<ContentBlock> = Vec::new();
        for block in resp.content {
            match block {
                AnthropicResponseBlock::Text { text } if !text.is_empty() => {
                    content.push(ContentBlock::Text { text });
                }
                AnthropicResponseBlock::ToolUse { id, name, input } => {
                    let input_map: HashMap<String, serde_json::Value> = match input {
                        serde_json::Value::Object(m) => m.into_iter().collect(),
                        _ => HashMap::new(),
                    };
                    content.push(ContentBlock::ToolUse(ToolUseBlock { id, name, input: input_map }));
                }
                _ => {}
            }
        }

        Ok(LLMResponse {
            id: resp.id,
            content,
            model: resp.model,
            stop_reason,
            usage: TokenUsage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests — wire format serialisation (no HTTP calls)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, LLMMessage, Role, ToolResultBlock, ToolUseBlock};
    use std::collections::HashMap;

    fn make_adapter() -> AnthropicAdapter {
        AnthropicAdapter::new("test-key".to_string(), None)
    }

    #[test]
    fn adapter_name_is_anthropic() {
        assert_eq!(make_adapter().name(), "anthropic");
    }

    #[test]
    fn default_base_url_is_anthropic() {
        let adapter = make_adapter();
        assert_eq!(adapter.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn custom_base_url_is_preserved() {
        let adapter = AnthropicAdapter::new("key".to_string(), Some("https://proxy.example.com".to_string()));
        assert_eq!(adapter.base_url, "https://proxy.example.com");
    }

    #[test]
    fn to_anthropic_messages_text_block() {
        let adapter = make_adapter();
        let messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Hello".to_string() }],
        }];
        let result = adapter.to_anthropic_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result.len(), 1);
        let block = &result[0].content[0];
        assert!(matches!(block, AnthropicContentBlock::Text { text } if text == "Hello"));
    }

    #[test]
    fn to_anthropic_messages_assistant_role() {
        let adapter = make_adapter();
        let messages = vec![LLMMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: "Hi".to_string() }],
        }];
        let result = adapter.to_anthropic_messages(&messages);
        assert_eq!(result[0].role, "assistant");
    }

    #[test]
    fn to_anthropic_messages_tool_use_block() {
        let adapter = make_adapter();
        let mut input = HashMap::new();
        input.insert("cmd".to_string(), serde_json::json!("ls"));
        let messages = vec![LLMMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse(ToolUseBlock {
                id: "tu-1".to_string(),
                name: "bash".to_string(),
                input,
            })],
        }];
        let result = adapter.to_anthropic_messages(&messages);
        assert_eq!(result.len(), 1);
        let block = &result[0].content[0];
        match block {
            AnthropicContentBlock::ToolUse { id, name, .. } => {
                assert_eq!(id, "tu-1");
                assert_eq!(name, "bash");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn to_anthropic_messages_tool_result_block() {
        let adapter = make_adapter();
        let messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResultBlock {
                tool_use_id: "tu-1".to_string(),
                content: "file contents".to_string(),
                is_error: None,
            })],
        }];
        let result = adapter.to_anthropic_messages(&messages);
        assert_eq!(result.len(), 1);
        let block = &result[0].content[0];
        match block {
            AnthropicContentBlock::ToolResult { tool_use_id, content, is_error } => {
                assert_eq!(tool_use_id, "tu-1");
                assert_eq!(content, "file contents");
                assert!(is_error.is_none());
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn to_anthropic_messages_skips_empty_text() {
        let adapter = make_adapter();
        let messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "".to_string() }],
        }];
        let result = adapter.to_anthropic_messages(&messages);
        // Empty text → empty content → message skipped
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn to_anthropic_messages_multi_block_message() {
        let adapter = make_adapter();
        let mut input = HashMap::new();
        input.insert("x".to_string(), serde_json::json!(1));
        let messages = vec![LLMMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text { text: "Thinking...".to_string() },
                ContentBlock::ToolUse(ToolUseBlock {
                    id: "c1".to_string(),
                    name: "calc".to_string(),
                    input,
                }),
            ],
        }];
        let result = adapter.to_anthropic_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content.len(), 2);
        assert!(matches!(result[0].content[0], AnthropicContentBlock::Text { .. }));
        assert!(matches!(result[0].content[1], AnthropicContentBlock::ToolUse { .. }));
    }

    #[test]
    fn to_anthropic_tools_converts_defs() {
        let tools = vec![
            crate::types::LLMToolDef {
                name: "bash".to_string(),
                description: "Run bash commands".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            },
        ];
        let result = AnthropicAdapter::to_anthropic_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "bash");
        assert_eq!(result[0].description, "Run bash commands");
        assert_eq!(result[0].input_schema, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn to_anthropic_tools_empty() {
        let result = AnthropicAdapter::to_anthropic_tools(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn request_serialises_to_expected_json() {
        // Verify the request struct serialises correctly for the wire
        let req = AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            max_tokens: 1024,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: vec![AnthropicContentBlock::Text { text: "hello".to_string() }],
            }],
            system: Some("You are helpful".to_string()),
            tools: None,
            temperature: Some(0.7),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-opus-4-6");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["system"], "You are helpful");
        assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-3);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"][0]["type"], "text");
        assert_eq!(json["messages"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn tool_use_block_serialises_with_correct_type_tag() {
        let block = AnthropicContentBlock::ToolUse {
            id: "id-1".to_string(),
            name: "my_tool".to_string(),
            input: serde_json::json!({"key": "val"}),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["id"], "id-1");
        assert_eq!(json["name"], "my_tool");
    }

    #[test]
    fn tool_result_block_serialises_with_correct_type_tag() {
        let block = AnthropicContentBlock::ToolResult {
            tool_use_id: "tu-1".to_string(),
            content: "result data".to_string(),
            is_error: Some(true),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "tu-1");
        assert_eq!(json["is_error"], true);
    }
}
