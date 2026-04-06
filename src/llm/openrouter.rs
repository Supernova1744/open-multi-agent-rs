/// OpenAI-compatible adapter that works with OpenRouter, OpenAI, Ollama, vLLM, etc.
use async_trait::async_trait;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{AgentError, Result};
use crate::types::{
    ContentBlock, LLMChatOptions, LLMMessage, LLMResponse, LLMStreamDelta, LLMToolDef, Role,
    TokenUsage, ToolUseBlock,
};

use super::{LLMAdapter, LLMStream};

// ---------------------------------------------------------------------------
// OpenAI-compatible wire types (request)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<OAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// Ask the provider to include token-usage stats in the last SSE chunk.
    /// Required by OpenAI-compatible servers (including OpenRouter) to get
    /// non-zero token counts when streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

// ---------------------------------------------------------------------------
// SSE / streaming wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SSEChunk {
    id: Option<String>,
    #[serde(default)]
    choices: Vec<SSEChoice>,
    /// Some providers send final usage in the last chunk.
    usage: Option<OAIUsage>,
}

#[derive(Debug, Deserialize)]
struct SSEChoice {
    delta: SSEDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct SSEDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<SSEToolCall>>,
}

#[derive(Debug, Deserialize)]
struct SSEToolCall {
    index: usize,
    id: Option<String>,
    function: Option<SSEFunction>,
}

#[derive(Debug, Deserialize)]
struct SSEFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct AccToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    /// Reasoning / thinking output (Qwen3, DeepSeek-R1, o1, etc.).
    /// Used as fallback when `content` is null or empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
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
    /// Some providers only return total_tokens. Used as fallback.
    total_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Text extraction helper
// ---------------------------------------------------------------------------

/// Extract displayable text from an OAIMessage using a fallback chain:
///   1. `content` string (standard)
///   2. `content` array of {type,text} blocks (some providers)
///   3. `reasoning_content` (Qwen3, DeepSeek-R1, o1 thinking field)
///
/// Strips `<think>…</think>` wrappers that some models leave in content.
fn extract_text_from_message(msg: &OAIMessage) -> String {
    // Try content field first.
    if let Some(val) = &msg.content {
        let text = match val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(blocks) => {
                blocks
                    .iter()
                    .filter(|b| {
                        b.get("type").and_then(|t| t.as_str()) == Some("text")
                    })
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("")
            }
            _ => String::new(),
        };
        let cleaned = strip_think_tags(&text);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }

    // Fall back to reasoning_content (Qwen3, DeepSeek-R1, o1, etc.).
    if let Some(rc) = &msg.reasoning_content {
        let cleaned = strip_think_tags(rc);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }

    String::new()
}

/// Remove `<think>…</think>` wrapper blocks that reasoning models sometimes
/// include in their output. Keeps everything outside the tags.
fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    loop {
        let start = result.find("<think>");
        let end = result.find("</think>");
        match (start, end) {
            (Some(s), Some(e)) if e >= s => {
                result.replace_range(s..e + 8, ""); // 8 = len("</think>")
            }
            _ => break,
        }
    }
    result.trim().to_string()
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                    name: None,
                });
            } else if !text_parts.is_empty() {
                result.push(OAIMessage {
                    role: role.to_string(),
                    content: Some(serde_json::Value::String(text_parts.join(""))),
                    reasoning_content: None,
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

        // Extract text content using a fallback chain:
        //   1. content (string) — standard OpenAI format
        //   2. content (array) — some models return [{type,text}] blocks
        //   3. reasoning_content — Qwen3, DeepSeek-R1, o1 thinking output
        //
        // <think>…</think> wrappers are stripped before use.
        let extracted_text = extract_text_from_message(&choice.message);
        if !extracted_text.is_empty() {
            content.push(ContentBlock::Text { text: extracted_text });
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

        let usage = resp.usage.map(|u| {
            let input = u.prompt_tokens.unwrap_or(0);
            let output = u.completion_tokens.unwrap_or(0);
            // If both are 0 but total_tokens is present, put total in output
            // so token tracking is non-zero (providers differ on field names).
            if input == 0 && output == 0 {
                if let Some(total) = u.total_tokens {
                    return TokenUsage { input_tokens: 0, output_tokens: total };
                }
            }
            TokenUsage { input_tokens: input, output_tokens: output }
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
            stream: None,
            stream_options: None, // only set for streaming path
            tools: if oai_tools.as_ref().map(|t| t.is_empty()).unwrap_or(true) {
                None
            } else {
                oai_tools
            },
            max_tokens: options.max_tokens,
            temperature: options.temperature,
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        // Retry up to 5 times on HTTP 429 with exponential backoff.
        let response = {
            let mut attempts = 0u32;
            const MAX_RETRIES: u32 = 5;
            loop {
                let resp = self
                    .client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await?;

                if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && attempts < MAX_RETRIES {
                    attempts += 1;
                    let retry_after_ms = resp.headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                        .map(|s| s * 1000)
                        .unwrap_or(0);
                    let backoff_ms = 5000u64 * (1u64 << (attempts - 1));
                    let delay_ms = retry_after_ms.max(backoff_ms).min(90_000);
                    eprintln!("[openrouter] HTTP 429 — retrying in {}ms (attempt {}/{})", delay_ms, attempts, MAX_RETRIES);
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    continue;
                }
                break resp;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::LlmError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        // Read raw body so we can log it on parse failure or empty content.
        let raw_body = response.text().await.unwrap_or_default();

        let chat_resp: ChatResponse = match serde_json::from_str(&raw_body) {
            Ok(r) => r,
            Err(e) => {
                return Err(AgentError::LlmError(format!(
                    "Failed to parse response: {}\nBody: {}",
                    e,
                    &raw_body[..raw_body.len().min(500)]
                )));
            }
        };

        let result = self.parse_response(chat_resp, &options.model)?;

        // Warn when the model returned no usable text content.
        if result.content.is_empty() || result.content.iter().all(|b| b.as_text().map(|t| t.is_empty()).unwrap_or(true)) {
            eprintln!(
                "[openrouter] WARNING: empty content from model '{}'. Raw body snippet:\n{}",
                options.model,
                &raw_body[..raw_body.len().min(800)]
            );
        }

        Ok(result)
    }

    fn stream<'a>(&'a self, messages: &'a [LLMMessage], options: &'a LLMChatOptions) -> LLMStream<'a> {
        let messages = messages.to_vec();
        let options = options.clone();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();

        Box::pin(async_stream::try_stream! {
            let oai_messages = self.to_oai_messages(&messages, options.system_prompt.as_deref());
            let oai_tools = options.tools.as_ref().map(|t| Self::to_oai_tools(t));

            let request = ChatRequest {
                model: options.model.clone(),
                messages: oai_messages,
                stream: Some(true),
                stream_options: Some(StreamOptions { include_usage: true }),
                tools: if oai_tools.as_ref().map(|t| t.is_empty()).unwrap_or(true) { None } else { oai_tools },
                max_tokens: options.max_tokens,
                temperature: options.temperature,
            };

            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

            // Outer retry loop — retries the entire HTTP+SSE request when:
            //   (a) HTTP 429 is received (up to 5 times, exponential backoff)
            //   (b) HTTP 200 but empty SSE body (up to 3 times, linear delay)
            // Text events are only emitted on a successful non-empty response,
            // so retrying is always a clean restart from the caller's perspective.
            let mut http_429_attempts = 0u32;
            const MAX_429_RETRIES: u32 = 5;
            let mut empty_retries = 0u32;
            const MAX_EMPTY_RETRIES: u32 = 3;

            let (final_content, final_resp_id, final_finish_reason, final_total_input, final_total_output) = 'request: loop {

            // ── HTTP request with 429 backoff ─────────────────────────────
            let http_resp = loop {
                let resp = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| AgentError::from(e))?;

                if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && http_429_attempts < MAX_429_RETRIES {
                    http_429_attempts += 1;
                    let retry_after_ms = resp.headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                        .map(|s| s * 1000)
                        .unwrap_or(0);
                    let backoff_ms = 5000u64 * (1u64 << (http_429_attempts - 1)); // 5s,10s,20s,40s,80s
                    let delay_ms = retry_after_ms.max(backoff_ms).min(90_000);
                    eprintln!("[openrouter] HTTP 429 — retrying in {}ms (attempt {}/{})", delay_ms, http_429_attempts, MAX_429_RETRIES);
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    continue;
                }
                break resp;
            };

            if !http_resp.status().is_success() {
                let status = http_resp.status();
                let body = http_resp.text().await.unwrap_or_default();
                Err(AgentError::LlmError(format!("HTTP {}: {}", status, body)))?;
                return;
            }

            // ── SSE parsing ────────────────────────────────────────────────
            let mut byte_stream = http_resp.bytes_stream();
            let mut buf = String::new();

            // State for this attempt (reset on each 'request retry).
            let mut acc_tools: Vec<AccToolCall> = Vec::new();
            let mut finish_reason = String::from("end_turn");
            let mut resp_id = String::new();
            let mut total_input = 0u64;
            let mut total_output = 0u64;
            let mut full_text = String::new();
            // Fallback: reasoning_content (Qwen3, DeepSeek-R1, o1, etc.)
            let mut full_reasoning = String::new();

            'sse: while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| AgentError::from(e))?;
                buf.push_str(&String::from_utf8_lossy(&chunk));

                loop {
                    match buf.find('\n') {
                        None => break,
                        Some(nl) => {
                            let line = buf[..nl].trim().to_string();
                            buf = buf[nl + 1..].to_string();

                            if !line.starts_with("data:") { continue; }
                            let data = line["data:".len()..].trim();
                            if data == "[DONE]" { break 'sse; }

                            let sse: SSEChunk = match serde_json::from_str(data) {
                                Ok(c) => c,
                                Err(_) => continue,
                            };

                            if let Some(id) = sse.id { resp_id = id; }
                            if let Some(u) = sse.usage {
                                total_input = u.prompt_tokens.unwrap_or(total_input);
                                total_output = u.completion_tokens.unwrap_or(total_output);
                                if total_input == 0 && total_output == 0 {
                                    if let Some(t) = u.total_tokens { total_output = t; }
                                }
                            }

                            for choice in sse.choices {
                                if let Some(fr) = choice.finish_reason {
                                    if fr == "tool_calls" { finish_reason = "tool_use".to_string(); }
                                    else if !fr.is_empty() { finish_reason = "end_turn".to_string(); }
                                }

                                // Text delta: emit as live streaming events.
                                if let Some(content) = choice.delta.content.filter(|s| !s.is_empty()) {
                                    full_text.push_str(&content);
                                    yield LLMStreamDelta::Text(content);
                                }
                                // Reasoning content: accumulate silently as fallback.
                                if let Some(rc) = choice.delta.reasoning_content.filter(|s| !s.is_empty()) {
                                    full_reasoning.push_str(&rc);
                                }

                                // Tool call accumulation.
                                if let Some(tool_calls) = choice.delta.tool_calls {
                                    for tc in tool_calls {
                                        let idx = tc.index;
                                        while acc_tools.len() <= idx {
                                            acc_tools.push(AccToolCall::default());
                                        }
                                        let acc = &mut acc_tools[idx];
                                        if let Some(id) = tc.id { acc.id = id; }
                                        if let Some(f) = tc.function {
                                            if let Some(n) = f.name { acc.name = n; }
                                            if let Some(a) = f.arguments { acc.arguments.push_str(&a); }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ── Build LLMResponse content ──────────────────────────────────
            let mut content: Vec<ContentBlock> = Vec::new();

            // Choose text source: delta.content preferred, reasoning_content as fallback.
            // Strip <think>…</think> on the complete string (handles cross-chunk tags).
            let raw = if !full_text.is_empty() { &full_text } else { &full_reasoning };
            let clean = strip_think_tags(raw);
            let final_text = if !clean.is_empty() {
                clean
            } else if !raw.is_empty() {
                raw.trim().to_string()
            } else {
                String::new()
            };
            if !final_text.is_empty() {
                content.push(ContentBlock::Text { text: final_text });
            }
            for acc in acc_tools {
                if acc.name.is_empty() { continue; }
                let input: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&acc.arguments).unwrap_or_default();
                content.push(ContentBlock::ToolUse(ToolUseBlock { id: acc.id, name: acc.name, input }));
            }

            // Guard: empty response (soft rate limit or model glitch).
            // Retry up to MAX_EMPTY_RETRIES times with a short delay.
            // Safe to retry because no Text events were emitted for an empty response.
            if content.is_empty() && total_output == 0 {
                if empty_retries < MAX_EMPTY_RETRIES {
                    empty_retries += 1;
                    let delay_ms = 3000u64 * empty_retries as u64;
                    eprintln!("[openrouter] Empty response — retrying in {}ms (attempt {}/{})", delay_ms, empty_retries, MAX_EMPTY_RETRIES);
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    continue 'request;
                }
                Err(AgentError::LlmError(
                    "Empty response from model after retries (rate limited or model unavailable)".to_string()
                ))?;
                return;
            }

            break 'request (content, resp_id, finish_reason, total_input, total_output);
            }; // end 'request loop

            yield LLMStreamDelta::Complete(LLMResponse {
                id: final_resp_id,
                content: final_content,
                model: options.model.clone(),
                stop_reason: final_finish_reason,
                usage: TokenUsage { input_tokens: final_total_input, output_tokens: final_total_output },
            });
        })
    }
}
