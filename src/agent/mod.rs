pub mod pool;
pub mod runner;

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::Result;
use crate::llm::LLMAdapter;
use crate::tool::{ToolExecutor, ToolRegistry};
use crate::types::{
    AgentConfig, AgentRunResult, AgentStatus, BeforeRunHookContext, ContentBlock, LLMMessage,
    Role, RunResult, StreamEvent, TokenUsage,
};

pub use runner::RunOptions;

use runner::AgentRunner;

/// High-level agent wrapper with state management and lifecycle hooks.
pub struct Agent {
    pub config: AgentConfig,
    pub status: AgentStatus,
    registry: Arc<Mutex<ToolRegistry>>,
    executor: Arc<ToolExecutor>,
    history: Vec<LLMMessage>,
    total_token_usage: TokenUsage,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        registry: Arc<Mutex<ToolRegistry>>,
        executor: Arc<ToolExecutor>,
    ) -> Self {
        Agent {
            config,
            status: AgentStatus::Idle,
            registry,
            executor,
            history: Vec::new(),
            total_token_usage: TokenUsage::default(),
        }
    }

    /// Run in a fresh conversation (no history).
    pub async fn run(
        &mut self,
        prompt: &str,
        adapter: Arc<dyn LLMAdapter>,
    ) -> Result<AgentRunResult> {
        let messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt.to_string() }],
        }];
        self.execute_run(messages, adapter, &RunOptions::default()).await
    }

    /// Run in a fresh conversation with explicit RunOptions (trace, callbacks).
    pub async fn run_with_opts(
        &mut self,
        prompt: &str,
        adapter: Arc<dyn LLMAdapter>,
        opts: RunOptions,
    ) -> Result<AgentRunResult> {
        let messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt.to_string() }],
        }];
        self.execute_run(messages, adapter, &opts).await
    }

    /// Run as a multi-turn conversation (persists history).
    pub async fn prompt(
        &mut self,
        message: &str,
        adapter: Arc<dyn LLMAdapter>,
    ) -> Result<AgentRunResult> {
        let user_msg = LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: message.to_string() }],
        };
        self.history.push(user_msg);

        let result = self
            .execute_run(self.history.clone(), adapter, &RunOptions::default())
            .await?;

        for msg in &result.messages {
            self.history.push(msg.clone());
        }
        Ok(result)
    }

    /// Stream a fresh-conversation response, yielding StreamEvents.
    pub fn stream<'a>(
        &'a mut self,
        prompt: &str,
        adapter: Arc<dyn LLMAdapter>,
    ) -> impl futures::Stream<Item = StreamEvent> + 'a {
        let messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt.to_string() }],
        }];
        let runner = AgentRunner {
            adapter,
            registry: Arc::clone(&self.registry),
            executor: Arc::clone(&self.executor),
            config: self.config.clone(),
        };
        async_stream::stream! {
            use futures::StreamExt;
            let opts = RunOptions::default();
            let s = runner.stream_run(messages, &opts);
            tokio::pin!(s);
            while let Some(event) = s.next().await {
                yield event;
            }
        }
    }

    /// Return a copy of the persistent conversation history.
    pub fn get_history(&self) -> Vec<LLMMessage> {
        self.history.clone()
    }

    /// Clear history and reset to idle.
    pub fn reset(&mut self) {
        self.history.clear();
        self.status = AgentStatus::Idle;
        self.total_token_usage = TokenUsage::default();
    }

    // -----------------------------------------------------------------------
    // Internal execution core
    // -----------------------------------------------------------------------

    async fn execute_run(
        &mut self,
        mut messages: Vec<LLMMessage>,
        adapter: Arc<dyn LLMAdapter>,
        opts: &RunOptions,
    ) -> Result<AgentRunResult> {
        self.status = AgentStatus::Running;

        // --- before_run hook ---
        if let Some(hook) = self.config.before_run.clone() {
            let prompt = extract_last_user_prompt(&messages);
            let ctx = BeforeRunHookContext {
                prompt: prompt.clone(),
                agent_name: self.config.name.clone(),
                agent_model: self.config.model.clone(),
            };
            match hook(ctx).await {
                Ok(modified_ctx) => {
                    if modified_ctx.prompt != prompt {
                        apply_modified_prompt(&mut messages, &modified_ctx.prompt);
                    }
                }
                Err(e) => {
                    self.status = AgentStatus::Error;
                    return Ok(make_error_result(format!("before_run hook failed: {}", e)));
                }
            }
        }

        let runner = AgentRunner {
            adapter,
            registry: Arc::clone(&self.registry),
            executor: Arc::clone(&self.executor),
            config: self.config.clone(),
        };

        let run_result = match runner.run(messages.clone(), opts).await {
            Ok(r) => r,
            Err(e) => {
                self.status = AgentStatus::Error;
                return Ok(make_error_result(e.to_string()));
            }
        };

        self.total_token_usage = self.total_token_usage.add(&run_result.token_usage);

        // --- Structured output validation ---
        let mut agent_result = if let Some(schema) = self.config.output_schema.clone() {
            self.validate_structured_output(run_result, &messages, &runner, opts, &schema)
                .await
        } else {
            to_agent_run_result(run_result, true, None)
        };

        // --- after_run hook ---
        if let Some(hook) = self.config.after_run.clone() {
            match hook(agent_result).await {
                Ok(modified) => agent_result = modified,
                Err(e) => {
                    self.status = AgentStatus::Error;
                    return Ok(make_error_result(format!("after_run hook failed: {}", e)));
                }
            }
        }

        self.status = if agent_result.success {
            AgentStatus::Completed
        } else {
            AgentStatus::Error
        };

        Ok(agent_result)
    }

    /// Validate output against schema. Retries once with error feedback on failure.
    async fn validate_structured_output(
        &self,
        result: RunResult,
        original_messages: &[LLMMessage],
        runner: &AgentRunner,
        opts: &RunOptions,
        schema: &serde_json::Value,
    ) -> AgentRunResult {
        // First attempt.
        match extract_and_validate_json(&result.output, schema) {
            Ok(parsed) => to_agent_run_result(result, true, Some(parsed)),
            Err(err_msg) => {
                // Retry with error feedback.
                let error_feedback = LLMMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: format!(
                            "Your previous response did not produce valid JSON matching the required schema.\n\nError: {}\n\nPlease try again. Respond with ONLY valid JSON, no other text.",
                            err_msg
                        ),
                    }],
                };
                let mut retry_messages = original_messages.to_vec();
                retry_messages.extend(result.messages.clone());
                retry_messages.push(error_feedback.clone());

                match runner.run(retry_messages, opts).await {
                    Ok(retry_result) => {
                        let merged_usage = result.token_usage.add(&retry_result.token_usage);
                        let mut merged_messages = result.messages.clone();
                        merged_messages.push(error_feedback);
                        merged_messages.extend(retry_result.messages.clone());
                        let mut merged_tool_calls = result.tool_calls.clone();
                        merged_tool_calls.extend(retry_result.tool_calls.clone());

                        let structured =
                            extract_and_validate_json(&retry_result.output, schema).ok();
                        let success = structured.is_some();
                        AgentRunResult {
                            success,
                            output: retry_result.output,
                            messages: merged_messages,
                            token_usage: merged_usage,
                            tool_calls: merged_tool_calls,
                            turns: result.turns + retry_result.turns,
                            structured,
                        }
                    }
                    Err(_) => to_agent_run_result(result, false, None),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_agent_run_result(
    r: RunResult,
    success: bool,
    structured: Option<serde_json::Value>,
) -> AgentRunResult {
    AgentRunResult {
        success,
        output: r.output,
        messages: r.messages,
        token_usage: r.token_usage,
        tool_calls: r.tool_calls,
        turns: r.turns,
        structured,
    }
}

fn make_error_result(msg: String) -> AgentRunResult {
    AgentRunResult {
        success: false,
        output: msg,
        messages: Vec::new(),
        token_usage: TokenUsage::default(),
        tool_calls: Vec::new(),
        turns: 0,
        structured: None,
    }
}

/// Extract the text of the last user message.
fn extract_last_user_prompt(messages: &[LLMMessage]) -> String {
    for msg in messages.iter().rev() {
        if msg.role == Role::User {
            return msg
                .content
                .iter()
                .filter_map(|b| b.as_text())
                .collect::<Vec<_>>()
                .join("");
        }
    }
    String::new()
}

/// Replace the text content of the last user message with a new prompt.
fn apply_modified_prompt(messages: &mut Vec<LLMMessage>, new_prompt: &str) {
    for msg in messages.iter_mut().rev() {
        if msg.role == Role::User {
            let non_text: Vec<ContentBlock> = msg
                .content
                .iter()
                .filter(|b| b.as_text().is_none())
                .cloned()
                .collect();
            msg.content = std::iter::once(ContentBlock::Text {
                text: new_prompt.to_string(),
            })
            .chain(non_text)
            .collect();
            return;
        }
    }
}

/// Attempt to extract JSON from raw text and validate it against a JSON schema.
pub fn extract_and_validate_json(
    raw: &str,
    schema: &serde_json::Value,
) -> std::result::Result<serde_json::Value, String> {
    let parsed = extract_json(raw)?;
    validate_against_schema(&parsed, schema)?;
    Ok(parsed)
}

/// Extract a JSON value from raw text using multiple fallback strategies.
pub fn extract_json(raw: &str) -> std::result::Result<serde_json::Value, String> {
    let trimmed = raw.trim();

    if let Ok(v) = serde_json::from_str(trimmed) {
        return Ok(v);
    }
    if let Some(cap) = trimmed.split("```json").nth(1) {
        if let Some(inner) = cap.split("```").next() {
            if let Ok(v) = serde_json::from_str(inner.trim()) {
                return Ok(v);
            }
        }
    }
    if let Some(cap) = trimmed.split("```").nth(1) {
        if let Ok(v) = serde_json::from_str(cap.trim()) {
            return Ok(v);
        }
    }
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if end > start {
            if let Ok(v) = serde_json::from_str(&trimmed[start..=end]) {
                return Ok(v);
            }
        }
    }
    if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
        if end > start {
            if let Ok(v) = serde_json::from_str(&trimmed[start..=end]) {
                return Ok(v);
            }
        }
    }
    Err(format!(
        "Failed to extract JSON from output. Raw output begins with: \"{}\"",
        &trimmed[..trimmed.len().min(100)]
    ))
}

/// Simple JSON Schema validation (type + required properties).
fn validate_against_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> std::result::Result<(), String> {
    if let Some(schema_type) = schema.get("type").and_then(|t| t.as_str()) {
        let actual_type = match value {
            serde_json::Value::Object(_) => "object",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::Bool(_) => "boolean",
            serde_json::Value::Null => "null",
        };
        if actual_type != schema_type {
            return Err(format!(
                "Type mismatch: expected '{}', got '{}'",
                schema_type, actual_type
            ));
        }
    }

    if let (Some(serde_json::Value::Array(required)), serde_json::Value::Object(obj_map)) =
        (schema.get("required"), value)
    {
        for req in required {
            if let Some(key) = req.as_str() {
                if !obj_map.contains_key(key) {
                    return Err(format!("Missing required property: '{}'", key));
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_direct() {
        let v = extract_json(r#"{"key": "value"}"#).unwrap();
        assert_eq!(v["key"], "value");
    }

    #[test]
    fn extract_json_fenced() {
        let v = extract_json("```json\n{\"key\": 42}\n```").unwrap();
        assert_eq!(v["key"], 42);
    }

    #[test]
    fn extract_json_bare_fence() {
        let v = extract_json("```\n{\"x\": true}\n```").unwrap();
        assert_eq!(v["x"], true);
    }

    #[test]
    fn extract_json_embedded_object() {
        let v = extract_json("Here is the result: {\"a\": 1} and more text").unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn extract_json_array() {
        let v = extract_json("[1, 2, 3]").unwrap();
        assert!(v.is_array());
    }

    #[test]
    fn extract_json_fails_gracefully() {
        assert!(extract_json("no json here at all").is_err());
    }

    #[test]
    fn validate_schema_type_check() {
        let schema = serde_json::json!({"type": "object"});
        assert!(validate_against_schema(&serde_json::json!({}), &schema).is_ok());
        assert!(validate_against_schema(&serde_json::json!("string"), &schema).is_err());
    }

    #[test]
    fn validate_schema_required_fields() {
        let schema = serde_json::json!({"type": "object", "required": ["name", "age"]});
        assert!(validate_against_schema(
            &serde_json::json!({"name": "Alice", "age": 30}),
            &schema
        )
        .is_ok());
        assert!(validate_against_schema(
            &serde_json::json!({"name": "Alice"}),
            &schema
        )
        .is_err());
    }

    #[test]
    fn extract_last_user_prompt_finds_last() {
        let messages = vec![
            LLMMessage {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "first".to_string() }],
            },
            LLMMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: "resp".to_string() }],
            },
            LLMMessage {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "second".to_string() }],
            },
        ];
        assert_eq!(extract_last_user_prompt(&messages), "second");
    }

    #[test]
    fn apply_modified_prompt_changes_last_user() {
        let mut messages = vec![LLMMessage {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "old".to_string() }],
        }];
        apply_modified_prompt(&mut messages, "new prompt");
        assert_eq!(extract_last_user_prompt(&messages), "new prompt");
    }
}
