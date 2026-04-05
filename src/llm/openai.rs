/// OpenAI API adapter.
///
/// Uses the standard OpenAI v1 API. Since OpenAI uses the same wire format
/// as OpenRouter, this is a thin wrapper around the shared OpenAI-compatible
/// implementation with the OpenAI base URL and auth format.
use async_trait::async_trait;

use crate::error::Result;
use crate::types::{LLMChatOptions, LLMMessage, LLMResponse};

use super::{LLMAdapter, openrouter::OpenRouterAdapter};

/// OpenAI API adapter (reuses OpenAI-compatible implementation).
pub struct OpenAIAdapter {
    inner: OpenRouterAdapter,
}

impl OpenAIAdapter {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        OpenAIAdapter {
            inner: OpenRouterAdapter::new(api_key, url),
        }
    }
}

#[async_trait]
impl LLMAdapter for OpenAIAdapter {
    fn name(&self) -> &str {
        "openai"
    }

    async fn chat(&self, messages: &[LLMMessage], options: &LLMChatOptions) -> Result<LLMResponse> {
        self.inner.chat(messages, options).await
    }
}
