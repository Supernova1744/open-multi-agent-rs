pub mod anthropic;
pub mod openai;
pub mod openrouter;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::error::Result;
use crate::types::{LLMChatOptions, LLMMessage, LLMResponse, LLMStreamDelta};

/// Convenience alias for a boxed, send-safe stream of stream deltas.
pub type LLMStream<'a> = Pin<Box<dyn Stream<Item = Result<LLMStreamDelta>> + Send + 'a>>;

/// Provider-agnostic interface every LLM backend must implement.
#[async_trait]
pub trait LLMAdapter: Send + Sync {
    fn name(&self) -> &str;

    /// Non-streaming: waits for the full response.
    async fn chat(&self, messages: &[LLMMessage], options: &LLMChatOptions) -> Result<LLMResponse>;

    /// Streaming: yields `Text` deltas as they arrive, then one `Complete`.
    ///
    /// Default implementation calls `chat()` and wraps the result — no real
    /// streaming, but a correct fallback. Override in adapters that support SSE.
    fn stream<'a>(&'a self, messages: &'a [LLMMessage], options: &'a LLMChatOptions) -> LLMStream<'a> {
        // Clone what we need to own inside the stream.
        let messages = messages.to_vec();
        let options = options.clone();
        Box::pin(async_stream::try_stream! {
            // Extract text before moving resp into Complete.
            let resp = self.chat(&messages, &options).await?;
            let text: String = resp.content.iter()
                .filter_map(|b| b.as_text())
                .collect();
            if !text.is_empty() {
                yield LLMStreamDelta::Text(text);
            }
            yield LLMStreamDelta::Complete(resp);
        })
    }
}

/// Create an adapter for the given provider.
///
/// Supported providers:
/// - `"anthropic"` — Anthropic Messages API
/// - `"openai"` — OpenAI Chat Completions API
/// - `"openrouter"` — OpenRouter (OpenAI-compatible, default)
/// - any other — treated as OpenRouter-compatible; set `base_url` for custom endpoints
pub fn create_adapter(
    provider: &str,
    api_key: Option<String>,
    base_url: Option<String>,
) -> Box<dyn LLMAdapter> {
    match provider {
        "anthropic" => {
            let key = api_key.unwrap_or_else(|| {
                std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()
            });
            Box::new(anthropic::AnthropicAdapter::new(key, base_url))
        }
        "openai" => {
            let key = api_key.unwrap_or_else(|| {
                std::env::var("OPENAI_API_KEY").unwrap_or_default()
            });
            Box::new(openai::OpenAIAdapter::new(key, base_url))
        }
        _ => {
            // Default: OpenRouter (or any OpenAI-compatible endpoint)
            let key = api_key.unwrap_or_else(|| {
                std::env::var("OPENROUTER_API_KEY").unwrap_or_default()
            });
            let url = base_url.unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());
            Box::new(openrouter::OpenRouterAdapter::new(key, url))
        }
    }
}
