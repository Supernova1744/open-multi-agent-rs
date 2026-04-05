pub mod anthropic;
pub mod openai;
pub mod openrouter;

use async_trait::async_trait;
use crate::error::Result;
use crate::types::{LLMChatOptions, LLMMessage, LLMResponse};

/// Provider-agnostic interface every LLM backend must implement.
#[async_trait]
pub trait LLMAdapter: Send + Sync {
    fn name(&self) -> &str;
    async fn chat(&self, messages: &[LLMMessage], options: &LLMChatOptions) -> Result<LLMResponse>;
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
