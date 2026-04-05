# Getting Started

## Prerequisites

- Rust 1.70 or newer (`rustup update stable`)
- An API key for at least one supported LLM provider

## Installation

Clone the repository and build:

```bash
git clone https://github.com/your-org/open-multi-agent-rs
cd open-multi-agent-rs
cargo build
```

Run the full test suite (no API key required — all tests use an offline mock adapter):

```bash
cargo test
```

## Environment Setup

Create a `.env` file at the project root (a template is provided as `.env.copy`):

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

The library calls `dotenvy::dotenv().ok()` automatically — you only need to set the
key for the provider you intend to use.

> **Windows note:** If you set the variable with `set KEY="value"` (double-quoted
> in cmd), the quotes become part of the value. Use `set KEY=value` without quotes,
> or store the key in `.env` to avoid this issue entirely.

## Supported Providers

| Provider string | Env var | Default endpoint |
|-----------------|---------|-----------------|
| `"openrouter"` (default) | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `"anthropic"` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| `"openai"` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |

Any OpenAI-compatible endpoint can be used by passing a custom `base_url`.

## Running Examples

```bash
# Basic single agent
cargo run --example 01_single_agent

# Streaming (shows tokens in real-time)
cargo run --example 03_streaming

# Custom tool
cargo run --example 04_custom_tool

# All ten examples
for i in $(seq -w 1 10); do
  cargo run --example "${i}_*"
done
```

See [examples.md](examples.md) for a full annotated walkthrough.

## Quick Example

```rust
use open_multi_agent::{OrchestratorConfig, OpenMultiAgent, AgentConfig};

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
        default_model: "claude-opus-4-6".to_string(),
        default_provider: "anthropic".to_string(),
        default_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        ..Default::default()
    });

    let config = AgentConfig {
        name: "assistant".to_string(),
        system_prompt: Some("You are a helpful assistant.".to_string()),
        ..Default::default()
    };

    match orchestrator.run_agent(config, "What is 2 + 2?").await {
        Ok(result) => println!("{}", result.output),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Cargo Dependency

Add to your own project's `Cargo.toml` (once published):

```toml
[dependencies]
open-multi-agent-rs = "0.1"
```

Or as a path dependency while developing locally:

```toml
[dependencies]
open-multi-agent-rs = { path = "../open-multi-agent-rs" }
```
