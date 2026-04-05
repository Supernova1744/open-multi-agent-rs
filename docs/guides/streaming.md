# Guide: Streaming

`Agent::stream` yields tokens in real time as the model generates them. This is the
primary execution path inside the library — `Agent::run` collects the stream and waits
for `StreamEvent::Done`.

## Basic streaming

```rust
use futures::StreamExt;
use open_multi_agent::types::StreamEvent;

let mut stream = agent.stream("Explain async/await in Rust.", Arc::clone(&adapter));
tokio::pin!(stream);

while let Some(event) = stream.next().await {
    match event {
        StreamEvent::Text(chunk) => {
            print!("{}", chunk);
            // Flush to see tokens as they arrive
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        StreamEvent::ToolUse(block) => {
            eprintln!("\n[calling tool: {}]", block.name);
        }
        StreamEvent::ToolResult(block) => {
            eprintln!("[tool result: {} chars]", block.content.len());
        }
        StreamEvent::Done(result) => {
            println!("\n\nDone. {} turn(s), {}in/{}out tokens",
                result.turns,
                result.token_usage.input_tokens,
                result.token_usage.output_tokens,
            );
        }
        StreamEvent::Error(msg) => {
            eprintln!("\nError: {}", msg);
            break;
        }
    }
}
```

## `tokio::pin!` is required

`agent.stream()` returns `impl Stream<Item = StreamEvent> + '_` — an unboxed, named
future that borrows `agent`. You must pin it before calling `.next()`:

```rust
let stream = agent.stream(prompt, adapter);
tokio::pin!(stream);  // required
while let Some(event) = stream.next().await { ... }
```

Alternatively, box it to erase the lifetime:

```rust
use futures::StreamExt;
let mut stream = Box::pin(agent.stream(prompt, adapter));
while let Some(event) = stream.next().await { ... }
```

## Event ordering

For each LLM turn the stream emits events in this order:

```
StreamEvent::Text(chunk)        // zero or more, in real time
...
StreamEvent::ToolUse(block)     // one per tool call (after text, if any)
...
StreamEvent::ToolResult(block)  // one per tool call (after all are executed)
...
(repeat for the next turn)
StreamEvent::Done(result)       // exactly once, last event
```

If an unrecoverable error occurs, `StreamEvent::Error` is emitted and the stream ends.

## Collecting into a RunResult

If you need the complete result without processing individual events:

```rust
use futures::StreamExt;
use open_multi_agent::types::{StreamEvent, RunResult};

let mut result = None;
let mut stream = Box::pin(agent.stream(prompt, adapter));
while let Some(event) = stream.next().await {
    if let StreamEvent::Done(r) = event {
        result = Some(r);
    }
}
let result: RunResult = result.expect("stream ended without Done");
```

This is exactly what `Agent::run` does internally.

## How real-time streaming works

The `OpenRouterAdapter` implements `LLMAdapter::stream` using SSE (Server-Sent Events):
it sends `stream: true` in the request body and processes the `data: {...}` lines that
the server pushes in real time. Each line yields a `LLMStreamDelta::Text(chunk)`.

Other adapters (`AnthropicAdapter`, `OpenAIAdapter`) fall back to the default
implementation in `LLMAdapter`, which calls `chat()` and emits a single `Text` event
with the complete response. The streaming interface is identical from the caller's
perspective; only the granularity of chunks differs.

## Concurrent streams

Multiple agents can stream simultaneously:

```rust
let (s1, s2) = tokio::join!(
    collect_stream(agent_a.stream(prompt_a, adapter.clone())),
    collect_stream(agent_b.stream(prompt_b, adapter.clone())),
);
```
