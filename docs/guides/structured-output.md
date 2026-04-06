# Guide: Structured Output

When `AgentConfig.output_schema` is set, the agent validates its final output against
the supplied JSON Schema. If validation fails, the agent is given one automatic retry
with a feedback message listing the validation errors.

## Setting up a schema

```rust
use open_multi_agent_rs::{OrchestratorConfig, OpenMultiAgent, AgentConfig};

let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
    default_provider: "openrouter".to_string(),
    default_api_key: std::env::var("OPENROUTER_API_KEY").ok(),
    ..Default::default()
});

let schema = serde_json::json!({
    "type": "object",
    "properties": {
        "name":    { "type": "string" },
        "age":     { "type": "integer", "minimum": 0 },
        "city":    { "type": "string" }
    },
    "required": ["name", "age", "city"]
});

let config = AgentConfig {
    name: "extractor".to_string(),
    system_prompt: Some(
        "Extract person information as JSON. Output only the JSON object, nothing else.".to_string()
    ),
    output_schema: Some(schema),
    ..Default::default()
};

let result = orchestrator
    .run_agent(config, "Alice is 30 years old and lives in London.")
    .await?;

if let Some(structured) = result.structured {
    println!("Name: {}", structured["name"]);
    println!("Age:  {}", structured["age"]);
    println!("City: {}", structured["city"]);
} else {
    // Schema validation failed even after retry; raw output is still available
    println!("Raw output: {}", result.output);
}
```

## How validation works

1. The agent runs normally and produces a final text response.
2. The library extracts the first JSON object or array found in the response — either
   bare JSON or JSON inside a fenced ` ```json … ``` ` block.
3. The extracted JSON is validated against `output_schema` using JSON Schema rules.
4. If valid, the parsed value is stored in `AgentRunResult.structured`.
5. If invalid, the agent receives a follow-up user message listing the errors and is
   asked to try again (one retry only).
6. If the second attempt also fails, `structured` is `None` and `result.output`
   contains the raw final text.

## Schema tips

**Always include a system prompt** that instructs the model to output only JSON.
Models are more reliable when instructed explicitly:

```
"Output only a JSON object matching this schema. No prose, no markdown."
```

**Use `"required"` for mandatory fields** — the validator checks that required
properties are present and have the right types.

**Keep schemas flat and simple.** Deeply nested schemas with `oneOf`/`anyOf`/`allOf`
increase the chance of model non-compliance.

## Supported JSON Schema keywords

The validator handles the most common keywords:

| Keyword | Supported |
|---------|-----------|
| `type` | string, number, integer, boolean, array, object, null |
| `properties` | yes |
| `required` | yes |
| `minimum` / `maximum` | yes (numbers) |
| `minLength` / `maxLength` | yes (strings) |
| `enum` | yes |
| `items` | yes (arrays) |
| `additionalProperties` | yes |
| `oneOf` / `anyOf` / `allOf` | no |

## Accessing structured output

```rust
let result = orchestrator.run_agent(config, prompt).await?;

match result.structured {
    Some(value) => {
        // value is serde_json::Value — index freely
        let name = value["name"].as_str().unwrap_or("unknown");
    }
    None => {
        // Validation failed; result.output has the raw text
        println!("Validation failed. Raw: {}", result.output);
    }
}
```

## Structured output vs. tool calling

Both patterns let you extract structured data. Choose based on your use case:

| | Structured Output | Tool Calling |
|---|---|---|
| Data flows | Model → structured response | Model → tool → model → response |
| Retries | 1 automatic retry | As many turns as needed |
| Schema lives | `AgentConfig.output_schema` | `Tool::input_schema` |
| Best for | Final extraction, single-pass | Multi-step workflows |
