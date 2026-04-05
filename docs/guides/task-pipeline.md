# Guide: Task Pipelines

A task pipeline is a directed acyclic graph of work items where each task can declare
other tasks as dependencies. The orchestrator resolves the graph, executes tasks
concurrently where possible, and routes results through shared memory so downstream
agents can build on earlier work.

## Creating tasks

```rust
use open_multi_agent::create_task;

let research = create_task(
    "Research",
    "Research the history of the Rust programming language.",
    Some("researcher".to_string()),  // agent name
    vec![],                          // no dependencies
);
let research_id = research.id.clone();

let analysis = create_task(
    "Analysis",
    "Analyse the research findings and identify key milestones.",
    Some("analyst".to_string()),
    vec![research_id.clone()],       // depends on research
);
let analysis_id = analysis.id.clone();

let report = create_task(
    "Report",
    "Write a 3-paragraph summary report based on the analysis.",
    Some("writer".to_string()),
    vec![analysis_id],               // depends on analysis
);
```

## Running the pipeline

```rust
use open_multi_agent::{OrchestratorConfig, OpenMultiAgent, TeamConfig, AgentConfig};

let orchestrator = OpenMultiAgent::new(OrchestratorConfig {
    default_provider: "openrouter".to_string(),
    default_api_key: std::env::var("OPENROUTER_API_KEY").ok(),
    max_concurrency: 3,
    ..Default::default()
});

let team = TeamConfig {
    name: "pipeline-team".to_string(),
    agents: vec![
        AgentConfig { name: "researcher".to_string(), ..Default::default() },
        AgentConfig { name: "analyst".to_string(), ..Default::default() },
        AgentConfig { name: "writer".to_string(), ..Default::default() },
    ],
    shared_memory: Some(true),   // agents can read each other's results
    max_concurrency: Some(3),
};

let result = orchestrator
    .run_tasks(&team, vec![research, analysis, report])
    .await?;

println!("Success: {}", result.success);
for (task_id, agent_result) in &result.agent_results {
    println!("\n--- {} ---\n{}", &task_id[..8], agent_result.output);
}
```

## Shared memory

When `shared_memory: Some(true)`, each agent's output is written to a shared store
keyed as `"{agent_name}/{task_id}"`. Downstream agents automatically receive prior
results injected into their prompt.

You can also use `SharedMemory` directly:

```rust
use std::sync::Arc;
use open_multi_agent::memory::{InMemoryStore, SharedMemory};

let store = Arc::new(InMemoryStore::new());
let memory = SharedMemory::new(Arc::clone(&store) as Arc<_>);

// Write
memory.write("researcher", "findings", "Rust started in 2006...").await;

// Read from another agent's namespace
let entry = memory.read("researcher", "findings").await;

// Format all entries as markdown (for injection into a prompt)
let context = memory.to_markdown().await;
// ## researcher/findings
// Rust started in 2006...
```

## Dependency resolution

`topological_sort` validates and orders the graph:

```rust
use open_multi_agent::topological_sort;

let ordered = topological_sort(&tasks)?;
// ordered is a Vec<Task> in safe execution order
// Returns Err("cycle detected") if a circular dependency exists
```

The orchestrator calls this automatically before starting.

## Per-task configuration

```rust
let task = Task {
    max_retries: Some(3),
    retry_delay_ms: Some(500),
    retry_backoff: Some(2.0),
    assignee: Some("worker-a".to_string()),
    ..create_task("My Task", "Do something", None, vec![])
};
```

## `TaskQueue` state machine

The queue manages all state transitions automatically:

```
Blocked → Pending  (when all dependencies complete)
Pending → InProgress
InProgress → Completed  (success)
InProgress → Failed     (error)
Failed → Failed cascade  (downstream tasks also fail)
any non-terminal → Skipped  (via skip_remaining or approval gate rejection)
```

## Parallel execution

Tasks with no unresolved dependencies run concurrently, up to `max_concurrency`. With
a diamond-shaped graph like:

```
A → B → D
A → C → D
```

After A completes, B and C run in parallel, then D runs once both finish.

## `TeamRunResult`

```rust
pub struct TeamRunResult {
    pub success: bool,
    pub agent_results: HashMap<String, AgentRunResult>,  // keyed by task ID
    pub token_usage: TokenUsage,                         // total across all agents
}
```

`success` is `true` only if every non-skipped task completed successfully.
