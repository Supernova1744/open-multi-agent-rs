/// Example 17 — RAG Knowledge Base Agent
///
/// Demonstrates the built-in knowledge base (RAG) tools:
///   • rag_add    — index a document into the in-process knowledge base
///   • rag_search — keyword-scored semantic search over stored documents
///   • rag_clear  — remove specific documents or clear the entire store
///
/// The agent:
///   1. Adds several documents about different programming topics
///   2. Answers questions by searching the knowledge base first
///   3. Demonstrates update and removal of documents
///
/// This simulates a lightweight RAG (Retrieval-Augmented Generation) pipeline
/// without requiring an external vector database.
///
/// Run:
///   cargo run --example 17_rag_knowledge_base
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry},
    types::AgentConfig,
    AgentRunResult,
};
use std::sync::Arc;
use tokio::sync::Mutex;

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

#[tokio::main]
async fn main() {
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    let config = AgentConfig {
        name: "rag-agent".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(
            "You are a knowledge management agent. Use rag_add to store documents, \
             rag_search to find relevant information before answering questions, \
             and rag_clear to manage the knowledge base. \
             Always search the knowledge base before answering factual questions."
            .to_string(),
        ),
        max_turns: Some(14),
        tools: Some(vec![
            "rag_add".to_string(),
            "rag_search".to_string(),
            "rag_clear".to_string(),
        ]),
        ..Default::default()
    };

    let mut agent = Agent::new(config, Arc::clone(&registry), Arc::clone(&executor));

    let task = "\
        Complete the following knowledge base tasks:\n\
        \n\
        1. ADD DOCUMENTS: Use rag_add to store these documents:\n\
           - id='rust-ownership', content='Rust ownership system ensures memory safety \
             without a garbage collector. Each value has a single owner. When the owner \
             goes out of scope the value is dropped. Borrowing allows references without \
             transferring ownership.'\n\
           - id='rust-async', content='Rust async/await enables non-blocking I/O. \
             Futures are lazy and only execute when polled. Tokio is the most popular \
             async runtime. The async keyword transforms a function into a Future.'\n\
           - id='python-gc', content='Python uses reference counting for memory management \
             with a cyclic garbage collector to handle reference cycles. Objects are \
             deallocated when their reference count drops to zero.'\n\
           - id='go-goroutines', content='Go goroutines are lightweight threads managed \
             by the Go runtime. Channels provide safe communication between goroutines. \
             The select statement waits on multiple channel operations.'\n\
        \n\
        2. SEARCH: Search for 'memory management garbage collector' and report the top 2 results.\n\
        \n\
        3. SEARCH: Search for 'async concurrency non-blocking' and report the top 2 results.\n\
        \n\
        4. UPDATE: Use rag_add with id='rust-ownership' to update the document \
           (add 'The borrow checker enforces these rules at compile time.' to the content).\n\
        \n\
        5. CLEAR ONE: Use rag_clear to remove document id='go-goroutines'.\n\
        \n\
        6. VERIFY: Search for 'goroutine channel' and confirm the Go document was removed.\n\
        \n\
        Summarise all findings.";

    println!("Task:\n{}\n{}", task, "─".repeat(60));

    match agent.run(task, adapter).await {
        Ok(AgentRunResult { output, turns, token_usage, tool_calls, .. }) => {
            println!("{}", "─".repeat(60));
            println!("Done in {} turns", turns);
            println!("Tokens — input: {}, output: {}", token_usage.input_tokens, token_usage.output_tokens);
            println!("Tool calls:");
            for tc in &tool_calls {
                println!("  • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
            println!("\nAgent response:\n{}", output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
