/// Example 19 — LLM Knowledge Base Pipeline (Karpathy-style)
///
/// Implements the full pipeline described by Andrej Karpathy:
///   https://x.com/karpathy/status/2039805659525644595
///
/// Pipeline stages:
///
///   Stage 1 — INGEST
///     article_fetch  → clip web articles with YAML frontmatter to raw/
///     image_download → save referenced images locally
///
///   Stage 2 — COMPILE
///     file_list      → discover new raw/ files
///     file_read      → read each raw article
///     file_write     → compile into wiki/ as .md files with [[WikiLinks]]
///     frontmatter    → set tags, categories, backlink metadata
///
///   Stage 3 — INDEX & Q&A
///     rag_index_dir  → bulk-index the entire wiki/ into the RAG store
///     wikilink_index → build the [[WikiLink]] backlink graph
///     rag_search     → answer questions against the indexed wiki
///     tavily_search  → fill knowledge gaps with live web search
///
///   Stage 4 — HEALTH CHECK
///     wikilink_index (orphans) → find disconnected pages
///     grep           → find inconsistencies
///     frontmatter    → update metadata on affected pages
///
/// The whole pipeline runs in a local sandbox directory so nothing
/// touches your real filesystem.
///
/// Run:
///   cargo run --example 19_knowledge_base_pipeline
///
/// With Tavily search (optional):
///   TAVILY_API_KEY=tvly-... cargo run --example 19_knowledge_base_pipeline
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

// ── Stage helpers ─────────────────────────────────────────────────────────────

async fn run_stage(
    name: &str,
    task: &str,
    tools: Vec<&str>,
    max_turns: u32,
    registry: Arc<Mutex<ToolRegistry>>,
    executor: Arc<ToolExecutor>,
    adapter: Arc<dyn open_multi_agent_rs::LLMAdapter>,
) -> AgentRunResult {
    println!("\n{}", "═".repeat(60));
    println!("STAGE: {}", name);
    println!("{}", "═".repeat(60));

    let config = AgentConfig {
        name: format!("kb-{}", name.to_lowercase().replace(' ', "-")),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(format!(
            "You are an agent executing stage '{}' of an LLM knowledge-base pipeline. \
             Complete your assigned tasks precisely using the available tools. \
             Be systematic: list what you'll do, do it, then report what was accomplished.",
            name
        )),
        max_turns: Some(max_turns),
        tools: Some(tools.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    };

    let mut agent = Agent::new(config, registry, executor);
    match agent.run(task, adapter).await {
        Ok(result) => {
            println!("✓ Done in {} turns ({} tool calls)", result.turns, result.tool_calls.len());
            for tc in &result.tool_calls {
                println!("  • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
            result
        }
        Err(e) => {
            eprintln!("Stage error: {}", e);
            AgentRunResult::default()
        }
    }
}

#[tokio::main]
async fn main() {
    // ── Sandbox ───────────────────────────────────────────────────────────────
    let sandbox = std::env::temp_dir().join("karpathy_kb_demo");
    tokio::fs::create_dir_all(&sandbox).await.unwrap();
    tokio::fs::create_dir_all(sandbox.join("raw")).await.unwrap();
    tokio::fs::create_dir_all(sandbox.join("wiki")).await.unwrap();
    tokio::fs::create_dir_all(sandbox.join("raw/images")).await.unwrap();
    std::env::set_current_dir(&sandbox).unwrap();
    println!("Knowledge base root: {}", sandbox.display());

    // ── Shared tool registry ──────────────────────────────────────────────────
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    let adapter: Arc<dyn open_multi_agent_rs::LLMAdapter> = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    let has_tavily = std::env::var("TAVILY_API_KEY").is_ok();

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 1 — INGEST
    // Clip articles from the web into raw/, saving YAML frontmatter.
    // ─────────────────────────────────────────────────────────────────────────
    run_stage(
        "Ingest",
        "Clip the following articles into the raw/ directory using article_fetch:\n\
         1. article_fetch url='https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html' \
            save_path='raw/rust-ownership.md'\n\
         2. article_fetch url='https://doc.rust-lang.org/book/ch16-00-concurrency.html' \
            save_path='raw/rust-concurrency.md'\n\
         After clipping, use file_list on 'raw' and report the files with their sizes.",
        vec!["article_fetch", "file_list", "file_read"],
        6,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    ).await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 2 — COMPILE
    // LLM reads raw files and compiles wiki pages with [[WikiLinks]] and metadata.
    // ─────────────────────────────────────────────────────────────────────────
    run_stage(
        "Compile",
        "Read the raw articles and compile a wiki:\n\
         1. file_list path='raw' to see available articles.\n\
         2. For each article, read it with file_read.\n\
         3. For 'rust-ownership.md': write wiki/Ownership.md summarising \
            the article in 200 words. Include [[Lifetimes]], [[Borrowing]], \
            and [[Memory Safety]] as wikilinks. Then use frontmatter \
            (operation='write') to add fields: title, tags=['rust','memory'], category='Language Core'.\n\
         4. For 'rust-concurrency.md': write wiki/Concurrency.md with a 200-word \
            summary including [[Ownership]], [[Threads]], [[Channels]] as wikilinks. \
            Add frontmatter: title, tags=['rust','async','threads'], category='Language Core'.\n\
         5. file_list on 'wiki' to confirm both files were written.",
        vec!["file_list", "file_read", "file_write", "frontmatter"],
        12,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    ).await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 3 — INDEX & Q&A
    // Bulk-index wiki into RAG, build wikilink graph, answer a question.
    // ─────────────────────────────────────────────────────────────────────────
    let qa_result = run_stage(
        "Index & Q&A",
        &format!(
            "Index the wiki and answer a question:\n\
             1. rag_index_dir path='wiki' to index all wiki articles.\n\
             2. wikilink_index operation='build' path='wiki' to build the link graph.\n\
             3. wikilink_index operation='stats' to show graph statistics.\n\
             4. wikilink_index operation='backlinks' page='Ownership' to see \
                which pages link to Ownership.\n\
             5. rag_search query='memory safety ownership borrowing' top_k=2 \
                to find relevant wiki pages.\n\
             {}6. Based on the search results, write a concise answer to: \
                'How does Rust ensure memory safety without a garbage collector?'",
            if has_tavily {
                "Optional: use tavily_search to supplement with live information.\n"
            } else {
                ""
            }
        ),
        {
            let mut tools = vec!["rag_index_dir", "rag_search", "wikilink_index", "file_read"];
            if has_tavily { tools.push("tavily_search"); }
            tools
        },
        10,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    ).await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 4 — HEALTH CHECK
    // Find orphaned pages, inconsistencies, and update metadata.
    // ─────────────────────────────────────────────────────────────────────────
    run_stage(
        "Health Check",
        "Run health checks on the wiki:\n\
         1. wikilink_index operation='orphans' to find pages with no backlinks.\n\
         2. grep pattern='TODO\\|FIXME\\|XXX' path='wiki' recursive=true to find \
            any TODO markers.\n\
         3. For each wiki file, use frontmatter (operation='read') to inspect metadata.\n\
         4. frontmatter operation='set' on wiki/Ownership.md: \
            set key='last_checked', value=<today's date from datetime tool>.\n\
         5. Report: total wiki pages, total backlinks, any orphaned pages, \
            and any missing metadata.",
        vec!["wikilink_index", "grep", "frontmatter", "datetime", "file_list"],
        10,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    ).await;

    // ─────────────────────────────────────────────────────────────────────────
    // Final summary
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(60));
    println!("PIPELINE COMPLETE");
    println!("{}", "═".repeat(60));
    if !qa_result.output.is_empty() {
        println!("\nKnowledge base Q&A answer:\n{}", qa_result.output);
    }

    // Show what was built
    let wiki_files = std::fs::read_dir(sandbox.join("wiki"))
        .map(|entries| {
            entries.flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    println!("\nWiki files created ({}):", wiki_files.len());
    for f in &wiki_files {
        let path = sandbox.join("wiki").join(f);
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!("  • {} ({} bytes)", f, size);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    let _ = tokio::fs::remove_dir_all(&sandbox).await;
    println!("\nSandbox cleaned up.");
}
