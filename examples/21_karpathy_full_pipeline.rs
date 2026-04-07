/// Example 21 — Full Karpathy LLM Knowledge-Base Pipeline
///
/// A complete, end-to-end implementation of the knowledge-base pipeline
/// described by Andrej Karpathy:
///   https://x.com/karpathy/status/2039805659525644595
///
/// This example uses 5 interrelated AI/ML Wikipedia articles to build a rich
/// local knowledge base from scratch. Each stage feeds the next:
///
///   Stage 1 — INGEST
///     article_fetch  → clip 5 Wikipedia AI articles to raw/
///     image_download → save one diagram image locally
///     file_list      → confirm all raw files arrived
///
///   Stage 2 — COMPILE
///     file_read      → read each raw article
///     file_write     → write wiki/<Page>.md with 200-word summary
///     frontmatter    → stamp title, tags, category, related pages
///     file_write     → write wiki/README.md index page
///
///   Stage 3 — INDEX & LINK GRAPH
///     rag_index_dir  → bulk-index all wiki pages into RAG
///     wikilink_index (build)   → build the [[WikiLink]] graph
///     wikilink_index (stats)   → print graph statistics
///     wikilink_index (orphans) → find pages with no backlinks
///
///   Stage 4 — STUB GENERATION
///     wikilink_index (links)   → list outgoing links per page
///     file_list                → find which linked pages don't exist yet
///     file_write               → create brief stub pages for missing links
///     wikilink_index (build)   → rebuild the graph with stubs included
///
///   Stage 5 — Q&A (multi-hop)
///     rag_search     → find relevant pages for each question
///     file_read      → read the top-ranked pages in full
///     (answer)       → synthesise a detailed answer
///
///   Stage 6 — HEALTH CHECK & REPORT
///     grep           → scan for TODO / FIXME markers
///     frontmatter    → update last_checked on all pages
///     datetime       → get today's date for the timestamp
///     file_list      → final file inventory
///     (report)       → print KB statistics summary
///
/// All work happens in a temporary sandbox directory that is cleaned up on exit.
///
/// Run:
///   cargo run --example 21_karpathy_full_pipeline
///
/// With Tavily live-search supplement (optional):
///   TAVILY_API_KEY=tvly-... cargo run --example 21_karpathy_full_pipeline
use open_multi_agent_rs::{
    agent::Agent,
    create_adapter,
    tool::{built_in::register_built_in_tools, ToolExecutor, ToolRegistry},
    types::AgentConfig,
    AgentRunResult,
};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn api_key() -> String {
    dotenvy::dotenv().ok();
    std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set — add it to .env or export it")
        .trim_matches('"')
        .to_string()
}

struct Stage {
    name: &'static str,
    emoji: &'static str,
}

async fn run_stage(
    stage: &Stage,
    task: &str,
    tools: &[&str],
    max_turns: usize,
    registry: Arc<Mutex<ToolRegistry>>,
    executor: Arc<ToolExecutor>,
    adapter: Arc<dyn open_multi_agent_rs::LLMAdapter>,
) -> AgentRunResult {
    println!("\n{}", "═".repeat(70));
    println!("{} STAGE: {}", stage.emoji, stage.name);
    println!("{}", "═".repeat(70));

    let config = AgentConfig {
        name: format!(
            "kb-{}",
            stage.name.to_lowercase().replace([' ', '/'], "-")
        ),
        model: "qwen/qwen3-32b:nitro".to_string(),
        system_prompt: Some(format!(
            "You are an agent executing stage '{}' of a multi-stage LLM \
             knowledge-base pipeline. Work systematically: announce what you \
             are about to do, call the tools, then report what was accomplished. \
             Be thorough and do not skip steps.",
            stage.name
        )),
        max_turns: Some(max_turns),
        tools: Some(tools.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    };

    let mut agent = Agent::new(config, registry, executor);
    match agent.run(task, adapter).await {
        Ok(result) => {
            println!(
                "\n✓ {} complete — {} turns, {} tool calls",
                stage.name,
                result.turns,
                result.tool_calls.len()
            );
            for tc in &result.tool_calls {
                println!("    • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
            result
        }
        Err(e) => {
            eprintln!("Stage error: {}", e);
            AgentRunResult {
                success: false,
                output: String::new(),
                messages: vec![],
                token_usage: open_multi_agent_rs::TokenUsage::default(),
                tool_calls: vec![],
                turns: 0,
                structured: None,
            }
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Load .env and resolve API key BEFORE changing the working directory.
    let key = api_key();
    let has_tavily = std::env::var("TAVILY_API_KEY").is_ok();

    // ── Sandbox ───────────────────────────────────────────────────────────────
    let sandbox = std::env::temp_dir().join("karpathy_full_kb");
    for dir in &["raw", "raw/images", "wiki", "wiki/stubs"] {
        tokio::fs::create_dir_all(sandbox.join(dir)).await.unwrap();
    }
    std::env::set_current_dir(&sandbox).unwrap();
    println!("Knowledge base root: {}", sandbox.display());

    // ── Shared infrastructure ─────────────────────────────────────────────────
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));
    let adapter: Arc<dyn open_multi_agent_rs::LLMAdapter> = Arc::from(create_adapter(
        "openrouter",
        Some(key),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    // ── Articles to ingest (interrelated AI/ML Wikipedia pages) ──────────────
    let articles = [
        ("https://en.wikipedia.org/wiki/Large_language_model",              "raw/large-language-model.md"),
        ("https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)", "raw/transformer.md"),
        ("https://en.wikipedia.org/wiki/Retrieval-augmented_generation",    "raw/retrieval-augmented-generation.md"),
        ("https://en.wikipedia.org/wiki/Prompt_engineering",                "raw/prompt-engineering.md"),
        ("https://en.wikipedia.org/wiki/Attention_(machine_learning)",      "raw/attention-mechanism.md"),
    ];

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 1 — INGEST
    // Clip five Wikipedia AI/ML articles into raw/, plus one diagram image.
    // ─────────────────────────────────────────────────────────────────────────
    let ingest_task = format!(
        "Clip the following Wikipedia articles into the raw/ directory using article_fetch. \
         Call article_fetch once for each URL:\n\
         {}\n\
         After all five fetches, use image_download to save the Wikipedia logo image:\n\
         image_download url='https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png' \
         save_path='raw/images/wikipedia-logo.png'\n\
         Finally, call file_list on 'raw' and report the files with their sizes.",
        articles
            .iter()
            .enumerate()
            .map(|(i, (url, path))| format!("  {}. article_fetch url='{}' save_path='{}'", i + 1, url, path))
            .collect::<Vec<_>>()
            .join("\n")
    );

    run_stage(
        &Stage { name: "Ingest", emoji: "📥" },
        &ingest_task,
        &["article_fetch", "image_download", "file_list"],
        12,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    )
    .await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 2 — COMPILE
    // Read each raw article, write a 200-word wiki page with [[WikiLinks]] and
    // YAML frontmatter. Then write a wiki/README.md index.
    // ─────────────────────────────────────────────────────────────────────────
    let compile_task = "
You must call tools now — do not write explanations, just call tools in sequence.

Step 1: file_read path='raw/large-language-model.md'
Step 2: file_write path='wiki/LargeLanguageModel.md' with a 150-word summary of what you read, \
        including [[TransformerModel]], [[AttentionMechanism]], [[PromptEngineering]], \
        [[RetrievalAugmentedGeneration]], [[FineTuning]] as wikilinks.
Step 3: frontmatter operation='write' path='wiki/LargeLanguageModel.md' \
        data={\"title\":\"Large Language Model\",\"category\":\"Architecture\",\"tags\":[\"llm\",\"nlp\",\"ai\"]}

Step 4: file_read path='raw/transformer.md'
Step 5: file_write path='wiki/TransformerModel.md' with a 150-word summary, \
        including [[AttentionMechanism]], [[LargeLanguageModel]], [[PositionalEncoding]], \
        [[SelfAttention]], [[FeedForwardNetwork]] as wikilinks.
Step 6: frontmatter operation='write' path='wiki/TransformerModel.md' \
        data={\"title\":\"Transformer Model\",\"category\":\"Architecture\",\"tags\":[\"transformer\",\"attention\",\"nlp\"]}

Step 7: file_read path='raw/retrieval-augmented-generation.md'
Step 8: file_write path='wiki/RetrievalAugmentedGeneration.md' with a 150-word summary, \
        including [[LargeLanguageModel]], [[VectorDatabase]], [[Embeddings]], \
        [[PromptEngineering]], [[HallucinationProblem]] as wikilinks.
Step 9: frontmatter operation='write' path='wiki/RetrievalAugmentedGeneration.md' \
        data={\"title\":\"Retrieval-Augmented Generation\",\"category\":\"Technique\",\"tags\":[\"rag\",\"retrieval\",\"nlp\"]}

Step 10: file_read path='raw/prompt-engineering.md'
Step 11: file_write path='wiki/PromptEngineering.md' with a 150-word summary, \
         including [[LargeLanguageModel]], [[ChainOfThought]], [[FewShotLearning]], \
         [[InContextLearning]], [[TransformerModel]] as wikilinks.
Step 12: frontmatter operation='write' path='wiki/PromptEngineering.md' \
         data={\"title\":\"Prompt Engineering\",\"category\":\"Technique\",\"tags\":[\"prompting\",\"llm\",\"ai\"]}

Step 13: file_read path='raw/attention-mechanism.md'
Step 14: file_write path='wiki/AttentionMechanism.md' with a 150-word summary, \
         including [[TransformerModel]], [[SelfAttention]], [[QueryKeyValue]], \
         [[LargeLanguageModel]], [[NeuralNetwork]] as wikilinks.
Step 15: frontmatter operation='write' path='wiki/AttentionMechanism.md' \
         data={\"title\":\"Attention Mechanism\",\"category\":\"Architecture\",\"tags\":[\"attention\",\"transformer\",\"ml\"]}

Step 16: file_write path='wiki/README.md' with a one-line description of each of the 5 pages above.
Step 17: file_list path='wiki' to confirm all 6 files exist.
";

    run_stage(
        &Stage { name: "Compile", emoji: "📝" },
        compile_task,
        &["file_list", "file_read", "file_write", "frontmatter"],
        30,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    )
    .await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 3 — INDEX & LINK GRAPH
    // Bulk-index the wiki into RAG, build the wikilink graph, show stats.
    // ─────────────────────────────────────────────────────────────────────────
    run_stage(
        &Stage { name: "Index & Link Graph", emoji: "🔍" },
        "Build the search index and wikilink graph:\n\
         1. rag_index_dir path='wiki' — index all wiki pages into the RAG store.\n\
         2. wikilink_index operation='build' path='wiki' — build the [[WikiLink]] graph.\n\
         3. wikilink_index operation='stats' — print total pages, total links, avg links/page.\n\
         4. For each of the five main pages, call wikilink_index operation='links' \
            page='<PageName>' to list its outgoing links.\n\
            Pages: LargeLanguageModel, TransformerModel, RetrievalAugmentedGeneration, \
            PromptEngineering, AttentionMechanism\n\
         5. wikilink_index operation='orphans' — list pages with no backlinks.\n\
         6. Report: which pages link to which, and which concepts appear most often as link targets.",
        &["rag_index_dir", "rag_search", "wikilink_index", "file_read"],
        12,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    )
    .await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 4 — STUB GENERATION
    // Find wikilinks that point to pages that don't exist yet and create stubs.
    // ─────────────────────────────────────────────────────────────────────────
    run_stage(
        &Stage { name: "Stub Generation", emoji: "🏗️" },
        "Find and create stub pages for all linked-but-missing wiki concepts:\n\
         1. file_list path='wiki' — get the list of existing pages.\n\
         2. For each of the five main pages, call wikilink_index operation='links' \
            page='<Name>' to get all outgoing wikilinks.\n\
         3. Compare the link targets against the existing pages. \
            For each target that does NOT have a corresponding .md file in wiki/, \
            create a stub using file_write at wiki/<TargetName>.md with this template:\n\
            ---\n\
            title: <TargetName>\n\
            stub: true\n\
            ---\n\
            # <TargetName>\n\
            *This is a stub page. Expand with more detail.*\n\
            ## Overview\n\
            Brief placeholder for [[<TargetName>]].\n\
            ## See Also\n\
            (Links to related pages will go here.)\n\
         4. After creating all stubs, rebuild the graph: \
            wikilink_index operation='build' path='wiki'\n\
         5. wikilink_index operation='stats' — show the updated statistics.\n\
         6. Report how many stubs were created and which concepts they represent.",
        &["file_list", "file_write", "file_read", "wikilink_index", "frontmatter"],
        16,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    )
    .await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 5 — MULTI-HOP Q&A
    // Answer three progressively harder questions by combining RAG search with
    // full page reads. Optionally augment with live web search.
    // ─────────────────────────────────────────────────────────────────────────
    let qa_task = format!(
        "Answer the following three questions using the knowledge base. \
         For EACH question:\n\
         a. Use rag_search with an appropriate query (top_k=3) to find relevant pages.\n\
         b. Use file_read to read the full content of the top-ranked page(s).\n\
         c. Synthesise a detailed answer (3-5 sentences) citing the pages used.\n\
         {}\
         \nQuestions:\n\
         Q1 (factual): 'What is the attention mechanism and why is it central to transformers?'\n\
         Q2 (comparative): 'How does Retrieval-Augmented Generation (RAG) differ from \
             standard LLM inference, and what problem does it solve?'\n\
         Q3 (synthesis): 'How do prompt engineering techniques interact with the \
             underlying transformer architecture to improve LLM outputs?'\n\
         \nPresent all three answers clearly labelled Q1, Q2, Q3.",
        if has_tavily {
            "You may also call tavily_search for live web context if the wiki pages \
             don't fully answer a question.\n"
        } else {
            ""
        }
    );

    let qa_result = run_stage(
        &Stage { name: "Multi-Hop Q&A", emoji: "💬" },
        &qa_task,
        &{
            let mut t = vec!["rag_search", "file_read", "wikilink_index"];
            if has_tavily {
                t.push("tavily_search");
            }
            t
        },
        16,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    )
    .await;

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 6 — HEALTH CHECK & REPORT
    // Scan for issues, stamp metadata, and print the final KB summary.
    // ─────────────────────────────────────────────────────────────────────────
    run_stage(
        &Stage { name: "Health Check & Report", emoji: "🩺" },
        "Run health checks on the wiki:\n\
         1. wikilink_index operation='orphans' to find pages with no backlinks.\n\
         2. grep pattern='TODO\\|FIXME\\|XXX' path='wiki' recursive=true to find \
            any TODO markers.\n\
         3. For each wiki file, use frontmatter (operation='read') to inspect metadata.\n\
         4. frontmatter operation='set' on wiki/LargeLanguageModel.md: \
            set key='last_checked', value=<today's date from datetime tool>.\n\
         5. Report: total wiki pages, total backlinks, any orphaned pages, \
            and any missing metadata.",
        &["wikilink_index", "grep", "frontmatter", "datetime", "file_list"],
        10,
        Arc::clone(&registry),
        Arc::clone(&executor),
        Arc::clone(&adapter),
    )
    .await;

    // ── Final summary ─────────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(70));
    println!("🎉  KARPATHY FULL PIPELINE COMPLETE");
    println!("{}", "═".repeat(70));

    if !qa_result.output.is_empty() {
        println!("\n── Knowledge Base Q&A Answers ──────────────────────────────────────\n");
        println!("{}", qa_result.output);
    }

    // Print wiki inventory
    let wiki_files = std::fs::read_dir(sandbox.join("wiki"))
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    println!("\n── Wiki inventory ({} files) ─────────────────────────────────────────", wiki_files.len());
    let mut wiki_sorted = wiki_files.clone();
    wiki_sorted.sort();
    for f in &wiki_sorted {
        let path = sandbox.join("wiki").join(f);
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let is_stub = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .contains("stub: true");
        println!(
            "  {} {} ({} bytes)",
            if is_stub { "·" } else { "•" },
            f,
            size
        );
    }

    let raw_files = std::fs::read_dir(sandbox.join("raw"))
        .map(|entries| entries.flatten().count())
        .unwrap_or(0);

    println!("\n  raw/ files: {}", raw_files);
    println!("  wiki/ files: {} ({} main, {} stubs/other)",
        wiki_files.len(),
        wiki_files.iter().filter(|f| !f.contains("stub") && !f.contains("README")).count(),
        wiki_files.iter().filter(|f| {
            let p = sandbox.join("wiki").join(f);
            std::fs::read_to_string(p).unwrap_or_default().contains("stub: true")
        }).count()
    );

    // ── Cleanup ───────────────────────────────────────────────────────────────
    // Comment out the line below to keep the generated files after the run.
    // let _ = tokio::fs::remove_dir_all(&sandbox).await;
    println!("\nFiles saved to: {}", sandbox.display());
    println!("Pipeline finished.");
}
