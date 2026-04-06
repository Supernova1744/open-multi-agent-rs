/// Example 11 — Python Coding Agent
///
/// Demonstrates the built-in Python coding tools:
///   • python_write  — write a .py file and check its syntax
///   • python_run    — execute a script or inline code snippet
///   • python_test   — run pytest on a test file
///
/// Also exercises the file-management tools:
///   • file_list, file_read, file_update, file_delete, dir_create, dir_delete
///
/// The agent is given a working directory (temp dir) and asked to:
///   1. Write a small Python module
///   2. Write a pytest test file for it
///   3. Run the tests
///   4. Fix any failures and re-run
///
/// Run:
///   cargo run --example 11_python_coding_agent
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
    // ── Sandbox: a temporary directory for all file operations ────────────────
    let sandbox = std::env::temp_dir().join("open_multi_agent_python_demo");
    tokio::fs::create_dir_all(&sandbox).await.unwrap();
    println!("Sandbox: {}\n", sandbox.display());

    // ── Tool registry with all built-ins ──────────────────────────────────────
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().await;
        register_built_in_tools(&mut reg).await;
    }
    let executor = Arc::new(ToolExecutor::new(Arc::clone(&registry)));

    // ── Adapter ───────────────────────────────────────────────────────────────
    let adapter = Arc::from(create_adapter(
        "openrouter",
        Some(api_key()),
        Some("https://openrouter.ai/api/v1".to_string()),
    ));

    // ── Agent config ──────────────────────────────────────────────────────────
    let config = AgentConfig {
        name: "python-coder".to_string(),
        model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        system_prompt: Some(format!(
            "You are an expert Python developer. \
             Your working directory is '{}'. \
             Use the available tools to write, run, and test Python code. \
             Always check syntax after writing files. \
             When tests fail, read the error output carefully, fix the code, and re-run. \
             Be concise — don't over-explain.",
            sandbox.display()
        )),
        max_turns: Some(12),
        // Give the agent access to the Python and file tools only.
        tools: Some(vec![
            "python_write".to_string(),
            "python_run".to_string(),
            "python_test".to_string(),
            "file_read".to_string(),
            "file_write".to_string(),
            "file_update".to_string(),
            "file_delete".to_string(),
            "file_list".to_string(),
            "dir_create".to_string(),
        ]),
        ..Default::default()
    };

    let mut agent = Agent::new(config, Arc::clone(&registry), Arc::clone(&executor));

    // Override the executor's context so it uses the sandbox directory.
    // (AgentRunner passes context.cwd from the config; here we set it via
    //  ToolUseContext which is constructed in the runner from AgentInfo.
    //  A simpler approach: set TMPDIR / cwd at the process level for the demo.)
    //
    // The simplest portable way: set the process working directory.
    std::env::set_current_dir(&sandbox).unwrap();

    // ── Task ──────────────────────────────────────────────────────────────────
    let task = "\
        Write a Python module called `calculator.py` with four functions: \
        add(a, b), subtract(a, b), multiply(a, b), divide(a, b). \
        The divide function should raise ValueError when b is zero. \
        Then write `test_calculator.py` with pytest tests covering all four functions \
        including the divide-by-zero case. \
        Run the tests and make sure they all pass. \
        Report the final pytest output.";

    println!("Task:\n{}\n", task);
    println!("{}", "─".repeat(60));

    match agent.run(task, adapter).await {
        Ok(AgentRunResult { output, turns, token_usage, tool_calls, .. }) => {
            println!("{}", "─".repeat(60));
            println!("\nAgent finished in {} turns", turns);
            println!("Tokens — input: {}, output: {}", token_usage.input_tokens, token_usage.output_tokens);
            println!("Tool calls: {}", tool_calls.len());
            for tc in &tool_calls {
                println!("  • {} ({}ms)", tc.tool_name, tc.duration_ms);
            }
            println!("\nFinal output:\n{}", output);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    let _ = tokio::fs::remove_dir_all(&sandbox).await;
    println!("\nSandbox cleaned up.");
}
