# Guide: RAG Knowledge Base

The library ships with an in-process Retrieval-Augmented Generation (RAG) store.
Documents are indexed in memory; queries are scored using term frequency (TF) and
ranked by relevance. No external vector database is required.

## Core tools

| Tool | What it does |
|------|-------------|
| `rag_add` | Add or update a document |
| `rag_search` | TF-scored search — returns the top-K most relevant docs |
| `rag_clear` | Remove one document or wipe the entire store |
| `rag_index_dir` | Bulk-index a directory of `.md` / `.txt` files |

## Adding documents

```
rag_add id="rust-ownership" content="Rust's ownership model ensures memory safety without GC..."
rag_add id="python-gc"      content="Python uses reference counting with a cyclic garbage collector..."
```

Each document has a unique `id`. Calling `rag_add` with an existing `id` **replaces** it.

Optional `metadata` object is preserved and returned in search results:

```
rag_add id="rust-ownership"
        content="..."
        metadata={"source": "https://doc.rust-lang.org", "category": "systems"}
```

## Searching

```
rag_search query="memory safety programming" top_k=3
```

Returns a JSON array of hits, ranked by TF score:

```json
[
  { "id": "rust-ownership", "score": 0.42, "content": "...", "metadata": {...} },
  { "id": "cpp-raii",       "score": 0.21, "content": "...", "metadata": {} }
]
```

`top_k` defaults to `5`. Set it lower for speed, higher for recall.

## Removing documents

```
rag_clear id="rust-ownership"   // remove one document
rag_clear                       // wipe the entire store
```

## Bulk indexing a directory

```
rag_index_dir path="wiki"
```

Recursively walks `path`, reads every `.md` and `.txt` file, and adds each as a
document. The document ID is the relative file path (e.g. `wiki/LargeLanguageModel.md`).
YAML frontmatter (if present) is parsed and stored as metadata.

## Typical agent workflow

```rust
// Setup
register_built_in_tools(&mut reg).await;

// Agent adds documents
agent.run("Add the following docs to the knowledge base: ...", adapter).await?;

// Agent searches and answers questions
agent.run("What does the knowledge base say about memory safety?", adapter).await?;
```

Or drive it programmatically from another stage:

```rust
// Stage 1: index the wiki
// Stage 2: answer questions using rag_search
```

## Combining with wikilink_index

For a full knowledge base, pair RAG with the `wikilink_index` tool:

- `rag_search` finds semantically relevant pages
- `wikilink_index backlinks` / `links` navigates the graph structure
- `file_read` loads the full page content for detailed answers

See [knowledge-base-pipeline.md](knowledge-base-pipeline.md) for the complete pattern.

## Persistence note

The RAG store is **in-process and in-memory**. It is reset when the process exits.
For persistence across runs, use `rag_index_dir` to rebuild the index from your
Markdown files at startup.

## Run the example

```bash
cargo run --example 17_rag_knowledge_base
```
