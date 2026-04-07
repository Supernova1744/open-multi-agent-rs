# Guide: LLM Knowledge-Base Pipeline

This guide describes how to implement Andrej Karpathy's LLM knowledge-base pipeline
using the built-in tools. The pipeline transforms raw web content into a structured,
searchable, and self-maintaining wiki.

## Overview

```
Web articles
    │
    ▼  article_fetch, image_download
raw/*.md  (plain Markdown clippings)
    │
    ▼  file_read → LLM → file_write, frontmatter
wiki/*.md  (polished pages with [[WikiLinks]] and YAML metadata)
    │
    ▼  rag_index_dir, wikilink_index build
RAG store + backlink graph
    │
    ▼  rag_search, file_read
Q&A answers
    │
    ▼  grep, frontmatter, datetime
Health check report
```

## Stage 1 — Ingest

Clip web articles into `raw/` as Markdown files with YAML frontmatter.

```
article_fetch url="https://en.wikipedia.org/wiki/Large_language_model"
              save_path="raw/large-language-model.md"
```

Each saved file looks like:

```markdown
---
title: Large language model
source: https://en.wikipedia.org/wiki/Large_language_model
date_clipped: 2026-04-07
word_count: 4821
---

A large language model (LLM) is a type of...
```

Download referenced images:

```
image_download url="https://upload.wikimedia.org/..."
               save_path="raw/images/transformer-arch.png"
```

Confirm:

```
file_list path="raw"
```

## Stage 2 — Compile

The LLM reads each raw article and writes a polished wiki page with:
- A 150-200 word summary
- `[[WikiLinks]]` to related concepts (5+ per page)
- YAML frontmatter: `title`, `tags`, `category`, `related`

```
file_read path="raw/large-language-model.md"
file_write path="wiki/LargeLanguageModel.md"
           content="---\ntitle: Large Language Model\n...\n---\n\n# Large Language Model\n\n[[TransformerModel]]..."
frontmatter operation="write" path="wiki/LargeLanguageModel.md"
            data={"title":"Large Language Model","category":"Architecture","tags":["llm","nlp"]}
```

Tip for reliable tool use: give the agent **numbered step-by-step instructions**
rather than open-ended prose. The agent is more likely to call tools when each
step maps to exactly one tool call.

## Stage 3 — Index & Link Graph

Build the search index and backlink graph:

```
rag_index_dir path="wiki"
wikilink_index operation="build" path="wiki"
wikilink_index operation="stats"
wikilink_index operation="links"    page="LargeLanguageModel"
wikilink_index operation="orphans"
```

`wikilink_index stats` returns:

```json
{"pages": 5, "total_links": 28, "avg_links_per_page": 5.6}
```

`wikilink_index orphans` returns pages that no other page links to — good
candidates for adding backlinks.

## Stage 4 — Stub Generation

Find `[[WikiLinks]]` that point to pages that don't exist yet and create stub files:

```
file_list path="wiki"
wikilink_index operation="links" page="LargeLanguageModel"
// compare results against file_list output
// for each missing page:
file_write path="wiki/VectorDatabase.md"
           content="---\ntitle: Vector Database\nstub: true\n---\n# Vector Database\n*Stub — expand this page.*"
wikilink_index operation="build" path="wiki"
```

Stubs keep the link graph consistent and can be expanded later.

## Stage 5 — Q&A

Answer questions using RAG retrieval + full-page reads:

```
// Find relevant pages
rag_search query="what is attention mechanism" top_k=3

// Read the top result in full
file_read path="wiki/AttentionMechanism.md"

// Synthesise answer from retrieved content
```

For multi-hop questions (e.g. "how does RAG improve LLM accuracy?"), retrieve
multiple pages:

```
rag_search query="RAG improves accuracy" top_k=3
// then file_read each top hit
```

## Stage 6 — Health Check

Scan for issues and stamp metadata:

```
grep pattern="stub: true" path="wiki" recursive=true
grep pattern="TODO|FIXME" path="wiki" recursive=true
wikilink_index operation="orphans"
datetime operation="now" format="%Y-%m-%d"
frontmatter operation="set" path="wiki/LargeLanguageModel.md"
            key="last_checked" value="2026-04-07"
file_list path="wiki"
```

## Running the examples

```bash
# Quick 4-stage demo (2 articles, Rust docs)
cargo run --example 19_knowledge_base_pipeline

# Full 6-stage pipeline (5 AI/ML Wikipedia articles)
cargo run --example 21_karpathy_full_pipeline

# With live Tavily web search for Q&A augmentation
TAVILY_API_KEY=tvly-... cargo run --example 21_karpathy_full_pipeline
```

## Tool summary

| Stage | Tool | Purpose |
|-------|------|---------|
| Ingest | `article_fetch` | Clip article to Markdown with frontmatter |
| Ingest | `image_download` | Save referenced images |
| Ingest | `file_list` | Verify ingested files |
| Compile | `file_read` | Read raw article |
| Compile | `file_write` | Write wiki page |
| Compile | `frontmatter` | Set YAML metadata |
| Index | `rag_index_dir` | Bulk RAG index |
| Index | `wikilink_index` | Build + inspect backlink graph |
| Stubs | `file_write` | Create stub pages |
| Q&A | `rag_search` | Find relevant pages |
| Q&A | `file_read` | Read full pages |
| Q&A | `tavily_search` | Live web supplement (optional) |
| Health | `grep` | Find stubs / TODOs |
| Health | `frontmatter` | Update `last_checked` |
| Health | `datetime` | Get today's date |
| Health | `file_list` | Final inventory |
