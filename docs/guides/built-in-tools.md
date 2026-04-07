# Guide: Built-in Tools

The library ships with 45 ready-to-use tools covering shell, file I/O, HTTP, data
processing, web research, RAG, messaging, and more. Register them all in one call:

```rust
register_built_in_tools(&mut registry).await;
```

For the two MessageBus-aware tools:

```rust
let bus = Arc::new(MessageBus::new());
register_bus_tools(&mut registry, Arc::clone(&bus)).await;
```

---

## Shell

### `bash`

Run an arbitrary shell command.

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | yes | Shell command to execute |
| `timeout_ms` | integer | no | Timeout in ms (default 30 000) |

```
bash command="ls -la src/"
bash command="cargo test --lib" timeout_ms=60000
```

---

## File Operations

All file tools are sandboxed to the agent's working directory (`ToolUseContext.cwd`).

### `file_read`
```
file_read path="src/main.rs"
```

### `file_write`
```
file_write path="output/report.md" content="# Report\n..."
```

### `file_update`
Replace an exact string within a file (fails if `old_text` is not found):
```
file_update path="config.toml" old_text='version = "1.0"' new_text='version = "2.0"'
```

### `file_delete`
```
file_delete path="temp/scratch.txt"
```

### `file_list`

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `.` | Directory to list |
| `recursive` | bool | false | Recurse into subdirectories |

```
file_list path="src" recursive=true
```

### `file_move`
```
file_move from="old/path.txt" to="new/path.txt"
```

### `dir_create` / `dir_delete`
```
dir_create path="output/reports"
dir_delete path="temp"
```

---

## Search

### `grep`

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `pattern` | string | yes | Regex pattern |
| `path` | string | no | File or directory (default `.`) |
| `recursive` | bool | no | Recurse (default false) |

```
grep pattern="TODO\|FIXME" path="src" recursive=true
grep pattern="fn run" path="src/agent/mod.rs"
```

---

## Python Coding

### `python_write`
```
python_write filename="calculator.py" code="def add(a, b): return a + b"
```

### `python_run`
```
python_run filename="calculator.py"
```

### `python_test`
Run pytest on a test file:
```
python_test filename="test_calculator.py"
```

---

## Repository Analysis

### `repo_ingest`

Recursively reads a directory and returns its structure + file contents as a single
context string. Useful for giving an agent full codebase awareness.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `.` | Root directory |
| `max_depth` | integer | 5 | Max directory depth |
| `extensions` | array | common code exts | File extensions to include |

```
repo_ingest path="src" max_depth=3
repo_ingest path="." extensions=[".rs",".toml"]
```

---

## HTTP / Networking

### `http_get`
```
http_get url="https://api.example.com/data"
http_get url="https://api.example.com/data" headers={"Authorization": "Bearer TOKEN"}
```

Response body is capped at 4 MB. Returns the response body as a string.

### `http_post`
```
http_post url="https://api.example.com/submit"
          body={"key": "value"}
          headers={"Content-Type": "application/json"}
```

---

## Data Processing

### `json_parse`

Extract a value using a JSON Pointer (`/field/sub`):
```
json_parse json='{"a":{"b":42}}' pointer="/a/b"   // returns "42"
json_parse json='{"items":[1,2,3]}'                // returns whole object
```

### `json_transform`

| `operation` | What it does |
|-------------|-------------|
| `keys` | Top-level keys as JSON array |
| `values` | Top-level values as JSON array |
| `length` | Number of keys or array elements |
| `/pointer` | Map an array: extract field at pointer from each element |

```
json_transform json='{"a":1,"b":2}' operation="keys"     // ["a","b"]
json_transform json='[{"n":"Alice"},{"n":"Bob"}]' operation="/n"  // ["Alice","Bob"]
```

### `csv_read`
```
csv_read path="sales.csv"   // returns JSON array of row objects
```

### `csv_write`
```
csv_write path="output.csv" rows=[{"name":"Alice","score":95},{"name":"Bob","score":87}]
```

---

## Math & Expressions

### `math_eval`

Evaluates a mathematical expression using `evalexpr`:
```
math_eval expression="2 * (3 + 4)"          // 14
math_eval expression="sqrt(144) + log(100)"  // 14
math_eval expression="42 > 10"              // true
```

---

## Date & Time

### `datetime`

| `operation` | Description |
|-------------|-------------|
| `now` | Current date/time |
| `format` | Format a timestamp |
| `parse` | Parse a date string |
| `diff` | Difference between two timestamps |

```
datetime operation="now" format="%Y-%m-%d"
datetime operation="now" format="%Y-%m-%dT%H:%M:%SZ"
datetime operation="diff" from="2024-01-01" to="2024-12-31" unit="days"
```

---

## Text Processing

### `text_regex`

| `operation` | Description |
|-------------|-------------|
| `find_all` | Return all matches as JSON array |
| `replace` | Replace all matches |
| `split` | Split text on pattern |

```
text_regex text="2024-01-15 event" pattern="\d{4}-\d{2}-\d{2}" operation="find_all"
text_regex text="hello world" pattern="world" operation="replace" replacement="Rust"
```

### `text_chunk`

| `method` | Description |
|----------|-------------|
| `chars` | Split by character count |
| `words` | Split by word count |
| `lines` | Split by line count |

```
text_chunk text="..." method="words" size=200 overlap=20
```

Returns a JSON array of chunk strings.

---

## Environment & System

### `env_get`

Reads allow-listed environment variables. Allowed names: `PATH`, `HOME`, `USER`,
`SHELL`, `LANG`, `TERM`, `EDITOR`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `TAVILY_API_KEY`.

```
env_get name="PATH"
```

### `system_info`

Returns a JSON object with `os`, `arch`, `cpu_count`, `cwd`:
```
system_info
// {"os":"linux","arch":"x86_64","cpu_count":8,"cwd":"/home/user/project"}
```

---

## Cache

In-process key-value store, shared across all agents in a run.

### `cache_set`
```
cache_set key="api_token"  value="abc123"
cache_set key="temp_result" value="42" ttl_ms=60000   // expires after 1 minute
```

### `cache_get`
```
cache_get key="api_token"   // returns value or "(not found / expired)"
```

---

## Encoding & Hashing

### `base64`
```
base64 operation="encode" data="Hello, World!"
base64 operation="decode" data="SGVsbG8sIFdvcmxkIQ=="
```

### `hash_file`

FNV-1a 64-bit hash of a file (fast, non-cryptographic):
```
hash_file path="output/report.md"
```

---

## Web & Search

### `web_fetch`

Fetches a URL and converts the HTML to clean Markdown (scripts, styles, and nav
elements stripped). 4 MB response cap.

```
web_fetch url="https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html"
```

### `tavily_search`

Real-time web search via Tavily API. Requires `TAVILY_API_KEY` in environment.

```
tavily_search query="Rust ownership model 2024" max_results=5
```

### `schema_validate`

Validate a JSON value against a JSON Schema:
```
schema_validate value={"name":"Alice","age":30}
                schema={"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
```

Returns `{"valid": true}` or `{"valid": false, "errors": [...]}`.

---

## MessageBus Tools

Require injection via `register_bus_tools(registry, bus)`.

### `bus_publish`

```
bus_publish from="researcher" to="writer"   content="findings: ..."   // point-to-point
bus_publish from="coordinator" to="*"       content="task complete"    // broadcast
```

### `bus_read`

Reads messages addressed to the current agent (identified by `AgentInfo.name`):
```
bus_read                        // all messages
bus_read unread_only=true       // only unread
```

---

## Knowledge Base (RAG)

See [rag.md](rag.md) for a full guide.

### `rag_add`
```
rag_add id="doc1" content="Rust ownership model..."
rag_add id="doc1" content="Updated content..."   // update existing
rag_add id="doc1" content="..." metadata={"source":"url","category":"systems"}
```

### `rag_search`
```
rag_search query="memory safety" top_k=3
```

### `rag_clear`
```
rag_clear id="doc1"   // remove one document
rag_clear             // wipe entire store
```

### `rag_index_dir`
```
rag_index_dir path="wiki"   // index all .md and .txt files recursively
```

---

## Karpathy Pipeline Tools

See [knowledge-base-pipeline.md](knowledge-base-pipeline.md) for a full guide.

### `article_fetch`

Clip a web article to Markdown with YAML frontmatter:
```
article_fetch url="https://en.wikipedia.org/wiki/Rust_(programming_language)"
              save_path="raw/rust.md"
```

Saves: `title`, `source`, `date_clipped`, `word_count` as frontmatter.

### `frontmatter`

Read and write YAML frontmatter in `.md` files:

| `operation` | Description |
|-------------|-------------|
| `read` | Return all frontmatter as JSON |
| `write` | Replace entire frontmatter with `data` object |
| `set` | Set a single key |
| `remove` | Remove a single key |
| `list_keys` | Return key names as array |

```
frontmatter operation="read"       path="wiki/Rust.md"
frontmatter operation="set"        path="wiki/Rust.md" key="last_checked" value="2026-04-07"
frontmatter operation="write"      path="wiki/Rust.md" data={"title":"Rust","tags":["systems","memory"]}
frontmatter operation="remove"     path="wiki/Rust.md" key="draft"
```

### `wikilink_index`

Build and query a `[[WikiLink]]` backlink graph:

| `operation` | Description |
|-------------|-------------|
| `build` | Scan directory and build graph |
| `stats` | Total pages, links, avg links/page |
| `links` | Outgoing links from a page |
| `backlinks` | Pages that link to a page |
| `orphans` | Pages with no backlinks |
| `add` | Add a single page's links |

```
wikilink_index operation="build"     path="wiki"
wikilink_index operation="stats"
wikilink_index operation="links"     page="LargeLanguageModel"
wikilink_index operation="backlinks" page="AttentionMechanism"
wikilink_index operation="orphans"
```

### `image_download`

Download an image from a URL into the sandbox (50 MB cap):
```
image_download url="https://upload.wikimedia.org/.../diagram.png"
               save_path="raw/images/diagram.png"
```

---

## Utility Tools

### `sleep`

Pause execution (rate limiting, polling):
```
sleep ms=500       // 500ms pause (max 300 000)
```

### `random`

| `kind` | Description | Extra inputs |
|--------|-------------|-------------|
| `uuid` | UUID v4 | — |
| `int` | Random integer | `min`, `max` (default 0–100) |
| `float` | Float in [0, 1) | — |
| `choice` | Pick from list | `items` (array) |
| `string` | Alphanumeric string | `length` (default 16) |

```
random kind="uuid"
random kind="int"    min=1 max=6
random kind="choice" items=["rock","paper","scissors"]
random kind="string" length=32
```

### `template`

Render `{{variable}}` placeholders:
```
template template="Hello, {{name}}! Order #{{id}} has shipped."
         vars={"name":"Alice","id":"9821"}
template template="{{missing}} stays" vars={} strict=false  // placeholder unchanged
template template="{{x}}" vars={} strict=true               // error: missing x
```

### `diff`

Unified diff between two strings or files:
```
diff a="old line\nsame line" b="new line\nsame line"
diff a="before.txt" b="after.txt" mode="files" context=5
```

Output mirrors `diff -u` format: `+` added, `-` removed, ` ` context.

### `zip`

| `operation` | Description |
|-------------|-------------|
| `create` | Bundle files into a zip archive |
| `extract` | Unpack to a directory |
| `list` | Inspect contents |

```
zip operation="create"  archive="bundle.zip" files=["a.txt","b.txt"]
zip operation="list"    archive="bundle.zip"
zip operation="extract" archive="bundle.zip" dest="output/"
```

### `git`

Safe Git wrapper. Allowed sub-commands: `status`, `log`, `diff`, `show`, `branch`,
`tag`, `remote`, `stash`, `ls-files`, `shortlog`, `describe`, `rev-parse`,
`cat-file`, `add`, `commit`, `init`. Force flags are blocked.

```
git args="status"
git args="log --oneline -10"
git args="diff HEAD~1"
git args="add src/main.rs"
git args="commit -m 'fix: update handler'"
```

### `url`

| `operation` | Description |
|-------------|-------------|
| `parse` | Break URL into components |
| `build` | Construct URL from parts |
| `encode` | Percent-encode a string |
| `decode` | Percent-decode a string |
| `join` | Resolve relative URL against a base |

```
url operation="parse"  url="https://example.com/api?q=rust#docs"
url operation="build"  scheme="https" host="api.example.com" path="/search" query={"q":"hello world"}
url operation="encode" url="hello world & foo=bar"
url operation="decode" url="hello%20world%20%26%20foo%3Dbar"
url operation="join"   base="https://docs.rs/tokio/latest/tokio/" url="../time/"
```
