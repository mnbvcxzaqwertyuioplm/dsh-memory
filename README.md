# dsh-memory

**Cross-session semantic memory for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)** — installed into any profile, so your agent actually remembers you across sessions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DeepSeek Harness plugin](https://img.shields.io/badge/DeepSeek%20Harness-plugin-202724.svg)](https://github.com/deepseek-ai/deepseek-harness)
[![dsh-tools](https://img.shields.io/badge/dsh--tools-%5E0.1.0--rc.0-blue.svg)]()

> 🌐 **English** · [简体中文](./README.zh-CN.md)

---

`dsh-memory` registers two tools for the **web / headless** profiles, so your agent can remember you across sessions:

- **`memory_add`** — persist a fact worth keeping long-term, auto-vectorized on the way in.
- **`memory_search`** — recall relevant facts by **semantics** (cosine similarity), with a keyword fallback.
- **`/mem <question>`** — type this in the composer and the agent will call `memory_search` first (results render as a collapsible tool card), then answer from memory.

Under the hood: Zhipu **`embedding-3`** (2048-dim) + SQLite (`node:sqlite`) + cosine similarity. **Zero new infrastructure** — no vector database, no sidecar containers. Hundreds to thousands of memories recall in milliseconds.

## Why

Your agent forgets everything between sessions. `dsh-memory` gives it a durable, semantic recall layer without adding a new service to run. It solves the "who am I / what did we agree on" problem with a few hundred bytes of SQLite and one HTTP call per write.

## Features

- **Zero-config, never crashes** — a missing Zhipu key, a missing database, a missing `tools` service, or a failing embedding call all degrade gracefully. This plugin will never take the DSH tree down (see [Design guarantee](#design-guarantee)).
- **Semantic recall with keyword fallback** — `memory_search` scores by embedding cosine similarity first, then weights keyword hits; when embeddings are unavailable it falls back to pure keyword (bigram) matching.
- **8s embedding timeout** — on a bad or missing network it fails fast and falls back to keyword search instead of hanging the session.
- **Write-time semantic dedup (merge)** — `memory_add` runs one cosine pass over the library before writing; a new fact that is highly similar to an existing entry (cosine ≥ 0.9) **merges into that row** — keeps the latest wording, inherits/updates `category`, maintains `updated_at`, and never adds a duplicate (returns `merged: true/false`). Correction convention: to override a stale/conflicting memory, simply record the new fact — newest wins.
- **SQLite WAL + busy_timeout=5000** — applied when the DB opens; concurrent writers (e.g. DSH web + the Jarvis brain sharing one library) no longer hit `SQLITE_BUSY`.

## Installation

This plugin follows the official DSH **bundle** convention (`dsh.bundle` in `package.json`), so `dsh plugin` recognizes it and activates it as a configuration layer — not a plain dependency.

Install into the target profile (e.g. `web`):

```bash
dsh plugin --profile web add dsh-memory
```

After installing, **restart DSH web** for it to take effect (the bundle layer is only composed at startup):

```bash
systemctl restart dsh    # or restart however you run DSH
```

Local / pre-release install:

```bash
# Put this directory into the profile's node_modules and append "dsh-memory"
# to dsh.profile.bundles in package.json, then restart.
```

## Configuration

Everything is optional — skip it all and the plugin still works (falling back to keyword search):

| Config | Description | Default |
|---|---|---|
| `enabled` | `false` disables tool registration | `true` |
| `semantic` | `false` skips embeddings, keyword-only | `true` |
| `memoryDbPath` | SQLite database path | `~/.dsh-memory/memories.db` |
| `zhipuEnvPath` | Path to the Zhipu credentials `.env` file | `~/.dsh-memory/.memenv` |
| `forceMemoryWords` | Force-memory signal words (array). When any word appears in a user message, the agent must call `memory_add` first, then answer | `[]` (disabled by default) |

`forceMemoryWords` example (set in your profile's `cordis.patch.yml`, not in the public repo):

```yaml
- id: dsh-memory
  config:
    forceMemoryWords:
      - 记住
      - 重要
      - 珍贵
      - 特别
      - 务必
      - 一定
      - 必须
```

Zhipu credentials are read from environment variables first, then from the file at `zhipuEnvPath`:

```
ZHIPU_API_KEY=your-zhipu-key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

Get a Zhipu key for free at [open.bigmodel.cn](https://open.bigmodel.cn) (`embedding-3` is billed per use).

## Design guarantee

This plugin was born from a real incident: a misconfigured plugin sent a DSH web profile into a crash-loop. So `dsh-memory` is deliberately defensive:

1. `apply()` is wrapped in an overall `try/catch` — no error is ever thrown up into the DSH tree.
2. `inject` only declares `tools` (a service `dsh-base` always provides). It does **not** depend on `dsh-llm`, `agents`, or `http`.
3. A missing key / database / service, or a failed embedding call → it logs and degrades, never interrupting startup.
4. Errors raised inside a tool's `execute()` are caught and returned as ordinary results, never bubbling up into a session exception.

## Tools

- **`memory_add`** — write a fact worth remembering; vectorized automatically.
- **`memory_search`** — semantically recall the most relevant facts for a question.
- **`/mem`** — a manual override: "search memory first, then answer." If you're not sure the agent will recall on its own, type `/mem <question>` and it will run a semantic search, inject the results, and answer from them.

## Storage schema

```sql
memories(id, text, category, source, created_at, updated_at)
memory_embeddings(memory_id, dim, vector)   -- 2048-dim JSON array
```

## License

[MIT](./LICENSE)
