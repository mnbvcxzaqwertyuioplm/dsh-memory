# dsh-memory

**Cross-session semantic memory for DSH (DeepSeek Harness), built into the Harness itself.**

`dsh-memory` registers two tools for the web / headless profiles so your agent can actually remember you across sessions:

- **`memory_add`** — persist a fact worth keeping for the long term (auto-vectorized on the way in).
- **`memory_search`** — recall relevant facts by **semantics** (cosine similarity) with keyword fallback.

Under the hood: Zhipu `embedding-3` (2048-dim) + SQLite (`node:sqlite`) + cosine similarity. **Zero new infrastructure** — no vector database, no containers. Hundreds to thousands of memories recall in milliseconds.

## Features

- **Config optional, never crashes**: missing Zhipu key, missing database, missing `tools` service, failing embedding call — all degrade gracefully. This plugin will never take down the DSH tree (see [Design guarantee](#design-guarantee)).
- **Semantic recall with keyword fallback**: `memory_search` scores by embedding cosine similarity first, then weights keyword hits; when embeddings are unavailable it falls back to pure keyword (bigram) matching.
- **8s embedding timeout**: on a bad or missing network it fails fast and falls back to keyword search instead of hanging the session.

## Installation

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
| `memoryDbPath` | SQLite database path | `~/workspace/Jarvis/data/jarvis-memory.db` |
| `zhipuEnvPath` | Path to the Zhipu credentials `.env` file | `~/workspace/Jarvis/.memenv` |

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

## Tools and commands

- **`memory_add` / `memory_search`** — as above; the agent calls them on its own when needed.
- **`/mem <question>`** — a slash command that forces "**search memory first, then answer**". E.g. `/mem what is jarvis' sibling agent called` runs a semantic memory search, injects the results into the session, and the agent answers from those results. Use it when you are not sure the agent will remember on its own.

## Storage schema

```sql
memories(id, text, category, source, created_at, updated_at)
memory_embeddings(memory_id, dim, vector)   -- 2048-dim JSON array
```

## License

[MIT](./LICENSE)

---

The original Chinese README lives at [README.md](./README.md).
