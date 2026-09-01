# dsh-memory

**DSH（DeepSeek Harness）本体的跨会话语义记忆插件**——装进任意 profile，让 agent 真的能跨会话记得你。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DeepSeek Harness 插件](https://img.shields.io/badge/DeepSeek%20Harness-plugin-202724.svg)](https://github.com/deepseek-ai/deepseek-harness)
[![dsh-tools](https://img.shields.io/badge/dsh--tools-%5E0.1.0--rc.0-blue.svg)]()

> 🌐 [English](./README.md) · **简体中文**

---

`dsh-memory` 为 DSH 的 **web / headless** profile 注册两个工具和一个斜杠命令，让 agent 能"跨会话记得你"：

- **`memory_add`** — 把一条值得长期记住的事实写入记忆（自动向量化）。
- **`memory_search`** — 按**语义**（余弦相似度）+ 关键词从记忆召回相关事实。
- **`/mem <问题>`** — 斜杠命令，手动触发"**先检索记忆、再回答**"。

实现：智谱 **`embedding-3`**（2048 维）+ SQLite（`node:sqlite`）+ 余弦相似度。**零新增基础设施**——不起向量库、不起容器，几百到几千条记忆毫秒级召回。

## 为什么

agent 每次会话之间都会忘掉一切。`dsh-memory` 给 DSH 一个**持久、语义化的召回层**，却不用新增一个要运行的服务：它用几百字节的 SQLite + 每次写入一次 HTTP 调用，解决"我是谁 / 我们之前约定了什么"这类问题。

## 特性

- **零配置、绝不崩**：缺智谱 key、缺数据库、缺 tools 服务、embedding 网络失败——全部优雅降级，绝不把 DSH 树搞崩（详见"设计红线"）。
- **语义召回 + 关键词兜底**：`memory_search` 先做 embedding 余弦召回，关键词命中做加权；embedding 不可用时自动退化为纯关键词（bigram）。
- **embedding 调用带 8s 超时**：网络差/无网时快速失败回退，不挂起会话。

## 安装

本插件遵循 DSH 官方 **bundle** 约定（`package.json` 的 `dsh.bundle` 声明），因此能被 `dsh plugin` 识别，并以一个配置图层（layer）激活——不是普通的依赖包。

安装进目标 profile（如 web）：

```bash
dsh plugin --profile web add dsh-memory
```

装完**重启 DSH web** 生效（bundle 层在启动时才组合）：

```bash
systemctl restart dsh    # 或以你的方式重启
```

本地安装（未发布/开发期）：

```bash
# 把本目录放进 profile 的 node_modules，并在 package.json 的
# dsh.profile.bundles 末尾加上 "dsh-memory"，然后重启。
```

## 配置

全部可选，不配也能用（退化为关键词检索）：

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `enabled` | `false` 则不注册工具 | `true` |
| `semantic` | `false` 则不做 embedding，纯关键词 | `true` |
| `memoryDbPath` | SQLite 库路径 | `~/.dsh-memory/memories.db` |
| `zhipuEnvPath` | 智谱凭据 `.env` 文件路径 | `~/.dsh-memory/.memenv` |

智谱凭据优先读环境变量，其次读 `zhipuEnvPath` 指向的文件：

```
ZHIPU_API_KEY=你的智谱key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

智谱 key 在 [开放平台 bigmodel.cn](https://open.bigmodel.cn) 免费领取（embedding-3 按量计费）。

## 设计红线（为什么"绝不崩"）

这个插件诞生于一次真实事故：一个未配置好的插件把 DSH web 搞成了 crash-loop。因此本插件：

1. `apply()` 整体 try/catch，任何错误都不向 DSH 树抛出。
2. `inject` 只声明 `tools`（dsh-base 必带的服务），不依赖 `dsh-llm`/`agents`/http。
3. 缺 key / 缺库 / 缺服务 / embedding 失败 → 记日志并降级，绝不中断启动。
4. 工具 `execute` 内错误被捕获并作为普通结果返回，不冒泡成会话异常。

## 工具

- **`memory_add`** — 写入一条值得记住的事实，自动向量化。
- **`memory_search`** — 按语义召回与问题最相关的事实。
- **`/mem`** — 手动兜底："先检索记忆、再回答"。不确定 agent 会不会主动回忆时，输入 `/mem <问题>`，它会先做一次语义检索、注入结果后据其作答。

## 存储结构

```sql
memories(id, text, category, source, created_at, updated_at)
memory_embeddings(memory_id, dim, vector)   -- 2048 维 JSON 数组
```

## 许可

[MIT](./LICENSE)
