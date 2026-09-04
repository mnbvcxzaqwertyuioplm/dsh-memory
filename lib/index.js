/**
 * dsh-memory — DSH 本体（web/headless profile）的跨会话语义记忆工具。
 *
 * 目标：让 DSH 自己"记得你"——通过 memory_add / memory_search
 * 两个工具做跨会话的语义召回（embedding 向量 + 余弦相似度，与常见的
 * agent-bridge 记忆方案同一套逻辑）。这个独立插件只关心"注册工具"，
 * 不关心 http 桥 / agents / session，因此能在任何带 tools 服务的 profile
 * （web、headless）里随 dsh-base 之后安全加载。
 *
 * 设计红线（吸取 dsh-headless-open 把 web profile 搞崩的教训）：
 *   - 配置可选：缺 ZHIPU_API_KEY → 退化为纯关键词检索；缺 .memenv / 缺库直接跳过。
 *   - 绝不因缺依赖 / 缺服务 / 缺库 / embedding 失败而崩：apply() 整体 try/catch，
 *     任何一步失败都优雅降级，绝不向 DSH 树抛出未捕获异常。
 *   - inject 只声明确定存在的服务（本插件只依赖 tools），避免因注入缺失服务而加载失败。
 *
 * 已知能力边界：
 *   - 智谱 embedding-3：2048 维，用于语义召回；失败自动回退关键词 bigram 打分。
 *   - 存储独立于 xiaozhi 的 powermem_dev.db（避免相互污染）。
 *   - P1（MEMORY-STRATEGY 2026-09-03 定稿，Phase 1 已实施）：memory_add 写前语义查重
 *     （与既有行余弦 ≥ 0.9 → 合并更新，不新增重复）；ensureMemoryDb 开 WAL + busy_timeout=5000
 *     （治双进程 SQLITE_BUSY）；合并路径维护 updated_at（纠错 = 新者胜）。
 *   - 不依赖 @deepseek-ai/dsh-llm（不驱动 agent，不建会话），故无需 dsh-llm symlink。
 */
import { defineTool } from "@deepseek-ai/dsh-tools";
import { DatabaseSync } from "node:sqlite";
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

const name = "dsh-memory";
const inject = ["tools"];

// 用普通 Map 缓存 DB 句柄（键是字符串路径；WeakMap 只接受对象键）。
const _memCache = /* @__PURE__ */ new Map();

/** 从 .env 文件粗略解析 key=value（不做复杂解析，够用即可）。 */
function loadDotEnv(path) {
	const out = {};
	try {
		for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
			const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/.exec(line);
			if (!m || m[2].startsWith("#")) continue;
			let v = m[2];
			if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
			out[m[1]] = v;
		}
	} catch {
		// 文件不存在/不可读 → 返回空对象
	}
	return out;
}

/** 取智谱凭证：优先环境变量，其次 cfg.zhipuEnvPath 指向的 .env。 */
function getZhipuCreds(cfg) {
	const key = process.env.ZHIPU_API_KEY;
	const base = process.env.ZHIPU_BASE_URL || "https://open.bigmodel.cn/api/paas/v4/";
	if (key) return { key, base };
	let env = {};
	try {
		env = loadDotEnv(cfg.zhipuEnvPath);
	} catch {
		env = {};
	}
	return { key: env.ZHIPU_API_KEY || "", base: env.ZHIPU_BASE_URL || base };
}

/** embedding 调用超时（毫秒）。网络差/无网时快速失败，回退关键词，避免挂起会话。 */
const EMBED_TIMEOUT_MS = 8000;

/** 调智谱 embedding-3 把文本向量化。失败/超时抛错（由调用方回退）。 */
async function zhipuEmbed(text, creds) {
	if (!creds || !creds.key) throw new Error("no zhipu key");
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS);
	try {
		const r = await fetch(`${creds.base.replace(/\/$/, "")}/embeddings`, {
			method: "POST",
			headers: { "content-type": "application/json", authorization: `Bearer ${creds.key}` },
			body: JSON.stringify({ model: "embedding-3", input: [text] }),
			signal: controller.signal,
		});
		const j = await r.json();
		const vec = j?.data?.[0]?.embedding;
		if (!Array.isArray(vec)) throw new Error(`embed 失败: ${j?.error?.message || r.status}`);
		return vec;
	} finally {
		clearTimeout(timer);
	}
}

/** 余弦相似度。 */
function cosine(a, b) {
	if (!Array.isArray(a) || !Array.isArray(b)) return 0;
	let dot = 0, na = 0, nb = 0;
	for (let i = 0; i < Math.min(a.length, b.length); i += 1) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
	const denom = Math.sqrt(na) * Math.sqrt(nb);
	return denom ? dot / denom : 0;
}

/** 打开/建库，建 memories + memory_embeddings 两张表。失败抛错。 */
function ensureMemoryDb(path) {
	const existing = _memCache.get(path);
	if (existing) return existing;
	mkdirSync(dirname(path), { recursive: true });
	const db = new DatabaseSync(path);
	db.exec(`
		CREATE TABLE IF NOT EXISTS memories (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			text TEXT NOT NULL,
			category TEXT,
			source TEXT DEFAULT 'dsh',
			created_at TEXT DEFAULT (datetime('now')),
			updated_at TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
		CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category);
		CREATE TABLE IF NOT EXISTS memory_embeddings (
			memory_id INTEGER PRIMARY KEY,
			dim INTEGER NOT NULL,
			vector TEXT NOT NULL,
			FOREIGN KEY(memory_id) REFERENCES memories(id)
		);
	`);
	try {
		// P0（MEMORY-STRATEGY Phase 1，2026-09-03 定稿）：WAL + busy_timeout，
		// 治 web 与 jarvis 大脑两进程并行写共享库的 SQLITE_BUSY。失败不阻塞。
		db.exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
	} catch {
		// 忽略（只读介质/旧 SQLite 等）：继续默认模式
	}
	_memCache.set(path, db);
	return db;
}

/** 首次建库时播一条通用说明（让记忆不是空的；不硬编码任何用户画像，避免越权）。 */
function seedInitialMemories(db) {
	try {
		const { c } = db.prepare("SELECT COUNT(*) AS c FROM memories").get();
		if (c > 0) return;
		const stmt = db.prepare("INSERT INTO memories (text, category, source) VALUES (?, ?, 'seed')");
		stmt.run("本助手已启用跨会话语义记忆。用户主动告知的重要个人信息（姓名、住址、喜好、约定）会存入记忆，供后续会话回忆。", "说明");
	} catch {
		// 播种失败不阻塞
	}
}

/**
 * P1 写前语义查重（轻量版）——MEMORY-STRATEGY 候选 2，2026-09-03 定稿、实施：
 * memory_add 写入前对整库向量做一次余弦查重；与既有行相似度 ≥ DUP_COSINE_THRESHOLD 视为同一事实，
 * 合并更新既有行（text 保留最新、category 空缺时继承旧值、updated_at 落时间、刷新该行向量），
 * 返回既有 id；否则照旧 INSERT 新行。embedFn 缺失/失败 → 跳过查重直接新增（fail-open，绝不阻塞写入）。
 */
const DUP_COSINE_THRESHOLD = 0.9;

/** 轻量版写前查重：返回与输入余弦最接近且 ≥ 阈值的既有行；无命中/无 embedding → null。 */
async function findDuplicateRow(db, text, embedFn) {
	if (!embedFn) return null;
	try {
		const qv = await embedFn(text);
		if (!Array.isArray(qv) || qv.length === 0) return null;
		const eRows = db.prepare("SELECT memory_id, dim, vector FROM memory_embeddings").all();
		let bestId = 0, bestC = 0;
		for (const e of eRows) {
			let arr;
			try { arr = JSON.parse(e.vector); } catch { continue; }
			if (!Array.isArray(arr) || arr.length !== qv.length) continue;
			const c = cosine(qv, arr);
			if (c > bestC) { bestC = c; bestId = Number(e.memory_id); }
		}
		if (!bestId || bestC < DUP_COSINE_THRESHOLD) return null;
		return db.prepare("SELECT id, text, category, created_at FROM memories WHERE id = ?").get(bestId) ?? null;
	} catch {
		return null;
	}
}

/** 写一条记忆；若带 embedFn 且成功则同时写向量。P1：先查重，命中高相似既有行 → 合并更新（merged=true）。 */
async function memoryAdd(db, text, category = "", embedFn) {
	const dup = await findDuplicateRow(db, text, embedFn);
	if (dup) {
		const mergedCategory = String(category || "").trim() || dup.category || "";
		db.prepare("UPDATE memories SET text = ?, category = ?, updated_at = datetime('now') WHERE id = ?").run(text, mergedCategory, dup.id);
		if (embedFn) {
			try {
				const vec = await embedFn(text);
				db.prepare("INSERT OR REPLACE INTO memory_embeddings (memory_id, dim, vector) VALUES (?, ?, ?)").run(dup.id, vec.length, JSON.stringify(vec));
			} catch {
				// embedding 失败不阻塞：既有向量仍可召回
			}
		}
		const row = db.prepare("SELECT id, text, category, created_at FROM memories WHERE id = ?").get(dup.id);
		return {
			id: dup.id,
			text: row?.text ?? text,
			category: row?.category ?? mergedCategory,
			created_at: row?.created_at ?? dup.created_at ?? "",
			merged: true,
		};
	}
	const info = db.prepare("INSERT INTO memories (text, category) VALUES (?, ?)").run(text, category);
	const id = Number(info.lastInsertRowid);
	if (embedFn) {
		try {
			const vec = await embedFn(text);
			db.prepare("INSERT OR REPLACE INTO memory_embeddings (memory_id, dim, vector) VALUES (?, ?, ?)").run(id, vec.length, JSON.stringify(vec));
		} catch {
			// embedding 失败不阻塞：仍可按关键词召回
		}
	}
	const row = db.prepare("SELECT id, text, category, created_at FROM memories WHERE id = ?").get(id);
	return {
		id,
		text: row?.text ?? text,
		category: row?.category ?? category,
		created_at: row?.created_at ?? "",
		merged: false,
	};
}

/** 中文 bigram 词集（最多 64 个）。 */
function memoryBigrams(s) {
	const out = [];
	for (let i = 0; i < s.length - 1; i += 1) out.push(s.slice(i, i + 2));
	return out.slice(0, 64);
}

/** 检索记忆：语义（余弦）优先 + 关键词加权做 tiebreak。embedFn 失败则纯关键词。 */
async function memorySearch(db, query, limit = 5, embedFn) {
	const q = (query || "").trim().toLowerCase();
	if (!q) return [];
	const terms = new Set();
	for (const w of q.split(/[\s,，。、；;:：!！?？]+/).filter(Boolean)) terms.add(w);
	if (q.length >= 2) for (const b of memoryBigrams(q)) terms.add(b);
	const rows = db.prepare("SELECT id, text, category, created_at FROM memories").all();
	const keyScored = new Map();
	for (const r of rows) {
		const t = (r.text || "").toLowerCase();
		let score = 0;
		for (const term of terms) if (t.includes(term)) score += Math.max(2, term.length);
		if (score > 0) keyScored.set(r.id, score);
	}
	// 语义：余弦相似度（若带 embedding 且成功）
	let vecResults = [];
	if (embedFn) {
		try {
			const qv = await embedFn(query);
			const eRows = db.prepare("SELECT memory_id, dim, vector FROM memory_embeddings").all();
			for (const e of eRows) {
				let arr; try { arr = JSON.parse(e.vector); } catch { continue; }
				if (!Array.isArray(arr)) continue;
				const c = cosine(qv, arr);
				if (c > 0.05) vecResults.push({ id: e.memory_id, c });
			}
			vecResults.sort((a, b) => b.c - a.c);
		} catch {
			// 语义失败则纯关键词
		}
	}
	// 混合：语义优先 + 关键词加权
	const byId = new Map(rows.map((r) => [r.id, r]));
	const scored = [];
	const seen = new Set();
	for (const v of vecResults) {
		const r = byId.get(v.id);
		if (!r) continue;
		const k = keyScored.get(v.id) || 0;
		scored.push({ ...r, c: v.c, score: v.c * 100 + k });
		seen.add(v.id);
	}
	for (const [id, k] of keyScored) {
		if (seen.has(id)) continue;
		const r = byId.get(id);
		if (r) scored.push({ ...r, c: 0, score: k });
	}
	scored.sort((a, b) => b.score - a.score || String(b.created_at).localeCompare(String(a.created_at)));
	return scored.slice(0, limit).map((r) => ({ id: r.id, text: r.text, category: r.category, created_at: r.created_at }));
}

/**
 * 插件的 apply：在 DSH 树启动时注册 memory_add / memory_search。
 * @param ctx - cordis 插件上下文（本插件只取 tools 服务）。
 * @param config - `{ memoryDbPath?, zhipuEnvPath?, enabled? }`，全部可选。
 */
function apply(ctx, config = {}) {
	// 红线：任何错误都不得让 DSH 树挂掉。
	try {
		const tools = ctx.get("tools");
		if (!tools || typeof tools.register !== "function") {
			ctx.logger?.warn?.("dsh-memory: tools 服务不可用，跳过记忆工具注册");
			return;
		}
		if (config.enabled === false) {
			ctx.logger?.info?.("dsh-memory: enabled=false，未注册");
			return;
		}

		const cfg = {
			memoryDbPath: config.memoryDbPath ?? join(process.env.HOME || "/home/ubuntu", ".dsh-memory/memories.db"),
			zhipuEnvPath: config.zhipuEnvPath ?? join(process.env.HOME || "/home/ubuntu", ".dsh-memory/.memenv"),
			semantic: config.semantic !== false,
		};

		// 打开/建库。失败 → 记日志并放弃，绝不崩。
		let memDb;
		try {
			memDb = ensureMemoryDb(cfg.memoryDbPath);
		} catch (err) {
			ctx.logger?.warn?.(`dsh-memory: 打开记忆库失败 ${err instanceof Error ? err.message : String(err)}；跳过`);
			return;
		}
		seedInitialMemories(memDb);

		// 智谱 key 可选：有则做语义召回；无则纯关键词。
		let embedFn = null;
		if (cfg.semantic) {
			const zhipu = getZhipuCreds(cfg);
			if (zhipu.key) embedFn = (t) => zhipuEmbed(t, zhipu);
		}

		const disposers = [];
		const ok = (fn, fallback) => {
			try { return fn(); } catch { return fallback; }
		};

		// 强制记忆信号词（用户自定义）。命中即要求 agent 调用 memory_add。
		// 默认留空；配置示例见 README（如 ["记住","重要","珍贵","特别","务必","一定","必须"]）。
		const forceWords = Array.isArray(config.forceMemoryWords)
			? config.forceMemoryWords.map((w) => String(w).trim()).filter(Boolean)
			: [];
		const addDescription = forceWords.length > 0
			? `把一条值得长期记住的事实写入跨会话记忆。当用户告诉你一件以后应当记住的事（姓名、住址、喜好、约定、重要背景）时调用；一般性闲聊不要调用。\n强制信号：用户消息中包含「${forceWords.join("、")}」任一词语时，视为用户明确要求记住，必须调用本工具。`
			: "把一条值得长期记住的事实写入跨会话记忆。当用户告诉你一件以后应当记住的事（姓名、住址、喜好、约定、重要背景）时调用；一般性闲聊不要调用。";

		disposers.push(ok(() => tools.register(
			defineTool({
				name: "memory_add",
				description: addDescription + " 写入前会自动做语义查重：与既有记忆语义高度相似的条目会合并更新既有行（保留最新措辞、刷新 updated_at），不会新增重复。纠错约定：新事实与旧记忆冲突或旧记忆已过时，直接按新事实再记（新者胜），无需删除旧条目。",
				parameters: {
					text: { type: "string", required: true, description: "要记住的事实，写成一句完整、独立的中文陈述。" },
					category: { type: "string", description: "分类，如 画像/偏好/约定/事实（可空）。" },
				},
				output: {
					schema: {
						type: "object",
						additionalProperties: false,
						properties: { id: { type: "integer" }, text: { type: "string" }, category: { type: "string" }, created_at: { type: "string" }, merged: { type: "boolean" } },
					},
					render: (_args, value) => [{ type: "text", text: value.merged ? `已记住（合并更新既有条目）：${value.text}` : `已记住：${value.text}` }],
				},
				async execute(args) {
					// 绝不让工具执行错误冒泡成会话异常。
					try {
						const t = String(args.text || "").trim();
						if (!t) return { id: 0, text: "", category: "", created_at: "", error: "text 为空" };
						return await memoryAdd(memDb, t, String(args.category || ""), embedFn);
					} catch (err) {
						return { id: 0, text: "", category: "", created_at: "", error: err instanceof Error ? err.message : String(err) };
					}
				},
			})
		), null));

		disposers.push(ok(() => tools.register(
			defineTool({
				name: "memory_search",
				description:
					"在跨会话长期记忆里检索与用户个人背景相关的事实（姓名、住址、喜好、约定、此前聊过的事）。回答任何涉及用户个人信息/背景的问题前，先调用它回忆，不要凭空猜测或编造。",
				parameters: {
					query: { type: "string", required: true, description: "要回忆的关键词或问题，尽量具体。" },
					limit: { type: "integer", description: "最多返回几条（默认 5，最大 20）。" },
				},
				output: {
					schema: {
						type: "object",
						additionalProperties: false,
						properties: {
							results: {
								type: "array",
								items: {
									type: "object",
									additionalProperties: true,
									properties: { id: { type: "integer" }, text: { type: "string" }, category: { type: "string" }, created_at: { type: "string" } },
								},
							},
						},
					},
					render: (_args, value) => [
						{
							type: "text",
							text: value.results?.length
								? value.results.map((r) => `- ${r.text}${r.category ? `（${r.category}）` : ""}`).join("\n")
								: "（记忆里没有相关条目）",
						},
					],
				},
				async execute(args) {
					try {
						const limit = Number.isInteger(args.limit) ? Math.max(1, Math.min(20, args.limit)) : 5;
						return { results: await memorySearch(memDb, String(args.query || ""), limit, embedFn) };
					} catch (err) {
						return { results: [], error: err instanceof Error ? err.message : String(err) };
					}
				},
			})
		), null));

		// /mem 用法：不注册 slash command（否则前端会拦截 /mem 并吃掉输入）。
		// 改为"普通消息 + agent 按指令调用 memory_search"：
		//   用户在输入框写  /mem <问题>  →  前端因无此命令而当作普通消息发给 agent
		//   →  agent 读取系统指令（dsh-memory:recall-hint）→ 调用 memory_search(query=<问题>)
		//   →  检索结果经 DSH 工具卡片渲染（可折叠，同 bash/think）→ agent 再基于其结果回答。
		//   这样既知道用户原问题，检索结果也以工具卡片形式展示。

		// 强制记忆信号词 → 注入 agent 系统指令（命中即先 memory_add 再回答）。
		// 运行时探测 systemPrompt 服务（不写进 inject，避免在无该服务的 profile 里崩）。
		if (forceWords.length > 0) {
			disposers.push(ok(() => {
				const sp = ctx.get("systemPrompt");
				if (!sp || typeof sp.section !== "function") {
					ctx.logger?.warn?.("dsh-memory: systemPrompt 服务不可用，强制记忆信号词指令未注入");
					return null;
				}
				return sp.section({
					name: "dsh-memory:force-recall",
					order: 60,
					text:
						`强制记忆信号词：用户消息中出现以下任一词语（${forceWords.join("、")}）时，` +
						"视为用户明确要求记住该内容。此时必须先调用 memory_add 把相关事实写入跨会话记忆" +
						"（category 可用 事实/偏好/约定），再继续回答；不要因消息是闲聊语气而跳过。",
				});
			}, null));
		}

		// 通用读取提示：回答可能涉及跨会话记忆的问题前，先 memory_search。
		// 默认开启（recallHint: true），可用 config.recallHint=false 关闭。
		if (config.recallHint !== false) {
			disposers.push(ok(() => {
				const sp = ctx.get("systemPrompt");
				if (!sp || typeof sp.section !== "function") return null;
				return sp.section({
					name: "dsh-memory:recall-hint",
					order: 61,
					text:
						"你拥有跨会话长期记忆工具 memory_search。规则：\n" +
						"1) 用户消息以 /mem 开头时，视为强制记忆检索：把 /mem 后的文本作为 query 调用 memory_search，" +
						"再基于检索结果回答用户；不要把 /mem 当普通文本，也不要漏掉用户的原问题。\n" +
						"2) 回答任何可能涉及过往会话内容的问题（项目历史、架构决策、个人背景、过往约定、踩坑教训、之前聊过的事）前，" +
						"先调用 memory_search 检索，基于检索结果回答；不要凭空猜测或编造。",
				});
			}, null));
		}

		// 卸载清理。整个插件生命周期内共享同一 DB 句柄。
		ctx.effect(
			() => () => {
				for (const d of disposers) { try { d && d(); } catch {} }
				try { memDb.close(); } catch {}
			},
			"dsh-memory"
		);

		ctx.logger?.info?.(`dsh-memory: 记忆工具就绪（semantic=${Boolean(embedFn)}，db=${cfg.memoryDbPath}）`);
	} catch (err) {
		// 最后一道防线：绝不让 DSH 树因本插件崩溃。
		ctx.logger?.warn?.(`dsh-memory: apply 失败 ${err instanceof Error ? err.message : String(err)}`);
	}
}

export { apply, inject, name };
