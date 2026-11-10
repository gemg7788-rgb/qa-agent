"""Agent 可调用的工具：文档检索、原文读取，以及结束任务的 final_answer。

工具执行永不抛异常给 Agent 循环，统一返回 {"ok": bool, ...}，由模型看到失败结果后自己决定下一步。
FaultInjector 用于模拟工具故障（测试和 --fail-tool 参数）。
"""
from __future__ import annotations

from .knowledge import KnowledgeBase

FINAL_STATUSES = ["answered", "conflict", "insufficient", "tool_failure"]

REASON_PARAM = {"type": "string", "description": "一句话说明为什么执行这个动作（决策依据），会写入 Trace"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在企业制度/FAQ 知识库中检索，返回相关片段（文档ID、片段ID、版本日期、摘要、相关度）。"
                           "命中为空或相关度低时，可换关键词、同义词再检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索语句或关键词"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5", "minimum": 1, "maximum": 10},
                    "exclude_doc_ids": {"type": "array", "items": {"type": "string"},
                                        "description": "补充检索时排除已看过的文档"},
                    "reason": REASON_PARAM,
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "读取文档原文（含元数据：版本日期、发布部门、适用范围）。可指定 chunk_id 只读某个片段。"
                           "引用原文前必须先读取。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "chunk_id": {"type": "string", "description": "可选，只读该片段"},
                    "reason": REASON_PARAM,
                },
                "required": ["doc_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "结束任务并输出结构化答案。quote 必须逐字摘自 read_doc 返回的原文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": FINAL_STATUSES,
                               "description": "answered=有依据的明确答案；conflict=文档冲突需确认；"
                                              "insufficient=资料不足/资料外；tool_failure=工具失败无法完成"},
                    "answer": {"type": "string", "description": "给员工的回答；冲突或资料不足时说明情况"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "chunk_id": {"type": "string"},
                                "quote": {"type": "string", "description": "逐字原文片段"},
                            },
                            "required": ["doc_id", "quote"],
                        },
                    },
                    "conflicts": {
                        "type": "array",
                        "description": "status=conflict 时列出冲突点",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "claims": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {"doc_id": {"type": "string"}, "claim": {"type": "string"}},
                                    "required": ["doc_id", "claim"]}},
                            },
                            "required": ["topic", "claims"],
                        },
                    },
                    "gaps": {"type": "array", "items": {"type": "string"},
                             "description": "资料缺口：知识库里没有、需要人工确认的信息"},
                    "clarifying_question": {"type": "string", "description": "需要用户确认的问题（可选）"},
                    "reason": REASON_PARAM,
                },
                "required": ["status", "answer", "citations", "reason"],
            },
        },
    },
]


class ToolError(Exception):
    pass


class FaultInjector:
    """按工具名注入故障。spec 形如 "search_docs" （一直失败）或 "search_docs:1"（前 1 次失败）。"""

    def __init__(self, specs: list[str] | None = None):
        self.remaining: dict[str, int | None] = {}
        for spec in specs or []:
            name, _, count = spec.partition(":")
            self.remaining[name.strip()] = int(count) if count else None

    def check(self, tool_name: str) -> None:
        if tool_name not in self.remaining:
            return
        left = self.remaining[tool_name]
        if left is None:
            raise ToolError(f"{tool_name} 服务不可用（模拟故障：连接超时）")
        if left > 0:
            self.remaining[tool_name] = left - 1
            raise ToolError(f"{tool_name} 服务暂时不可用（模拟故障，剩余 {left - 1} 次）")


class ToolBox:
    def __init__(self, kb: KnowledgeBase, faults: FaultInjector | None = None):
        self.kb = kb
        self.faults = faults or FaultInjector()

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def execute(self, name: str, args: dict) -> dict:
        try:
            self.faults.check(name)
            if name == "search_docs":
                query = str(args.get("query", "")).strip()
                if not query:
                    raise ToolError("query 不能为空")
                top_k = int(args.get("top_k") or 5)
                result = self.kb.search(query, top_k=max(1, min(top_k, 10)),
                                        exclude_doc_ids=args.get("exclude_doc_ids"))
                return {"ok": True, **result}
            if name == "read_doc":
                doc_id = str(args.get("doc_id", "")).strip()
                return {"ok": True, **self.kb.read(doc_id, args.get("chunk_id") or None)}
            raise ToolError(f"未知工具 {name}")
        except (ToolError, KeyError, ValueError, TypeError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            return {"ok": False, "error": str(msg), "tool": name}


def summarize_observation(name: str, result: dict) -> str:
    """把工具返回压成一行，写进 Trace 的“返回摘要”。"""
    if not result.get("ok"):
        return f"失败：{result.get('error')}"
    if name == "search_docs":
        hits = result.get("hits", [])
        if not hits:
            return "无命中" + (f"（同义词扩展：{'、'.join(result['expanded_terms'])}）" if result.get("expanded_terms") else "")
        parts = [f"{h['chunk_id']}《{h['title']}》v{h['version']} score={h['score']}" for h in hits]
        return f"命中 {len(hits)} 条：" + "；".join(parts)
    if name == "read_doc":
        return (f"读取 {result['doc_id']}《{result['title']}》v{result['version']}，适用范围：{result['scope']}，"
                f"{len(result['chunks'])} 个片段")
    return "ok"
