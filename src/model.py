"""模型接入。

- OpenAICompatibleModel：调用任意 OpenAI 兼容的 /chat/completions（OpenAI、DeepSeek、通义千问 DashScope 兼容模式、
  Moonshot、本地 vLLM/Ollama 等），使用原生 function calling。只用标准库 urllib，不依赖 SDK。
- MockModel：离线规则策略，只用于在没有 API Key 时验证 Agent 循环、工具、Trace、校验逻辑是否通畅。
  它的“决策”是写死的启发式规则，不能证明真实模型的决策效果。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .knowledge import tokenize


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class ModelReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ModelError(Exception):
    pass


def load_env(path: Path | str = Path(__file__).resolve().parent.parent / ".env") -> None:
    """极简 .env 加载：KEY=VALUE，已存在的环境变量不覆盖。"""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class OpenAICompatibleModel:
    mode = "real"

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0,
                 timeout: float = 60.0, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        payload = {"model": self.name, "messages": messages, "tools": tools,
                   "tool_choice": "auto", "temperature": self.temperature}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_err = ModelError(f"HTTP {exc.code}: {detail}")
                if exc.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = ModelError(f"模型请求失败：{exc}")
            time.sleep(1.5 * (attempt + 1))
        raise last_err or ModelError("模型请求失败")

    @staticmethod
    def _parse(data: dict) -> ModelReply:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise ModelError(f"模型返回格式异常：{str(data)[:200]}") from exc
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"__invalid_json__": raw}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args, raw))
        return ModelReply(msg.get("content") or "", calls, data.get("usage") or {})


# ---------------------------------------------------------------------------
# 离线 Mock：根据对话里的工具结果做启发式决策，只验证流程
# ---------------------------------------------------------------------------

_FOLLOWUP_PREFIX = ("那", "这", "它", "还", "如果", "另外", "那么")
_QUESTION_WORDS = ("请问", "一下", "什么", "多少", "怎么", "如何", "是否", "可以", "吗", "呢", "？", "?", "，", "。")
_NUM_CLAIM_RE = re.compile(r"(\d+)\s*(个工作日|天|日)")


class MockModel:
    mode = "mock"
    name = "mock-policy"

    def __init__(self):
        self._n = 0

    def _call(self, name: str, **args) -> ModelReply:
        self._n += 1
        return ModelReply(tool_calls=[ToolCall(f"mock_{self._n}", name, args, json.dumps(args, ensure_ascii=False))])

    @staticmethod
    def _split(messages: list[dict]):
        """拆出：历史用户问题、当前问题、本轮工具观察 [(name, args, result)]。"""
        last_q_idx = max(i for i, m in enumerate(messages)
                         if m["role"] == "user" and not str(m.get("content", "")).startswith("【校验反馈】"))
        question = messages[last_q_idx]["content"]
        history_qs = [m["content"] for m in messages[:last_q_idx] if m["role"] == "user"]
        pending: dict[str, tuple[str, dict]] = {}
        obs = []
        for m in messages[last_q_idx + 1:]:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    pending[tc["id"]] = (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
            elif m["role"] == "tool" and m["tool_call_id"] in pending:
                name, args = pending[m["tool_call_id"]]
                obs.append((name, args, json.loads(m["content"])))
        return history_qs, question, obs

    @staticmethod
    def _build_query(question: str, history_qs: list[str]) -> tuple[str, str]:
        if history_qs and (question.startswith(_FOLLOWUP_PREFIX) or len(question) <= 8):
            return f"{history_qs[-1]} {question}", "问题是对上一轮的追问，合并上一轮问题补全检索语句"
        return question, "先用原问题检索知识库"

    @staticmethod
    def _keywords(question: str) -> str:
        q = question
        for w in _QUESTION_WORDS:
            q = q.replace(w, " ")
        return " ".join(q.split())

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        history_qs, question, obs = self._split(messages)
        if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("【校验反馈】"):
            return self._call("final_answer", status="insufficient", answer="引用未通过校验，无法给出可靠答案。",
                              citations=[], gaps=["引用校验失败"], reason="校验反馈指出引用有问题，保守结束")

        searches = [o for o in obs if o[0] == "search_docs"]
        reads = [o for o in obs if o[0] == "read_doc"]

        if not searches:
            query, reason = self._build_query(question, history_qs)
            return self._call("search_docs", query=query, reason=reason)

        name, args, result = obs[-1]
        if name == "search_docs":
            if not result["ok"]:
                if sum(1 for s in searches if not s[2]["ok"]) < 2:
                    return self._call("search_docs", query=args["query"], reason="检索工具返回失败，重试一次")
                return self._final_failure("search_docs", result["error"])
            if not result["hits"]:
                if len(searches) < 2:
                    return self._call("search_docs", query=self._keywords(args["query"]),
                                      reason="原问题无命中，去掉疑问词后用关键词补充检索")
                return self._call("final_answer", status="insufficient",
                                  answer="知识库中没有找到与该问题相关的制度或 FAQ，无法回答，建议咨询相关业务部门。",
                                  citations=[], gaps=[f"知识库未覆盖：{question}"],
                                  reason="两次检索（原句+关键词）均无命中，判定为资料外问题，不编造")

        hits = next(s[2]["hits"] for s in reversed(searches) if s[2]["ok"] and s[2]["hits"])
        top = hits[0]["score"]
        candidates = []
        for h in hits:
            if h["score"] >= 0.6 * top and h["doc_id"] not in candidates:
                candidates.append(h["doc_id"])
        read_ok = {r[1]["doc_id"]: r[2] for r in reads if r[2]["ok"]}
        read_fail = [r for r in reads if not r[2]["ok"]]
        if name == "read_doc" and not result["ok"]:
            if len(read_fail) < 2:
                return self._call("read_doc", doc_id=args["doc_id"], reason="读取原文失败，重试一次")
            return self._final_failure("read_doc", result["error"])
        for doc_id in candidates[:3]:
            if doc_id not in read_ok:
                reason = ("检索到相关片段，读取原文核对" if not read_ok
                          else f"还有文档 {doc_id} 也涉及该问题，读取比对是否一致")
                return self._call("read_doc", doc_id=doc_id, reason=reason)
        return self._conclude(question, hits, read_ok)

    def _final_failure(self, tool: str, error: str) -> ModelReply:
        return self._call("final_answer", status="tool_failure",
                          answer=f"知识库工具 {tool} 连续失败，本次无法查询制度原文，请稍后重试或联系 IT。",
                          citations=[], gaps=[f"{tool} 失败：{error}"],
                          reason="工具重试后仍失败，如实返回失败状态，不凭记忆作答")

    @staticmethod
    def _best_sentence(chunk_text: str, question: str) -> str:
        # 只按句号切：分号连接的分档规则（如年假 5/10/15 天）要整句引用，Mock 不做“12 年落在哪一档”的推理
        sentences = [s for s in re.split(r"(?<=[。])", chunk_text) if s.strip()]
        q_tokens = set(tokenize(question))
        return max(sentences, key=lambda s: len(q_tokens & set(tokenize(s))))

    def _conclude(self, question: str, hits: list[dict], read_ok: dict) -> ModelReply:
        # 每个已读文档取最相关的命中片段
        best: dict[str, dict] = {}
        for h in hits:
            if h["doc_id"] in read_ok and h["doc_id"] not in best:
                best[h["doc_id"]] = h
        evidence = []
        for doc_id, h in best.items():
            doc = read_ok[doc_id]
            text = next(c["text"] for c in doc["chunks"] if c["chunk_id"] == h["chunk_id"])
            quote = self._best_sentence(text, question)
            evidence.append({"doc_id": doc_id, "chunk_id": h["chunk_id"], "quote": quote, "doc": doc})

        claims = {e["doc_id"]: set(_NUM_CLAIM_RE.findall(e["quote"])) for e in evidence}
        with_nums = {k: v for k, v in claims.items() if v}
        if len(with_nums) >= 2 and len({frozenset(v) for v in with_nums.values()}) > 1:
            conflict_ev = [e for e in evidence if e["doc_id"] in with_nums]
            desc = "；".join(f"{e['doc_id']}《{e['doc']['title']}》（{e['doc']['version']}，适用范围：{e['doc']['scope']}）"
                            f"规定“{e['quote']}”" for e in conflict_ev)
            return self._call(
                "final_answer", status="conflict",
                answer=f"文档之间存在冲突：{desc}。文档未说明适用范围或优先级，无法确定以哪份为准。",
                citations=[{"doc_id": e["doc_id"], "chunk_id": e["chunk_id"], "quote": e["quote"]} for e in conflict_ev],
                conflicts=[{"topic": question, "claims": [{"doc_id": e["doc_id"], "claim": e["quote"]} for e in conflict_ev]}],
                gaps=["文档之间的适用范围与优先级未说明"],
                clarifying_question="请确认您适用哪份制度，或向财务部确认最新口径。",
                reason="多份文档对同一问题给出不同数值，且未说明优先级，按冲突处理")

        e = evidence[0]
        return self._call(
            "final_answer", status="answered",
            answer=f"根据 {e['doc_id']}《{e['doc']['title']}》（{e['doc']['version']}）：{e['quote']}",
            citations=[{"doc_id": e["doc_id"], "chunk_id": e["chunk_id"], "quote": e["quote"]}],
            gaps=[], reason="已读原文且无冲突，依据最相关片段作答")


def build_model(mock: bool = False):
    if mock:
        return MockModel()
    load_env()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    missing = [k for k, v in (("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not v]
    if missing:
        raise ModelError(f"缺少环境变量 {', '.join(missing)}（见 .env.example），或加 --mock 使用离线模式")
    return OpenAICompatibleModel(base_url, api_key, model,
                                 temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                                 timeout=float(os.getenv("LLM_TIMEOUT", "60")))
