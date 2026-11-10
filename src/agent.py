"""企业知识问答 Agent：模型选择动作 → 执行工具 → 观察结果 → 模型决定下一步或结束。

运行：
    python -m src.agent "报销期限多久？"                 # 真实模型（读 .env）
    python -m src.agent "报销期限多久？" --mock          # 离线 Mock，仅验证流程
    python -m src.agent --all-cases                      # 跑 examples/ 下全部用例，写 Trace
    python -m src.agent --chat                           # 多轮对话（支持上下文追问）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .knowledge import KnowledgeBase
from .model import ModelError, ModelReply, build_model
from .tools import FINAL_STATUSES, FaultInjector, ToolBox, summarize_observation

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
TRACES_DIR = EXAMPLES_DIR / "traces"

SYSTEM_PROMPT = """你是企业知识问答助手，帮助员工从公司制度和产品 FAQ 中找到可靠答案。你只能依据工具返回的文档内容回答。

工作方式（每一步都由你根据上一步的工具结果决定）：
1. 先调用 search_docs 检索。追问类问题（如“那…呢”）要结合对话历史补全主语和关键词再检索。
2. 无命中或相关度低：换关键词、同义词或更短的说法补充检索（最多改写 2 次）；仍无结果就以 insufficient 结束，说明资料缺口，绝不编造。
3. 有命中：用 read_doc 读取原文核对，不能只凭摘要回答。多个文档都涉及同一问题时，全部读取并比较内容、版本日期、适用范围、是否声明替代关系。
4. 文档内容不一致：若文档明确写了优先级/替代关系/适用范围，按其说明作答并解释依据；若没有说明，以 conflict 结束，列出每份文档的说法和引用，并在 clarifying_question 中请用户确认适用制度。不得擅自选择其中一个（包括“选最新的”）。
5. 工具返回 ok=false：可以重试一次或换参数；仍失败则以 tool_failure 结束，说明哪个工具失败、缺什么信息。
6. 最后必须调用 final_answer。citations 的 quote 必须逐字复制自 read_doc 返回的原文，doc_id 必须是本轮工具返回过的文档。
7. 每次调用工具都在 reason 里用一句话写明决策依据。
"""

FEEDBACK_PREFIX = "【校验反馈】"


@dataclass
class AgentConfig:
    max_steps: int = 8          # 模型决策轮数上限
    max_tool_calls: int = 10    # 工具调用次数上限（不含 final_answer）
    max_validation_retries: int = 1


class QAAgent:
    def __init__(self, model, toolbox: ToolBox, config: AgentConfig | None = None):
        self.model = model
        self.toolbox = toolbox
        self.kb = toolbox.kb
        self.config = config or AgentConfig()

    # ------------------------------------------------------------------ 主循环
    def run(self, question: str, history: list[dict] | None = None) -> dict:
        started = time.time()
        history = history or []
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]
        messages.append({"role": "user", "content": question})

        steps: list[dict] = []
        observed_docs: set[str] = set()
        tool_calls_used = 0
        validation_retries = 0
        usage_total: dict[str, int] = {}
        result: dict | None = None

        for step_no in range(1, self.config.max_steps + 1):
            try:
                reply: ModelReply = self.model.chat(messages, self.toolbox.schemas)
            except ModelError as exc:
                steps.append({"step": step_no, "action": "model_error", "ok": False, "observation": str(exc)})
                result = self._terminal("incomplete", f"模型调用失败，任务未完成：{exc}", [f"模型不可用：{exc}"])
                break
            for k, v in reply.usage.items():
                if isinstance(v, int):
                    usage_total[k] = usage_total.get(k, 0) + v

            if not reply.tool_calls:
                # 模型直接输出文本而不调用工具：不接受为答案，提醒它走 final_answer
                messages.append({"role": "assistant", "content": reply.content})
                messages.append({"role": "user", "content": f"{FEEDBACK_PREFIX}请通过工具检索，并用 final_answer 结束。"})
                steps.append({"step": step_no, "action": "text_without_tool", "ok": False,
                              "observation": f"模型未调用工具，输出被拒收：{reply.content[:80]}"})
                continue

            messages.append({
                "role": "assistant", "content": reply.content or None,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name,
                                             "arguments": tc.raw_arguments or json.dumps(tc.arguments, ensure_ascii=False)}}
                               for tc in reply.tool_calls]})

            feedback: str | None = None
            for tc in reply.tool_calls:
                args = dict(tc.arguments)
                reason = args.pop("reason", "")
                if result is not None:
                    # 同一轮里 final_answer 之后的调用不再执行，但要回 tool 消息保持协议完整
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps({"ok": False, "error": "任务已结束"}, ensure_ascii=False)})
                    continue

                if tc.name == "final_answer":
                    errors = self._validate_final(args, observed_docs)
                    step = {"step": step_no, "action": "final_answer", "reason": reason,
                            "args": {"status": args.get("status"),
                                     "citations": [c.get("doc_id") for c in args.get("citations") or []]}}
                    if not errors:
                        step.update(ok=True, observation="校验通过，结束")
                        result = self._build_result(args)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": '{"ok": true}'})
                    elif validation_retries < self.config.max_validation_retries:
                        validation_retries += 1
                        step.update(ok=False, observation="校验未通过：" + "；".join(errors))
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": json.dumps({"ok": False, "errors": errors}, ensure_ascii=False)})
                        feedback = f"{FEEDBACK_PREFIX}final_answer 未通过校验：{'；'.join(errors)}。请修正（必要时重新读取原文）后再提交。"
                    else:
                        step.update(ok=False, observation="校验再次未通过，降级处理：" + "；".join(errors))
                        result = self._downgrade(args, observed_docs, errors)
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": json.dumps({"ok": False, "errors": errors}, ensure_ascii=False)})
                    steps.append(step)
                    continue

                if tool_calls_used >= self.config.max_tool_calls:
                    obs = {"ok": False, "error": f"工具调用次数已达上限 {self.config.max_tool_calls}，请用 final_answer 结束"}
                else:
                    tool_calls_used += 1
                    obs = self.toolbox.execute(tc.name, args)
                    self._track_docs(tc.name, obs, observed_docs)
                steps.append({"step": step_no, "action": tc.name, "args": args, "reason": reason,
                              "ok": obs.get("ok", False), "observation": summarize_observation(tc.name, obs)})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs, ensure_ascii=False)})

            if result is not None:
                break
            if feedback:
                messages.append({"role": "user", "content": feedback})

        if result is None:
            result = self._terminal("incomplete", f"已达到最大执行步数 {self.config.max_steps}，任务未完成。",
                                    ["执行步数耗尽，未得到经校验的答案"])

        for i, s in enumerate(steps):
            s["next_action"] = steps[i + 1]["action"] if i + 1 < len(steps) else "结束"
        result["steps_used"] = len({s["step"] for s in steps})
        result["tool_calls_used"] = tool_calls_used
        trace = {
            "run_id": uuid.uuid4().hex[:12],
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "model": getattr(self.model, "name", "unknown"),
            "mode": getattr(self.model, "mode", "unknown"),
            "question": question,
            "history": history,
            "limits": {"max_steps": self.config.max_steps, "max_tool_calls": self.config.max_tool_calls},
            "steps": steps,
            "result": result,
            "usage": usage_total,
            "duration_ms": int((time.time() - started) * 1000),
        }
        return {"result": result, "trace": trace}

    # ------------------------------------------------------------------ 辅助
    @staticmethod
    def _track_docs(name: str, obs: dict, observed: set[str]) -> None:
        if not obs.get("ok"):
            return
        if name == "read_doc":
            observed.add(obs["doc_id"])

    def _validate_final(self, args: dict, observed_docs: set[str]) -> list[str]:
        errors: list[str] = []
        status = args.get("status")
        if status not in FINAL_STATUSES:
            errors.append(f"status 必须是 {FINAL_STATUSES} 之一")
        if not str(args.get("answer", "")).strip():
            errors.append("answer 不能为空")
        valid_docs = set()
        for c in args.get("citations") or []:
            doc_id, quote = c.get("doc_id", ""), str(c.get("quote", ""))
            if doc_id not in self.kb.docs:
                errors.append(f"引用的文档 {doc_id} 不存在")
            elif doc_id not in observed_docs:
                errors.append(f"引用的文档 {doc_id} 本轮未通过 read_doc 读取原文")
            elif len(quote.strip()) < 4 or not self.kb.quote_in_doc(doc_id, quote):
                errors.append(f"{doc_id} 的引用“{quote[:30]}”不是原文")
            else:
                valid_docs.add(doc_id)
        if status == "answered" and not valid_docs:
            errors.append("answered 必须至少有一条经校验的原文引用")
        if status == "conflict" and len(valid_docs) < 2:
            errors.append("conflict 必须引用至少两份不同文档的原文")
        return errors

    def _build_result(self, args: dict) -> dict:
        citations = []
        for c in args.get("citations") or []:
            doc = self.kb.docs[c["doc_id"]]
            citations.append({"doc_id": doc.doc_id, "chunk_id": c.get("chunk_id", ""), "title": doc.title,
                              "version": doc.version, "quote": c["quote"]})
        return {
            "status": args["status"],
            "answer": args["answer"],
            "citations": citations,
            "conflicts": args.get("conflicts") or [],
            "gaps": args.get("gaps") or [],
            "clarifying_question": args.get("clarifying_question", ""),
        }

    def _downgrade(self, args: dict, observed_docs: set[str], errors: list[str]) -> dict:
        """两次校验不过：只保留可校验的引用；没有可靠引用时不输出答案。"""
        good = [c for c in args.get("citations") or []
                if c.get("doc_id") in observed_docs and self.kb.quote_in_doc(c["doc_id"], str(c.get("quote", "")))]
        status = args.get("status")
        docs = {c["doc_id"] for c in good}
        ok = (status == "answered" and docs) or (status == "conflict" and len(docs) >= 2) \
            or status in ("insufficient", "tool_failure")
        if ok:
            res = self._build_result({**args, "citations": good})
            res["gaps"] = res["gaps"] + ["部分引用未通过原文校验，已剔除"]
            return res
        return self._terminal("insufficient", "未能给出有原文依据的答案，请人工确认。",
                              (args.get("gaps") or []) + ["模型答案未通过引用校验：" + "；".join(errors)])

    @staticmethod
    def _terminal(status: str, answer: str, gaps: list[str]) -> dict:
        return {"status": status, "answer": answer, "citations": [], "conflicts": [], "gaps": gaps,
                "clarifying_question": ""}


# ---------------------------------------------------------------------- 用例
def check_expectations(case: dict, result: dict) -> list[str]:
    """按用例 expect 字段核对实际输出，返回不满足的项。"""
    exp = case.get("expect", {})
    fails = []
    if "status" in exp and result["status"] != exp["status"]:
        fails.append(f"status 期望 {exp['status']}，实际 {result['status']}")
    cited = {c["doc_id"] for c in result["citations"]}
    for d in exp.get("must_cite", []):
        if d not in cited:
            fails.append(f"缺少对 {d} 的引用")
    for d in exp.get("must_not_cite", []):
        if d in cited:
            fails.append(f"不应引用 {d}")
    for kw in exp.get("answer_contains", []):
        import re
        clean_answer = re.sub(r'[^\w\s]', '', result["answer"])
        clean_kw = re.sub(r'[^\w\s]', '', kw)
        if clean_kw not in clean_answer:
            fails.append(f"答案应包含“{kw}”")
    return fails


def run_case(case_path: Path, mock: bool, data_dir: Path | None = None, write_trace: bool = True) -> dict:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    kb = KnowledgeBase(data_dir) if data_dir else KnowledgeBase()
    agent = QAAgent(build_model(mock), ToolBox(kb, FaultInjector(case.get("fail_tools"))),
                    AgentConfig(**case.get("config", {})))
    out = agent.run(case["question"], case.get("history"))
    out["trace"]["case_id"] = case["id"]
    out["trace"]["scenario"] = case.get("scenario", "")
    out["trace"]["fail_tools"] = case.get("fail_tools", [])
    fails = check_expectations(case, out["result"])
    out["trace"]["expectation_check"] = {"passed": not fails, "failures": fails}
    if write_trace:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACES_DIR / f"{case['id']}.{out['trace']['mode']}.json"
        path.write_text(json.dumps(out["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
        out["trace_path"] = str(path.relative_to(ROOT))
    return out


def print_result(out: dict, verbose: bool = True) -> None:
    trace, res = out["trace"], out["result"]
    if verbose:
        print(f"[模式] {trace['mode']}  模型={trace['model']}  步数上限={trace['limits']['max_steps']}")
        for s in trace["steps"]:
            mark = "✓" if s.get("ok") else "✗"
            print(f"  step{s['step']} {mark} {s['action']} {json.dumps(s.get('args', {}), ensure_ascii=False)}")
            if s.get("reason"):
                print(f"         依据: {s['reason']}")
            print(f"         返回: {s['observation']}")
    print(f"\n[状态] {res['status']}\n[回答] {res['answer']}")
    for c in res["citations"]:
        print(f"[引用] {c['doc_id']} {c['chunk_id']}《{c['title']}》v{c['version']}：“{c['quote']}”")
    for c in res["conflicts"]:
        print(f"[冲突] {c['topic']}: " + " vs ".join(f"{x['doc_id']}={x['claim']}" for x in c["claims"]))
    for g in res["gaps"]:
        print(f"[缺口] {g}")
    if res.get("clarifying_question"):
        print(f"[需确认] {res['clarifying_question']}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="企业知识问答 Agent")
    p.add_argument("question", nargs="?", help="要问的问题")
    p.add_argument("--mock", action="store_true", help="离线 Mock 模型（只验证流程，不代表真实模型效果）")
    p.add_argument("--history", help="对话历史 JSON 文件：[{role, content}, ...]")
    p.add_argument("--fail-tool", action="append", default=[], help="注入工具故障，如 search_docs 或 search_docs:1")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-tool-calls", type=int, default=10)
    p.add_argument("--data-dir", help="替换知识库目录（用于替换文档验证）")
    p.add_argument("--case", help="运行单个用例 JSON")
    p.add_argument("--all-cases", action="store_true", help="运行 examples/case_*.json 并写入 examples/traces/")
    p.add_argument("--chat", action="store_true", help="多轮对话模式")
    p.add_argument("--trace-out", help="把 Trace 写到指定文件")
    p.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir) if a.data_dir else None

    try:
        if a.all_cases or a.case:
            paths = [Path(a.case)] if a.case else sorted(EXAMPLES_DIR.glob("case_*.json"))
            summary = []
            for path in paths:
                out = run_case(path, a.mock, data_dir)
                chk = out["trace"]["expectation_check"]
                print(f"\n===== {out['trace']['case_id']} ({out['trace']['scenario']}) → {out['trace_path']}")
                print_result(out)
                print(f"[期望核对] {'PASS' if chk['passed'] else 'FAIL ' + '；'.join(chk['failures'])}")
                summary.append({"case_id": out["trace"]["case_id"], "status": out["result"]["status"],
                                "passed": chk["passed"], "failures": chk["failures"],
                                "steps": [s["action"] for s in out["trace"]["steps"]]})
            if a.all_cases:
                mode = "mock" if a.mock else "real"
                (TRACES_DIR / f"summary.{mode}.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                passed = sum(s["passed"] for s in summary)
                print(f"\n合计 {passed}/{len(summary)} 通过，汇总见 examples/traces/summary.{mode}.json")
                return 0 if passed == len(summary) else 1
            return 0

        kb = KnowledgeBase(data_dir) if data_dir else KnowledgeBase()
        agent = QAAgent(build_model(a.mock), ToolBox(kb, FaultInjector(a.fail_tool)),
                        AgentConfig(max_steps=a.max_steps, max_tool_calls=a.max_tool_calls))

        if a.chat:
            history: list[dict] = []
            print("多轮对话模式，输入空行退出。")
            while True:
                q = input("\n你: ").strip()
                if not q:
                    return 0
                out = agent.run(q, history)
                print_result(out)
                history += [{"role": "user", "content": q},
                            {"role": "assistant", "content": out["result"]["answer"]}]

        if not a.question:
            p.error("请提供问题，或使用 --chat / --all-cases")
        history = json.loads(Path(a.history).read_text(encoding="utf-8")) if a.history else []
        out = agent.run(a.question, history)
        if a.trace_out:
            Path(a.trace_out).write_text(json.dumps(out["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
        if a.json:
            print(json.dumps(out["result"], ensure_ascii=False, indent=2))
        else:
            print_result(out)
        return 0
    except ModelError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
