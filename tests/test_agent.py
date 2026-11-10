"""可执行测试：python -m pytest -q

分三层：
1. 知识库/工具单元测试（检索、同义词、故障注入）
2. Agent 循环机制测试（用 ScriptedModel 精确控制模型输出，验证校验、上限、降级）
3. 端到端用例（离线 Mock 跑 examples/case_*.json）与替换文档测试
另有真实模型测试，需配置 LLM_* 环境变量，否则自动跳过。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from src.agent import AgentConfig, QAAgent, check_expectations, run_case
from src.knowledge import KnowledgeBase
from src.model import MockModel, ModelError, ModelReply, OpenAICompatibleModel, ToolCall, build_model
from src.tools import FaultInjector, ToolBox

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CASES = sorted((ROOT / "examples").glob("case_*.json"))


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(DATA)


class ScriptedModel:
    """按顺序返回预先写好的回复，用于精确测试 Agent 循环的各个分支。"""
    mode = "scripted"
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen_messages = []

    def chat(self, messages, tools):
        self.seen_messages.append(json.loads(json.dumps(messages)))
        if not self.replies:
            raise ModelError("脚本已耗尽")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name, i=0, **args):
    return ModelReply(tool_calls=[ToolCall(f"c{i}_{name}", name, args)])


# ------------------------------------------------------------------ 1. 知识库与工具
def test_documents_parsed_with_metadata(kb):
    assert {"D01", "D02", "D03", "D04"} <= set(kb.docs)
    assert kb.docs["D01"].version == "2024-01-10"
    assert kb.docs["D02"].scope == "未注明"


def test_search_finds_both_conflicting_reimbursement_docs(kb):
    docs = {h["doc_id"] for h in kb.search("报销期限多久")["hits"][:3]}
    assert {"D01", "D02"} <= docs


def test_paraphrase_hits_same_chunk_as_original(kb):
    original = kb.search("年假有多少天")["hits"][0]["chunk_id"]
    paraphrase = kb.search("一年能休几天带薪休假")["hits"][0]["chunk_id"]
    assert original == paraphrase == "D03-01"


def test_synonym_expansion_is_what_makes_paraphrase_work(tmp_path):
    for p in DATA.glob("*.txt"):
        shutil.copy(p, tmp_path / p.name)          # 不拷贝 synonyms.json
    no_syn = KnowledgeBase(tmp_path)
    hits = no_syn.search("一年能休几天带薪休假")["hits"]
    assert not hits or hits[0]["chunk_id"] != "D03-01"


def test_out_of_scope_query_has_no_hits(kb):
    assert kb.search("股票期权怎么行权")["hits"] == []


def test_quote_check_rejects_fabricated_text(kb):
    assert kb.quote_in_doc("D01", "员工应在费用发生之日起 30 天内提交报销申请")
    assert kb.quote_in_doc("D01", "员工应在费用发生之日起30天内提交报销申请")   # 空白差异容忍
    assert not kb.quote_in_doc("D01", "员工应在费用发生之日起 15 天内提交报销申请")


def test_fault_injector_fail_n_times_then_recover(kb):
    box = ToolBox(kb, FaultInjector(["search_docs:1"]))
    assert box.execute("search_docs", {"query": "年假"})["ok"] is False
    assert box.execute("search_docs", {"query": "年假"})["ok"] is True


def test_tool_errors_are_returned_not_raised(kb):
    box = ToolBox(kb)
    assert box.execute("read_doc", {"doc_id": "D99"})["ok"] is False
    assert box.execute("search_docs", {"query": ""})["ok"] is False
    assert box.execute("delete_all", {})["ok"] is False


# ------------------------------------------------------------------ 2. Agent 循环机制
QUOTE_30 = "员工应在费用发生之日起 30 天内提交报销申请，逾期未提交的，原则上不予报销。"


def test_fabricated_quote_rejected_then_corrected(kb):
    model = ScriptedModel([
        call("search_docs", 1, query="报销期限", reason="检索"),
        call("read_doc", 2, doc_id="D01", reason="读原文"),
        call("final_answer", 3, status="answered", answer="15 天", reason="作答",
             citations=[{"doc_id": "D01", "quote": "费用发生后 15 天内提交"}]),
        call("final_answer", 4, status="answered", answer="30 天", reason="修正引用",
             citations=[{"doc_id": "D01", "quote": QUOTE_30}]),
    ])
    out = QAAgent(model, ToolBox(kb)).run("报销期限多久？")
    assert out["result"]["status"] == "answered"
    assert out["result"]["citations"][0]["quote"] == QUOTE_30
    # 模型确实收到了校验反馈
    assert any("【校验反馈】" in str(m.get("content")) for m in model.seen_messages[-1])
    actions = [s["action"] for s in out["trace"]["steps"]]
    assert actions == ["search_docs", "read_doc", "final_answer", "final_answer"]


def test_citing_unread_document_is_rejected(kb):
    model = ScriptedModel([
        call("search_docs", 1, query="报销期限", reason="检索"),
        call("final_answer", 2, status="answered", answer="30 天", reason="直接答",
             citations=[{"doc_id": "D01", "quote": QUOTE_30}]),
        call("final_answer", 3, status="answered", answer="30 天", reason="再答",
             citations=[{"doc_id": "D01", "quote": QUOTE_30}]),
    ])
    out = QAAgent(model, ToolBox(kb)).run("报销期限多久？")
    # 两次都没读原文 → 降级为 insufficient，不输出无依据答案
    assert out["result"]["status"] == "insufficient"
    assert out["result"]["citations"] == []


def test_conflict_requires_two_documents(kb):
    model = ScriptedModel([
        call("read_doc", 1, doc_id="D01", reason="读"),
        call("final_answer", 2, status="conflict", answer="有冲突", reason="冲突",
             citations=[{"doc_id": "D01", "quote": QUOTE_30}]),
        call("final_answer", 3, status="conflict", answer="有冲突", reason="冲突",
             citations=[{"doc_id": "D01", "quote": QUOTE_30}]),
    ])
    out = QAAgent(model, ToolBox(kb)).run("报销期限多久？")
    assert out["result"]["status"] == "insufficient"


def test_max_steps_returns_incomplete(kb):
    model = ScriptedModel([call("search_docs", i, query="年假", reason="循环") for i in range(10)])
    out = QAAgent(model, ToolBox(kb), AgentConfig(max_steps=3)).run("年假？")
    assert out["result"]["status"] == "incomplete"
    assert out["result"]["steps_used"] == 3


def test_max_tool_calls_blocks_further_tools(kb):
    model = ScriptedModel([call("search_docs", i, query="年假", reason="循环") for i in range(3)]
                          + [call("final_answer", 9, status="insufficient", answer="达到上限", citations=[],
                                  reason="上限")])
    out = QAAgent(model, ToolBox(kb), AgentConfig(max_tool_calls=2)).run("年假？")
    steps = out["trace"]["steps"]
    assert out["result"]["tool_calls_used"] == 2
    assert steps[2]["ok"] is False and "上限" in steps[2]["observation"]


def test_plain_text_answer_without_tools_is_not_accepted(kb):
    model = ScriptedModel([ModelReply(content="报销期限是 15 天。")] * 2 + [
        call("final_answer", 1, status="insufficient", answer="未检索", citations=[], reason="结束")])
    out = QAAgent(model, ToolBox(kb)).run("报销期限多久？")
    assert out["trace"]["steps"][0]["action"] == "text_without_tool"
    assert "15 天" not in out["result"]["answer"]


def test_model_error_returns_incomplete(kb):
    out = QAAgent(ScriptedModel([ModelError("HTTP 401")]), ToolBox(kb)).run("年假？")
    assert out["result"]["status"] == "incomplete"
    assert "401" in out["result"]["answer"]


def test_trace_records_args_reason_observation_next_action(kb):
    out = QAAgent(MockModel(), ToolBox(kb)).run("报销期限多久？")
    for s in out["trace"]["steps"]:
        assert {"step", "action", "reason", "observation", "next_action"} <= set(s)
    assert out["trace"]["steps"][-1]["next_action"] == "结束"


def test_openai_response_parsing():
    data = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x1", "type": "function",
         "function": {"name": "search_docs", "arguments": "{\"query\": \"年假\", \"reason\": \"r\"}"}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    reply = OpenAICompatibleModel._parse(data)
    assert reply.tool_calls[0].name == "search_docs"
    assert reply.tool_calls[0].arguments["query"] == "年假"
    assert reply.usage["prompt_tokens"] == 10


# ------------------------------------------------------------------ 3. 端到端（Mock）与替换文档
@pytest.mark.parametrize("case_path", CASES, ids=[p.stem for p in CASES])
def test_example_cases_offline(case_path):
    out = run_case(case_path, mock=True, write_trace=False)
    assert out["trace"]["expectation_check"]["passed"], out["trace"]["expectation_check"]["failures"]


def test_mock_conflict_and_failure_paths_differ():
    paths = {}
    for cid in ("case_01_normal", "case_02_conflict", "case_03_tool_failure", "case_04_out_of_scope"):
        out = run_case(ROOT / "examples" / f"{cid}.json", mock=True, write_trace=False)
        paths[cid] = tuple((s["action"], s["ok"]) for s in out["trace"]["steps"])
    assert len(set(paths.values())) == 4


def _copy_data(tmp_path: Path) -> Path:
    for p in DATA.iterdir():
        shutil.copy(p, tmp_path / p.name)
    return tmp_path


def test_replacing_document_resolves_conflict(tmp_path):
    """审核要点：替换文档内容后，答案和动作应随之改变。D02 改为 30 天 → 不再冲突。"""
    d = _copy_data(tmp_path)
    p = d / "D02.txt"
    p.write_text(p.read_text(encoding="utf-8").replace("15 天", "30 天"), encoding="utf-8")
    out = run_case(ROOT / "examples" / "case_02_conflict.json", mock=True, data_dir=d, write_trace=False)
    assert out["result"]["status"] == "answered"
    assert "30 天" in out["result"]["answer"]


def test_removing_conflicting_document_answers_from_d01(tmp_path):
    """参考实例：只检索到 D01 且无冲突时可依据 D01 回答。"""
    d = _copy_data(tmp_path)
    (d / "D02.txt").unlink()
    out = run_case(ROOT / "examples" / "case_02_conflict.json", mock=True, data_dir=d, write_trace=False)
    assert out["result"]["status"] == "answered"
    assert {c["doc_id"] for c in out["result"]["citations"]} == {"D01"}
    assert "30 天" in out["result"]["answer"]


def test_check_expectations_detects_mismatch():
    case = {"expect": {"status": "conflict", "must_cite": ["D01", "D02"]}}
    fails = check_expectations(case, {"status": "answered", "answer": "15 天",
                                      "citations": [{"doc_id": "D02"}]})
    assert len(fails) == 2


# ------------------------------------------------------------------ 4. 真实模型（可选）
@pytest.mark.skipif(not os.getenv("LLM_API_KEY") and not (ROOT / ".env").exists(),
                    reason="未配置 LLM_API_KEY / .env，跳过真实模型测试")
@pytest.mark.parametrize("case_path", CASES, ids=[p.stem for p in CASES])
def test_example_cases_real_model(case_path):
    build_model(mock=False)     # 配置缺失会直接报错
    out = run_case(case_path, mock=False, write_trace=False)
    assert out["trace"]["expectation_check"]["passed"], out["trace"]["expectation_check"]["failures"]
