# enterprise-qa-agent

> Version 1.0.0 | 2026-11-09 | 小影校招 AI Agent 考核 · 题号 3：企业知识问答 Agent

帮员工从公司制度和产品 FAQ 里找到有原文依据的答案；文档之间冲突时列出冲突、请用户确认，查不到就明说，不编。

## 目录

- [评审入口](#评审入口)
- [1. 项目背景](#1-项目背景)
- [2. 数据目录](#2-数据目录)
- [3. 快速启动](#3-快速启动)
- [4. 架构](#4-架构)
- [5. 使用方式](#5-使用方式)
- [6. 工具接口](#6-工具接口)
- [7. 项目结构](#7-项目结构)
- [8. 技术栈](#8-技术栈)
- [9. 开发路线图](#9-开发路线图)
- [10. 配置与环境变量](#10-配置与环境变量)
- [11. 测试](#11-测试)

## 评审入口

| 看什么 | 去哪里 |
|--------|--------|
| ReAct 循环（模型选动作 → 执行工具 → 观察 → 再决策） | `src/agent.py` 的 `QAAgent.run` |
| 引用校验（引用必须是已读文档的逐字原文） | `src/agent.py` 的 `_validate_final` |
| 检索改进（BM25 + 同义词扩展 + 覆盖率过滤） | `src/knowledge.py` 的 `KnowledgeBase.search` |
| 六类验收场景的输入与期望 | `examples/case_*.json` |
| 实际输出与 Trace | `examples/traces/*.json`，汇总 `examples/traces/summary.*.json` |
| 测试结果 | `examples/test_results.txt` |
| 设计取舍、评估、AI 使用说明 | `docs/01` ~ `docs/05` |

一条命令复现全部用例（离线）：

```bash
python -m src.agent --all-cases --mock
```

## 1. 项目背景

**来源**：小影校招 AI Agent 开发岗位考核，四选一中的第 3 题“企业知识问答 Agent”，建议投入 3 小时。

**业务场景**：员工问“报销期限多久”“年假有几天”“会员能退款吗”这类问题，公司制度散落在多份文档里，版本不同、口径可能打架。员工要的是**可靠**答案：说清依据哪份文档哪句话；文档打架时告诉他打架了，而不是随便挑一个。

**输入 / 输出**：

| 输入 | 输出 |
|------|------|
| 问题 + 可选对话历史 | `status`：answered / conflict / insufficient / tool_failure / incomplete |
| 带文档 ID、版本日期的合成制度 / FAQ（`data/`） | `answer` 回答、`citations` 引用（文档 ID + 逐字原文）、`conflicts` 冲突点、`gaps` 资料缺口、`clarifying_question` 需确认的问题 |

**端到端例子**：问“报销期限多久？”

1. 模型调用 `search_docs("报销期限多久")` → 命中 D01-02、D02-02
2. 模型 `read_doc("D01")` → 原文写“30 天内提交”，适用范围“全体员工”，版本 2024-01-10
3. 模型发现 D02 也相关，`read_doc("D02")` → 原文写“15 天内提交”，适用范围“未注明”
4. 两份文档说法不一致，且都没写优先级 → `final_answer(status=conflict)`，列出两条引用，请用户确认适用哪份制度
5. Agent 校验两条 quote 确实出自 D01、D02 原文，通过后输出

如果把 D02 删掉或改成 30 天，同一个问题会变成 `answered`，依据 D01 作答（见测试 `test_replacing_document_resolves_conflict`）。

## 2. 数据目录

```
data/
├── D01.txt          # 员工费用报销管理制度 v2024-01-10：报销期限 30 天
├── D02.txt          # 关于加快报销单据流转的通知 v2025-06-01：15 天，适用范围未注明（与 D01 冲突）
├── D03.txt          # 员工休假管理制度 v2025-01-01：年假天数、顺延、申请
├── D04.txt          # 小影 App 会员服务 FAQ v2025-09-15：退款、分辨率、取消续费
└── synonyms.json    # 同义词组，用于改写后的相似问题检索
```

| 文件 | 用途 | 来源 |
|------|------|------|
| `D*.txt` | 知识库文档。头部是元数据（doc_id / title / version / issuer / scope），`---` 之后是正文，`## ` 开头是条款标题，空行分片 | 全部为合成数据，无真实制度 |
| `synonyms.json` | 同义词组。查询命中组内任一词时，以 0.6 权重补充组内其他词 | 手工整理 |

替换知识库：`--data-dir <目录>`，目录结构同上。没有运行时生成的数据，没有需要 gitignore 的数据。

## 3. 快速启动

环境：Python 3.10+。运行时只用标准库，测试需要 pytest。

```bash
git clone <本仓库 URL>
cd enterprise-qa-agent
pip install -r requirements.txt
```

离线跑（不需要 API Key，Mock 只验证流程）：

```bash
python -m src.agent "报销期限多久？" --mock
```

接真实模型：

```bash
cp .env.example .env        # Windows: copy .env.example .env，然后填写 LLM_API_KEY 等
python -m src.agent "报销期限多久？"
python -m src.agent --all-cases          # 跑全部用例，Trace 写入 examples/traces/*.real.json
```

## 4. 架构

```
         问题 + 历史
              │
              ▼
   ┌──────────────────────┐   tools schema    ┌──────────────────────┐
   │  QAAgent.run（循环）  │ ────────────────▶ │  模型（真实 / Mock）  │
   │  步数上限 8           │ ◀──────────────── │  返回 tool_calls      │
   │  工具调用上限 10      │                   └──────────────────────┘
   └──────────┬───────────┘
              │ search_docs / read_doc           final_answer
              ▼                                       │
   ┌──────────────────────┐                ┌──────────▼───────────┐
   │ ToolBox（+故障注入） │                │ 校验：status 合法、   │
   │ KnowledgeBase BM25   │                │ 引用=已读文档原文、   │
   └──────────────────────┘                │ conflict≥2 份文档     │
     结果以 tool 消息回给模型                │ 不过→反馈重试 1 次    │
                                            │ 再不过→降级 insufficient│
                                            └──────────────────────┘
```

| 模块 | 职责 |
|------|------|
| `src/agent.py` | ReAct 循环、步数/工具调用上限、final_answer 校验与降级、Trace、CLI、用例运行 |
| `src/model.py` | OpenAI 兼容接口（function calling）；离线 MockModel |
| `src/tools.py` | 工具 schema、执行（不抛异常，统一返回 `ok`）、故障注入、Trace 摘要 |
| `src/knowledge.py` | 文档解析、分片、BM25 + 同义词检索、引用原文校验 |

每一步“做什么”都由模型根据上一步工具结果决定，代码不预设分支顺序；代码只负责执行工具、限制步数、校验最终答案。详细说明见 `docs/02-solution.md`。

## 5. 使用方式

| 命令 | 作用 |
|------|------|
| `python -m src.agent "问题"` | 单次问答，打印每步动作、依据、返回摘要和最终结果 |
| `python -m src.agent "问题" --json` | 只输出结构化结果 |
| `python -m src.agent --chat` | 多轮对话，自动携带历史，可测上下文追问 |
| `python -m src.agent "那没休完的呢？" --history h.json` | 指定历史（`[{role, content}]`） |
| `python -m src.agent "问题" --fail-tool search_docs` | 注入故障：一直失败；`search_docs:1` 表示前 1 次失败 |
| `python -m src.agent "问题" --max-steps 3 --max-tool-calls 2` | 调整上限，达到上限返回 `incomplete` |
| `python -m src.agent "问题" --data-dir other_data/` | 换知识库 |
| `python -m src.agent --case examples/case_02_conflict.json` | 跑单个用例并核对期望 |
| `python -m src.agent --all-cases` | 跑全部用例，写 Trace 和汇总 |
| `python -m src.agent "问题" --trace-out t.json` | 保存 Trace |

以上命令都可以加 `--mock` 离线运行。

## 6. 工具接口

本项目是 CLI，没有 HTTP API。下表是模型可调用的工具（function calling），schema 见 `src/tools.py`。

| 工具 | 参数 | 返回 |
|------|------|------|
| `search_docs` | `query`、`top_k`、`exclude_doc_ids`、`reason` | 命中片段：doc_id、chunk_id、title、version、snippet、score、coverage；以及触发的同义词 |
| `read_doc` | `doc_id`、`chunk_id`（可选）、`reason` | 文档元数据（版本、发布部门、适用范围）和原文片段 |
| `final_answer` | `status`、`answer`、`citations[{doc_id, chunk_id, quote}]`、`conflicts`、`gaps`、`clarifying_question`、`reason` | 由 Agent 校验，不通过会把错误回给模型 |

每个工具都要求 `reason`（一句话决策依据），写入 Trace，不记录模型内部思维过程。

## 7. 项目结构

```
enterprise-qa-agent/
├── README.md
├── src/
│   ├── agent.py       # Agent 循环 + 校验 + CLI
│   ├── model.py       # 真实模型接入 + 离线 Mock
│   ├── tools.py       # 工具定义与执行
│   └── knowledge.py   # 文档加载与检索
├── tests/
│   └── test_agent.py  # 单元、机制、端到端、替换文档、真实模型（可选）测试
├── examples/
│   ├── case_01_normal.json ~ case_07_tool_recover.json   # 输入与期望
│   ├── test_results.txt                                  # pytest 输出
│   └── traces/        # 每个用例的实际输出与 Trace（*.mock.json / *.real.json）
├── data/              # 合成文档与同义词
├── docs/              # 5 份技术文档
├── requirements.txt
└── .env.example
```

## 8. 技术栈

| 项 | 选择 | 原因 |
|----|------|------|
| 语言 | Python 3.10+ | — |
| Agent 框架 | 无，原生实现 | 循环本身不到 150 行，自己写能完全控制校验、上限和 Trace 格式；没有框架版本问题 |
| 模型接入 | OpenAI 兼容 `/chat/completions` + function calling，`urllib` 直连 | 一套代码可切 DeepSeek / 通义 / OpenAI / 本地 vLLM |
| 检索 | 自研 BM25（中文二元组）+ 同义词 | 数据量小，结果可解释、可在测试里断言；不需要向量库和 embedding 服务 |
| 测试 | pytest 8.3.3 | — |
| 运行方式 | CLI | — |

## 9. 开发路线图

- [x] ReAct 循环：search_docs / read_doc / final_answer，模型自主选择下一步
- [x] 步数上限、工具调用上限，达到上限返回 `incomplete`
- [x] 引用校验：文档必须本轮读过，quote 必须是原文，conflict 至少两份文档
- [x] 冲突处理：列出冲突与引用，请求确认
- [x] 资料外问题、工具失败、追问、相似问题
- [x] 故障注入、替换文档测试、离线 Mock
- [ ] 真实模型 Trace（需配置 API Key 后运行 `--all-cases`，见 `docs/03-evaluation.md`）
- [ ] 向量检索与 BM25 混合召回
- [ ] 冲突检测从“数值不一致”扩展到条件、范围不一致
- [ ] HTTP API

## 10. 配置与环境变量

`--mock` 模式无需任何配置。真实模型从 `.env` 或系统环境变量读取：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | 无，必填 | OpenAI 兼容接口地址，如 `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | 无，必填 | API Key，不要提交到仓库（`.env` 已在 `.gitignore`） |
| `LLM_MODEL` | 无，必填 | 模型名，需支持 function calling，如 `deepseek-chat`、`qwen-plus` |
| `LLM_TEMPERATURE` | `0` | 采样温度 |
| `LLM_TIMEOUT` | `60` | 单次请求超时（秒） |

## 11. 测试

```bash
python -m pytest -q                 # 未配置 API Key 时，真实模型测试自动跳过
python -m pytest -q -k real_model   # 配置 .env 后只跑真实模型用例
```

| 测试组 | 覆盖 |
|--------|------|
| 知识库与工具 | 元数据解析、冲突文档都能召回、同义词让改写问题命中、资料外无命中、引用原文比对、故障注入 |
| Agent 机制（ScriptedModel） | 伪造引用被拒并修正、引用未读文档被拒、conflict 需两份文档、步数上限、工具调用上限、纯文本回答不被接受、模型报错 |
| 端到端（Mock） | 7 个用例全部通过期望核对；不同场景走不同路径 |
| 替换文档 | D02 改成 30 天 → 冲突消失；删除 D02 → 依据 D01 回答 |
| 真实模型（可选） | 7 个用例在真实模型下的期望核对 |

最近一次结果见 `examples/test_results.txt`。
