# 02 方案：调用流程、动态决策、技术取舍

## 调用流程

```
messages = [system, *history, user]
for step in 1..max_steps:
    reply = model.chat(messages, tools)            # 模型选动作
    if 没有 tool_call:  拒收，提醒走 final_answer；continue
    for tool_call in reply:
        if final_answer:
            errors = validate(args)                # 引用校验
            通过 → 返回结果
            不通过且首次 → 把错误作为 tool 结果 + 校验反馈回给模型，继续循环
            不通过且再次 → 降级（只保留可校验引用，否则 insufficient）
        else:
            超过 max_tool_calls → 返回 ok=false “已达上限”
            obs = toolbox.execute(...)             # 执行工具，失败也返回 ok=false，不抛异常
            messages += tool(obs)                  # 观察结果交还模型
返回 incomplete（步数耗尽）
```

代码只做三件事：执行模型选的工具、限制步数、校验最终答案。**下一步做什么完全由模型基于工具返回决定**，循环里没有“先搜再读再答”的固定分支。

## 动态决策点

系统提示（`src/agent.py` 的 `SYSTEM_PROMPT`）给出决策规则，模型根据观察结果自选：

| 观察到的工具结果 | 模型应选的下一步 |
|------------------|------------------|
| 检索无命中 / 相关度低 | 换关键词、同义词补充检索；两次改写仍无结果 → `insufficient` |
| 检索命中 | `read_doc` 读原文核对，不凭摘要作答 |
| 多个文档都涉及该问题 | 逐个读取，比较内容、版本、适用范围、替代声明 |
| 内容一致 | `answered` |
| 内容不一致且无优先级说明 | `conflict`，列出各方说法，`clarifying_question` 请用户确认 |
| 工具 `ok=false` | 重试或换参数；仍失败 → `tool_failure` |
| 追问（“那…呢”） | 结合历史补全检索词 |
| 校验反馈指出引用错误 | 重新读取原文、修正引用 |

用例里能看到的不同路径（Mock 下，`examples/traces/summary.mock.json`）：

| 用例 | 路径 | 调整原因 |
|------|------|----------|
| case_01 正常 | search → read D03 → answered | 只有一份相关文档 |
| case_02 冲突 | search → read D01 → read D02 → conflict | 检索命中两份文档，读第二份比对 |
| case_03 工具失败 | search ✗ → search ✗ → tool_failure | 工具连续失败 |
| case_04 资料外 | search（无命中）→ search 关键词（无命中）→ insufficient | 无命中触发改写 |
| case_07 失败恢复 | search ✗ → search ✓ → read D01 → answered | 重试成功后继续正常流程 |

## 防止“无依据回答”的三道关

1. **提示约束**：只能依据工具返回作答，quote 必须逐字复制。
2. **协议约束**：模型直接输出文本不被当作答案，必须调 `final_answer`。
3. **代码校验**（`_validate_final`）：
   - 引用的文档必须存在，且本轮通过 `read_doc` 读过（光检索到摘要不算）；
   - quote 去空白后必须是原文子串；
   - `answered` 至少一条有效引用，`conflict` 至少两份不同文档；
   - 不通过把具体错误回给模型修正一次；再不通过就剔除无效引用，没有可靠引用则降级为 `insufficient`。

第 3 关是这个方案的重点：审核要点里“错误引用属于关键缺陷”，这个问题不能指望模型自觉，必须在代码层拦住。

## 检索改进

| 问题 | 做法 |
|------|------|
| 中文无分词 | 字符二元组 + 英文/数字词，去掉“什么/多少/公司”等疑问词、虚词 |
| 换说法搜不到（“带薪休假”“工龄”vs 原文“年休假”“累计工作”） | `data/synonyms.json` 同义词组，命中组内任一词时以 0.6 权重补充其他词 |
| 只沾一个字也被召回 | `coverage` = 命中的查询词权重 / 原查询词总权重，低于 0.2 过滤 |
| 补充检索重复看同一文档 | `exclude_doc_ids` 参数 |
| 检索词质量依赖模型 | 工具返回 `expanded_terms`、`score`、`coverage`，模型能判断是否需要改写 |

测试 `test_synonym_expansion_is_what_makes_paraphrase_work` 证明去掉同义词表后，改写问题就召回不到 D03-01。

## 技术取舍

| 决策 | 选择 | 放弃的方案及原因 |
|------|------|------------------|
| Agent 框架 | 原生实现 | LangGraph / ADK 能做，但这里的循环很短，自己写更容易控制校验、上限和 Trace 字段，也没有版本兼容问题。代价是没有现成的持久化、并发、可视化。 |
| 模型协议 | OpenAI 兼容 function calling | 文本解析 ReAct（Thought/Action 正则）易碎；function calling 参数是结构化 JSON。几乎所有国内外模型都兼容这个协议。 |
| 结束方式 | `final_answer` 作为工具 | 结构化输出可直接校验；比“模型输出文本即结束”更可控。 |
| 检索 | BM25 + 同义词 | 4 份文档用向量检索收益小，还要多一个 embedding 服务；BM25 结果可解释，测试能断言具体 chunk。缺点是同义词要人工维护。 |
| 冲突判定 | 交给模型；代码只要求 conflict 引用两份文档 | 规则判定冲突覆盖面窄（Mock 里只能比较数值）。 |
| 工具失败 | 工具层不抛异常，返回 `ok=false` 给模型 | 如果在代码里自动重试，就成了“自动重试”而不是模型决策（题目明确不算）。重试与否由模型决定。 |
| 离线 Mock | 启发式规则策略 | 只用于验证循环、工具、校验、Trace 是否通畅，**不证明真实模型的决策效果**。 |

## 上限

| 上限 | 默认 | 达到后 |
|------|------|--------|
| `max_steps`（模型决策轮数） | 8 | 返回 `status=incomplete`，gaps 写明“执行步数耗尽” |
| `max_tool_calls`（不含 final_answer） | 10 | 后续工具调用直接返回 `ok=false, 已达上限`，提示模型结束 |
| `max_validation_retries` | 1 | 再次不过就降级 |

## Trace 格式

每个用例一份 `examples/traces/<case_id>.<mode>.json`：

```json
{
  "case_id": "...", "mode": "real|mock", "model": "...", "question": "...", "history": [],
  "limits": {"max_steps": 8, "max_tool_calls": 10},
  "steps": [
    {"step": 1, "action": "search_docs", "args": {"query": "..."}, "reason": "决策依据",
     "ok": true, "observation": "返回摘要", "next_action": "read_doc"}
  ],
  "result": {"status": "...", "answer": "...", "citations": [...], "conflicts": [...], "gaps": [...]},
  "expectation_check": {"passed": true, "failures": []},
  "usage": {"prompt_tokens": 0}, "duration_ms": 0
}
```
