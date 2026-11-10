# 03 评估：测试预期、实际结果、失败和限制

## 用例与结果

所有用例在 `examples/case_*.json`，每个用例含 `expect`（期望 status、必须引用的文档、答案必须包含的文字），运行时自动核对并写入 Trace 的 `expectation_check`。

```bash
python -m src.agent --all-cases --mock    # 离线
python -m src.agent --all-cases           # 真实模型（需 .env）
```

| 用例 | 验收点 | 输入 | 期望 | Mock 实际 | Trace |
|------|--------|------|------|-----------|-------|
| case_01 | 明确答案 | 工作满 3 年，年假有多少天？ | answered，引用 D03 | ✅ answered，引用 D03-01 | `examples/traces/case_01_normal.mock.json` |
| case_02 | 文档冲突 | 报销期限多久？ | conflict，引用 D01+D02 | ✅ conflict，列出 30 天 vs 15 天并请求确认 | `examples/traces/case_02_conflict.mock.json` |
| case_03 | 工具失败 | 单笔超 5000 元谁审批？（search_docs 一直失败） | tool_failure | ✅ 重试 1 次后 tool_failure | `examples/traces/case_03_tool_failure.mock.json` |
| case_04 | 资料外问题 | 股票期权怎么行权？ | insufficient | ✅ 两次检索无命中，insufficient | `examples/traces/case_04_out_of_scope.mock.json` |
| case_05 | 相似问题 | 工龄 12 年，一年能休几天带薪休假？ | answered，引用 D03 | ✅ 同义词扩展命中 D03-01 | `examples/traces/case_05_paraphrase.mock.json` |
| case_06 | 上下文追问 | （上一轮问年假）那没休完的能留到明年用吗？ | answered，引用 D03 含“3 月 31 日” | ✅ 合并历史检索，命中 D03-02 | `examples/traces/case_06_followup.mock.json` |
| case_07 | 失败后恢复（第二条路径） | 同 case_03，search_docs 只失败 1 次 | answered，引用 D01 | ✅ 重试成功后正常作答 | `examples/traces/case_07_tool_recover.mock.json` |

汇总：`examples/traces/summary.mock.json`，7/7 通过。

**真实模型结果**：运行 `python -m src.agent --all-cases` 后生成 `examples/traces/*.real.json` 和 `summary.real.json`。
【待补充：使用的模型名称、通过数、与 Mock 路径的差异。提交前必须用真实模型跑一遍并把 Trace 提交进仓库。】

## 两条以上决策路径

同一套代码，不同工具结果 → 不同后续动作（`summary.mock.json` 的 `steps` 字段）：

- case_01：search → read → final（单文档）
- case_02：search → read D01 → **read D02**（发现第二份相关文档）→ final(conflict)
- case_04：search（无命中）→ **改写再 search** → final(insufficient)
- case_03：search ✗ → **重试** ✗ → final(tool_failure)
- case_07：search ✗ → 重试 ✓ → **继续 read** → final(answered)

注意：Mock 下这些路径由启发式规则产生，只证明循环能根据观察结果走不同分支；真实模型的决策效果以 `*.real.json` 为准。

## 自动化测试

```bash
python -m pytest -q
```

最近一次输出见 `examples/test_results.txt`（28 passed，7 个真实模型用例因未配置 Key 跳过）。

| 测试 | 证明了什么 |
|------|------------|
| `test_fabricated_quote_rejected_then_corrected` | 模型给出原文里没有的“15 天”引用 → 被拒、模型收到反馈、修正后通过 |
| `test_citing_unread_document_is_rejected` | 没 read_doc 就引用 → 两次都被拒 → 降级 insufficient，不输出答案 |
| `test_conflict_requires_two_documents` | 声称冲突但只引用一份文档 → 不接受 |
| `test_max_steps_returns_incomplete` | 模型一直搜不结束 → 第 3 步截止，返回 incomplete |
| `test_max_tool_calls_blocks_further_tools` | 第 3 次工具调用被拒，返回“已达上限” |
| `test_plain_text_answer_without_tools_is_not_accepted` | 模型不查资料直接说“15 天” → 不被当作答案 |
| `test_model_error_returns_incomplete` | 模型接口 401 → incomplete，不崩溃 |
| `test_replacing_document_resolves_conflict` | **替换文档**：D02 改为 30 天 → 同一问题从 conflict 变为 answered |
| `test_removing_conflicting_document_answers_from_d01` | 删除 D02 → 依据 D01 回答 30 天（对应题目参考实例） |
| `test_synonym_expansion_is_what_makes_paraphrase_work` | 去掉同义词表后改写问题召回失败，说明检索改进有效 |
| `test_example_cases_offline[*]` | 7 个用例端到端通过 |
| `test_example_cases_real_model[*]` | 真实模型下 7 个用例（需 Key） |

## 失败与已修复的问题

| 发现 | 原因 | 修复 |
|------|------|------|
| 改写问题“一年能休几天带薪休假”无命中 | coverage 分母包含同义词权重，同义词越多覆盖率越低 | 分母只计原查询词，上限 1 |
| case_05 期望“包含 10”虚假通过 | Mock 引用了“满 1 年不满 10 年 5 天”一句，碰巧含“10” | Mock 改为整句引用分档规则；期望收紧为“年假 10 天” |
| 工具失败与资料外路径的动作序列相同（search, search, final） | 只比较动作名 | 路径比较加入每步 ok 状态 |

## 已知限制

1. **Mock 不代表真实模型效果**。Mock 的冲突检测只比较引用句里的“N 天/日/个工作日”，换成非数值冲突就识别不了；它也不做“12 年属于哪一档”的推理，只引用整条规则。
2. 引用校验是**子串匹配**：能拦住编造的引用，但拦不住“引用是真的、结论推错了”的情况（例如引用 30 天却回答 15 天）。
3. 同义词表人工维护，覆盖面有限；未接向量检索。
4. 冲突处理依赖模型判断“是否有优先级说明”；当前数据里没有带替代声明的文档，这条分支未被用例覆盖。
5. 真实模型每次运行可能走不同路径；当前只设计了单次运行核对，未统计多次运行的稳定性。
6. 对话历史只传文本，不传上一轮的引用结构。
