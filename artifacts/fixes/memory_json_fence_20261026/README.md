# 记忆 JSON 代码块导致实验中断：修复验证

2026-10-26。对应运行：`experiments/fixed_preferences_memory_v1/runs/baseline/20261026T081152Z-1edfe2d0`。

第22轮 Agent 5 在 waiting 任务中返回完整 JSON，但外层带有 Markdown 代码块。此前直接 `json.loads` 导致 `JSONDecodeError`，整场仅结算21轮。

修复在 `custom/agents/fixed_preferences_memory_v1/state.py` 的 `parse_memory_output`：只剥离整段唯一、完整的 JSON 或无语言标签代码块，再执行原有解析和状态校验。不从说明文字中截取内容、不修复残缺 JSON、不补字段或经济决定。原始输出在日志中保留；磁盘状态文件仍必须是严格 JSON。没有增加模型调用，也没有改变记忆提示词、偏好或实验规则。

实际失败输出已原样送入新解析器，并用当时的 observation 和之前有效状态完成 `updated_state` 校验。内容未改写，原运行70个文件的指纹保持不变。详见 [real_failure_replay.json](real_failure_replay.json)。

完整测试命令：

```bash
PYTHON_DOTENV_DISABLED=1 LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest -q
```

```text
532 passed in 64.57s (0:01:04)
```

完整输出见 [pytest.txt](pytest.txt)。新增边界测试覆盖围栏大小写、空白、CRLF、原文保存，以及多代码块、解释文字、截断、重复字段、非法金额和证据继续拒绝。

新增真实框架集成测试让所有 scripted 记忆输出带上代码块：

| 实验 | Agent数 / 并发 | 完成轮数 | 成功记忆更新 | 接受转账 |
| --- | --- | ---: | ---: | ---: |
| Baseline | 4 / 4 | 31 | 124 | 62 |
| 25V | 4 / 4 | 31 | 128 | 62 |
| 25F | 4 / 4 | 31 | 124 | 62 |

三类均通过第22轮等待者场景，保留原始围栏响应；后续 participant 输入、保存的记忆修订号和环境接受金额对应一致。25F 使用显式离线测试概率0.5/0.25/0.25。

本次未调用外部 LLM API。原运行仍如实标记为失败；重新执行原实验命令会创建新目录并从第1轮开始，当前没有断点续跑。此前审计发现的自然语言事实误读、惯例强化和重复更新属于其他问题，本次没有改变这些行为设置。
