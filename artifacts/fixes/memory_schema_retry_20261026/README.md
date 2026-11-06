# 记忆结构偏离后的有限修正

2026-10-26。对应运行：`experiments/fixed_preferences_memory_v1/runs/baseline/20261026T121103Z-6255a6c3`。

这次运行只结算到第7轮。第8轮三个等待者返回了合法JSON，但内容不符合记忆结构：Agent 1额外写入`partner_id`、`last_reviewed_period`和`last_reviewed_task`；Agent 10、11的`evidence_periods`各有7项，超过6项限制。三条原始输出仍会被完整内容校验拒绝，见[实际失败核对](real_failure_check.json)。这与之前已修复的Markdown代码块包装问题不同。

## 修改

- `custom/agents/fixed_preferences_memory_v1/participant.py`：原参与者模型收到具体错误和被拒绝草稿后，可再修正两次，每次仍只更新记忆、不使用工具。总计最多三次请求。程序不删除字段、不挑选六项证据、不补金额。
- `state.py`：统一保存`MEMORY_UPDATE_MAX_ATTEMPTS=3`。JSON解析、字段、身份、金额预测及历史引用的校验不放宽。
- `memory_update.md`：明确只输出五个内容字段，不能复制私人状态的外围元数据；超过六轮历史时，由模型自行选最多六项依据。
- 新版runner：每次尝试计入原预算，实际模型输入和调用日志记录`memory_update_attempt`。成功/拒绝日志记录`attempt`和`max_attempts`；manifest和summary记录尝试上限及拒绝次数。内部参数不会传给模型供应商。
- 新版README及集成测试同步更新。

失败草稿仅作为转义后的修正输入，不写入状态，也不加入经济决策输入。只有通过全部校验的响应才保存一次、增加一次revision。三次都失败时保留旧状态并停止；预算耗尽、网络错误、取消和写盘错误不会被当成格式问题自动重试。

## 验证

```bash
PYTHON_DOTENV_DISABLED=1 LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest -q
```

```text
536 passed in 71.09s (0:01:11)
```

完整输出：[pytest.txt](pytest.txt)。

新增20人、并发20的真实框架离线测试，在第8轮对Agent 1注入额外元数据、对Agent 10/11注入7项依据；第二次由相同participant客户端生成合法输出，全部通过校验并继续到第9轮。180次记忆保存对应183次记忆请求，包含3次修正；participant共363次、Router共90次，总计453次scripted调用。新证据子集由测试模型输出，程序没有截取列表。

还验证了：持续非法输出每人最多三次且不生成经济行动；修正时预算耗尽不会越界；网络异常不进入格式重试；坏草稿中的私有消息标签经转义后不会破坏绑定或进入经济消息。原有代码块兼容和三类实验测试仍通过。

本次未调用外部LLM。离线测试验证修正流程和边界，不保证外部模型每次都能在三次内成功。原运行70个文件保持不变，仍如实记录失败；重新运行原命令会从第1轮建立新目录，当前不支持断点续跑。
