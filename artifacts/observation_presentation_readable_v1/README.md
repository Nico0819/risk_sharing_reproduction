# readable_v1 信息呈现验证

日期：2026-10-26。此次新增可切换的呈现模式，没有调用外部 LLM。原有真实实验文件没有改写。

当前原版和固定偏好版的所有实验均使用共享呈现函数 `custom/agents/observation_presentation.py`，无需增加参数。格式切换参数和旧格式实现已删除；日志中的 `observation_presentation` 是当前格式的版本标记。以下是当时保存的验证结果；统一实现后的验证见 [共享呈现验证](../shared_observation_presentation_20261026/README.md)。历史运行记录没有改写。

## 实际输入预览

- [25F：Agent 10，第2轮](25f_agent_10_period_2_preview.md)：上轮由本人转给搭档40，本轮仍是本人获得高收入；历史表明确发送方向。
- [25V：Agent 6，第30轮](25v_agent_6_period_30_preview.md)：当前双方收入300和125，展示 `300 - x` 与 `125 + x`，提醒保险扣费已计入，不给出建议金额。

预览由已有真实输入离线重新呈现生成，不是新调用 LLM 的结果。完整 observation 同时保留在消息内，转换只解码原先嵌套的 JSON 字符串。没有添加私人记忆、修改个人偏好或改变环境规则。

## 验证结果

```text
.venv/bin/python -m pytest -q

........................................................................ [ 19%]
........................................................................ [ 38%]
........................................................................ [ 58%]
........................................................................ [ 77%]
........................................................................ [ 97%]
..........                                                               [100%]
370 passed in 31.04s
```

[完整 pytest 输出](pytest.txt)。新增28项测试，包括信息无损、零转账和无机会的区别、发送方向、保险撤回、20人隔离并发、三种实验新旧格式的逐字段对照、原格式兼容及 Router 修改金额被拒绝。

随后通过 CLI 运行三种完整 scripted 实验：

| 实验 | Agent / pair | 并发上限 | 完成轮数 | 结果 |
| --- | --- | --- | --- | --- |
| Baseline | 20 / 10 | 20 | 48 | success，natural_termination |
| 25V | 20 / 10 | 20 | 48 | success，natural_termination |
| 25F | 20 / 10 | 20 | 48 | success，natural_termination |

运行参数：seed=2026，preference_seed=2026，request_limit=5000，max_output_tokens=512，router_max_steps=6，agent_max_turns=4，timeout_seconds=300。25F 使用显式测试分配概率0.5、0.25、0.25，不表示论文参数已确认。

完整命令和产物目录见 [scripted_runs.json](scripted_runs.json)，原始命令输出为 [Baseline](baseline_scripted.log)、[25V](25v_scripted.log)、[25F](25f_scripted.log)。本次产物位于 `runs/20261026T055103Z-0cf0a3eb/` 下各 treatment 目录。

## 既有真实输入的无损检查

对三次既有 LLM 实验的全部2,956条 participant 输入进行了离线转换：Baseline 984条，25V 989条，25F 983条。解码后所有 observation 字段、完整历史和外层上下文逐项一致。检查记录见 [historical_input_audit.json](historical_input_audit.json)。

为避免缩进 JSON 放大长历史，完整原始数据使用紧凑 JSON，前部以表格和会计关系展示。第45轮 user 消息的平均字符数：

| 实验 | legacy | readable_v1 |
| --- | ---: | ---: |
| Baseline | 22052.90 | 22684.40 |
| 25V | 28019.38 | 27995.19 |
| 25F | 27999.50 | 28027.90 |

字符数不是模型 token 数或费用估计。呈现层本身不增加模型调用。

## 文件与复核

业务改动位于：

- `custom/agents/fixed_preferences_v1/presentation.py`：确定性呈现与来源一致性检查。
- `custom/agents/fixed_preferences_v1/participant.py`：仅在选用新格式时替换 user 消息。
- `experiments/fixed_preferences_v1/run_smoke.py`：模式参数、兼容解析、配置和日志记录。
- `experiments/fixed_preferences_v1/README.md`：使用方法和对照说明。
- `tests/test_observation_presentation.py`、`tests/test_observation_presentation_integration.py`：新增回归测试。

重新检查历史输入（不会调用模型）：

```bash
.venv/bin/python artifacts/observation_presentation_readable_v1/audit_historical_inputs.py
```

重新运行三种离线完整实验（生成新的运行目录）：

```bash
.venv/bin/python artifacts/observation_presentation_readable_v1/validate_scripted_sessions.py
```

这些验证证明信息被保留、身份和金额边界继续有效、真实框架链路兼容。是否减少真实 LLM 的历史误读或算术误述，还需要在相同配置下比较 legacy 与 readable_v1。
