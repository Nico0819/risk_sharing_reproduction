# 共享信息呈现验证

日期：2026-10-26。原版和固定偏好版的 Baseline、25V、25F 均使用同一个呈现函数：[`custom/agents/observation_presentation.py`](../../custom/agents/observation_presentation.py)。

`RiskSharingParticipant` 统一构造当前收入、转账方向、历史表和完整 observation；`FixedPreferenceParticipant` 继承这一输入，只增加本人的固定偏好 system 提示词。原版没有注入固定偏好。

运行命令无需呈现参数。格式切换参数、配置项、独立呈现副本和旧观察消息解析分支已删除。日志中的 `observation_presentation=readable_v1` 只是当前格式版本。原有真实运行日志及比较结果未改写。

## 完整测试

```text
.venv/bin/python -m pytest -q

........................................................................ [ 19%]
........................................................................ [ 39%]
........................................................................ [ 59%]
........................................................................ [ 78%]
........................................................................ [ 98%]
......                                                                   [100%]
366 passed in 32.26s
```

[完整输出](pytest.txt)。测试覆盖两版实际输入的逐字段一致性、原版不含固定偏好、20人并发与身份隔离、三种实验的保险期及撤回、金额与 receipt 一致、Router 改金额被拒绝。无 observation 时，在调用模型前停止，模型请求、提交和结算均为零。格式切换相关测试已清理。

## 六个入口的完整离线运行

以下运行均为 scripted 后端，没有调用外部 LLM。每次20个 Agent、10个 pair，并发上限20，seed=2026，完整运行到自然终止。

| 版本 | 实验 | 完成轮数 | 已接受转账 | 结果 |
| --- | --- | ---: | ---: | --- |
| 原版 | Baseline | 48 | 480 | success |
| 原版 | 25V | 48 | 480 | success |
| 原版 | 25F | 48 | 480 | success |
| 固定偏好 | Baseline | 48 | 480 | success |
| 固定偏好 | 25V | 48 | 480 | success |
| 固定偏好 | 25F | 48 | 480 | success |

共2,880次接受的转账，参与者输出金额、实际提交金额和环境 receipt 金额全部一致。

公共参数为 request_limit=5000、max_output_tokens=512、router_max_steps=6、agent_max_turns=4、timeout_seconds=300。固定偏好版 preference_seed=2026。25F 明确使用测试分配概率0.5、0.25、0.25，不表示已确认论文概率。

完整命令、退出码、运行目录和命令输出路径见 [scripted_runs.json](scripted_runs.json)。本次产物在 `runs/20261026T061720Z-681468e1/`。代码指纹见 [source_fingerprints.json](source_fingerprints.json)。

重跑六个入口的离线验证：

```bash
.venv/bin/python artifacts/shared_observation_presentation_20261026/validate_sessions.py
```

## 修改范围

- `custom/agents/observation_presentation.py`：两版共享的唯一呈现实现。
- `custom/agents/risk_sharing_participant.py`：始终调用共享实现。
- `custom/agents/fixed_preferences_v1/participant.py`：继承同一 user 输入，仅注入固定偏好。
- `experiments/baseline/run_smoke.py`、`experiments/fixed_preferences_v1/run_smoke.py`：仅解析对象 observation，取消格式选择，记录呈现版本和 SHA256。
- `tests/test_observation_presentation.py`、`tests/test_observation_presentation_integration.py`、`tests/test_baseline_integration.py`：统一格式回归及无 observation 时的提前停止。
- 项目与三个实验、固定偏好版本的 README：统一使用说明。

固定偏好目录下原独立 `presentation.py` 已移除。环境经济规则、冻结论文协议、参与者偏好提示词、安装的 AgentSociety2 和 `.env` 未修改。

这次验证覆盖工程兼容性、信息边界与金额来源，不代表已经验证真实 LLM 的理解准确率或论文行为结果。
