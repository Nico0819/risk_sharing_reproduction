# 固定偏好与私人记忆：实现和离线验证

2026-10-26。新增版本为 `fixed_preferences_memory_v1`；原有 `fixed_preferences_v1` 保留作为基准。本次没有调用外部 LLM API。

## 实现

每个 Agent 继续使用原来的风险厌恶和优势不平等厌恶配置。新版主提示词与固定偏好基准逐字节相同，额外的记忆提示词单独保存。

```text
本人当前 observation + 本人上次保存的主观记录
  → 本人的 participant client 更新记录（无工具权限）
  → 验证字段、身份、搭档、运行范围、偏好绑定和历史引用
  → 原子保存到本人 workspace/private_memory.json
  → 同一个 participant client 读取更新后的记录，选择当轮行动
  → Router 转发明确选择 → 环境验证并保存 → env.step() 统一结算
```

记录包括对搭档的判断、预期对方的转账、认为对方对自己的预期、不确定性及至多六个已结算轮次的依据。初始记录为空，不预设信任、合作或转账金额。程序从合法 observation 附上被引用的事实，不让模型编写历史账目。预测金额不能直接成为提交金额。

每份新 observation 增加一次 participant 调用，低收入等待者也更新；25V 的投保和转账分别更新。同一 observation 的行动重试复用已保存的记录。记忆更新失败时保留旧记录，本次不继续经济决策。文件损坏、缺失或身份绑定不符时拒绝继续，不静默重置。

使用本项目专用的 workspace JSON 状态，`private_memory_enabled=True`。AgentSociety2 通用记忆保持关闭（`enable_memory=False`、`framework_memory_enabled=False`），因此不会额外调用通用记忆整理服务。每轮框架重新恢复 Agent 时读取对应文件。

## 回归测试

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：

```text
482 passed in 52.93s
```

完整输出见 [pytest.txt](pytest.txt)。新增 97 项状态测试和 19 项真实框架集成测试，覆盖跨轮持久化、20 人并发、私有输入隔离、本人客户端更新记忆、三种处理、25V 阶段切换、重试、非法输出、预算限制、文件绑定、Router 修改金额被拒绝，以及经济调用与记忆修订号的对应。受控 scripted fixture 证明决定能够读取记忆；这项测试不用于判断实际经济行为效果。

## 三类完整离线实验

复现本次验证：

```bash
.venv/bin/python artifacts/fixed_preferences_memory_v1_20261026/validate_sessions.py
```

每类 20 人、10 组、并发上限 20，`seed=2026`、`preference_seed=2026`、`request_limit=5000`、`max_output_tokens=512`、`router_max_steps=6`、`agent_max_turns=4`、`timeout_seconds=300`。25F 使用显式测试概率 `0.5 0.25 0.25`；它不是已确认的论文原始概率。

| 实验 | 完成轮数 | 记忆更新 | 环境接受转账 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Baseline | 48 | 960 | 480 | 自然终止，通过 |
| 25V | 48 | 980 | 480 | 自然终止，通过 |
| 25F | 48 | 960 | 480 | 自然终止，通过 |

机器可读的完整命令、计数和产物路径见 [scripted_runs.json](scripted_runs.json)。其中 participant/router request 计数均为 scripted 模拟调用，不是外部 API 请求数。

运行产物：

- [Baseline summary](../../experiments/fixed_preferences_memory_v1/runs/baseline/20261026T071741Z-8c0a7b5c/summary.md)
- [25V summary](../../experiments/fixed_preferences_memory_v1/runs/25v/20261026T071741Z-8c0a7b5c/summary.md)
- [25F summary](../../experiments/fixed_preferences_memory_v1/runs/25f/20261026T071741Z-8c0a7b5c/summary.md)

验证脚本逐条检查了：记忆模型原始 JSON 等于保存内容；同一个 Agent 的修订号及前后状态连续；最终文件等于最后一次更新；经济调用读取的记忆指纹和修订号与行动日志相符；participant 输出金额、提交金额、环境回执金额一致。三类实验全部通过这些检查。

## 基准与后续比较

[baseline_preservation.json](baseline_preservation.json) 记录了原有 56 份代码和文档的指纹检查结果，未发现变化；三次完整实验结束后也分别复核。原有实验历史及比较文件未覆盖。

新版入口及运行说明见 [README](../../experiments/fixed_preferences_memory_v1/README.md)。代码和提示词位于 [custom/agents/fixed_preferences_memory_v1](../../custom/agents/fixed_preferences_memory_v1/)。

尚未用外部 LLM 验证记忆输出的实际稳定性、费用和行为效果。正式 scripted 金额策略沿用基准，因此本次运行不提供改善同质化或接近论文结果的证据。新增持久状态的同时也增加了决策前的一次模型调用和提示内容，之后做效果比较时需要一并考虑。主观记录可能判断错误，程序保证的是输入边界和可追溯性，并不保证模型的社会判断正确。
