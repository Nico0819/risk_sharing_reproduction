# 固定偏好版本验证记录

验证日期：2026-10-25。所有运行使用 `scripted`，未调用外部 LLM API。

本次只新增版本文件和测试。创建版本时记录的 27 个原有文件逐一核对 SHA256，内容全部未变，参见 [baseline_reference.json](baseline_reference.json)。原有未提交修改保持原样。

## 全量测试

命令：

```bash
.venv/bin/python -m pytest -q
```

完整输出：

```text
........................................................................ [ 21%]
........................................................................ [ 42%]
........................................................................ [ 63%]
........................................................................ [ 84%]
......................................................                   [100%]
342 passed in 24.02s
```

新增 37 项配置测试和 16 项真实框架集成测试，包括：严格配置校验、独立随机数与均衡分配、每人隔离、跨轮恢复、20 人并发、保险阶段、输入日志一致性、导出配置复用、原版与新版交替运行、异常输入在调用模型前拒绝、可选 deliberation，以及 Router 篡改金额、身份和 token 的拒绝。

## 单轮与完整运行

| 运行 | Agent / 并发上限 | 结算轮数 | 接受转账数 | participant / Router 脚本调用数 | 结果 |
|---|---|---:|---:|---:|---|
| Baseline 单轮 | 4 / 4 | 1 | 2 | 4 / 2 | 成功 |
| Baseline 完整 | 20 / 20 | 48 | 480 | 960 / 480 | 自然终止 |
| 25V 完整 | 20 / 20 | 48 | 480 | 980 / 500 | 自然终止；20 个保险选择 |
| 25F 完整 | 20 / 20 | 48 | 480 | 960 / 480 | 自然终止；20 个保险分配 |

这些调用数是本地 scripted client 调用数，不是外部 API 请求数。

单轮命令：

```bash
.venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --backend scripted --treatment baseline \
  --participants 4 --agent-concurrency 4 --completed-period-limit 1 \
  --seed 2026 --preference-seed 2026 --request-limit 100 \
  --max-output-tokens 512 --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 180
```

完整运行分别将 `--treatment` 设为 `baseline`、`25v`、`25f`，其余参数为：

```text
--backend scripted --participants 20 --agent-concurrency 20 --full-session
--seed 2026 --preference-seed 2026 --request-limit 2000
--max-output-tokens 512 --router-max-steps 6 --agent-max-turns 4
--timeout-seconds 300
```

25F 另外传入 `--forced-assignment-probabilities 0.5 0.25 0.25`。这是明确指定的测试配置，不声称为论文确定的概率。

运行产物（运行目录由 Git 忽略，链接指向本机结果）：

- [Baseline 单轮 summary](runs/baseline/20261025T124401Z-0f68cfd9/summary.md) · [完整控制台输出](runs/baseline/20261025T124401Z-0f68cfd9/console.log)
- [Baseline 完整 summary](runs/baseline/20261025T124036Z-75bcff43/summary.md) · [完整控制台输出](runs/baseline/20261025T124036Z-75bcff43/console.log)
- [25V 完整 summary](runs/25v/20261025T124252Z-f521a883/summary.md) · [完整控制台输出](runs/25v/20261025T124252Z-f521a883/console.log)
- [25F 完整 summary](runs/25f/20261025T124252Z-f5b79b10/summary.md) · [完整控制台输出](runs/25f/20261025T124252Z-f5b79b10/console.log)

每个目录同时包含 `report.html`、`manifest.json`、`participant_profiles.json`、`participant_contexts.jsonl`、`agent_actions.jsonl` 和结算记录。

## 产物核对

逐条核对四次运行的 participant 输入：每次均包含且只包含本人的双维偏好配置，配置指纹匹配；observation 身份匹配，没有其他 pair 的历史或 treatment 等研究字段。每个 workspace 的固定配置保持一致，私人记忆关闭。

逐条核对环境接受的转账：金额是合法整数，participant 输出、实际提交、环境 receipt 三者相同；当前 observation 与提交的 Agent、decision ID 一致，decision ID 不重复，关联偏好指纹正确。

三次 20 人完整实验导出的配置文件完全相同，SHA256 均为：

```text
75bdc98b88945880a16a5dbebdc282c17a399d7653867ba8494e42bf0417e3c3
```

## 验证范围

以上证明配置真实送达 participant model、本人信息隔离、框架调用和环境结算正常。scripted 策略沿用基准的工程策略，故意不按偏好改变金额，因此这些运行不能证明偏好的经济行为效果。

后续仍需用独立选择任务检验真实 LLM 对风险与优势不平等等级的响应，以及进行人类数据校准和正式对照实验。本次没有启用私人记忆，没有增加互惠或内疚维度。
