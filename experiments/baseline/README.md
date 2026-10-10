# Baseline 实验入口

这是当前 Baseline 的最小工程入口：四个独立 `PersonAgent`、两个固定 pair，
通过 AgentSociety2 2.8.3 的 `ReActRouter` 和项目适配层进入同一个
`RiskSharingEnv`。入口有两种模式：smoke 模式用于快速验证，full-session 模式运行到论文规定的自然终止并选择支付轮。

## 配置

配置加载器固定从项目根目录读取：
`/home/zhicheng/risk_sharing_reproduction/.env`。请先复制/编辑 `.env.example`，
只填写：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

当前官方模型标识采用 `deepseek-flash`；入口会将同一套 DeepSeek 配置映射给
参与者模型和 Router 的 coder/summary 客户端，并强制关闭 thinking。真实密钥不会
进入 prompt、Agent 记忆、实验结果或 manifest。`.env` 已在 `.gitignore` 中排除。

## 离线 smoke 验证

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted --participants 4 --completed-period-limit 3 --seed 2026
```

该命令不联网，使用本地 scripted 模型响应，但仍运行真实
`PersonAgent -> ReActRouter -> RiskSharingRouterAdapter -> @tool -> RiskSharingEnv`。
它完成指定的工程轮数后停止，不执行支付轮选择。每次运行在
`experiments/baseline/runs/` 生成独立目录，包含 `manifest.json`、
`agent_actions.jsonl`、`period_records.jsonl`、`summary.md` 和结构化诊断日志 `run.log`。

## 离线完整 Baseline

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted --participants 4 --full-session --seed 2026 \
  --request-limit 2000 --max-output-tokens 512 --timeout-seconds 300
```

`--full-session` 忽略 `--completed-period-limit`，持续运行到 `RiskSharingEnv.finished`。
入口会保留所有已结算轮次的 `period_records.jsonl`，并在实际终止于第 48、49 或 50
轮后，用不消耗环境随机数的独立选择器随机选出一个已实现轮次，写入 `payment.json`。
其中包含 `selected_period` 和该轮每位参与者的结算记录；这是研究者侧支付轮数据，
不表示已经接入外部支付系统。

## Participant 决策边界

每个 `PersonAgent` 的 step 先通过真实 `ask_env(readonly=true)` 读取自己的原始
`get_observation`，再由该 Agent 的 participant client 选择 amount。participant client
通过 `RISK_SHARING_PARTICIPANT_DECISION` envelope 提交 `decision_type`、当前
`decision_id` 和严格整数 amount。`RiskSharingRouterAdapter` 将这三个值绑定在环境
上下文中，Router 只能转发完全相同的参数；缺少参数、改写 amount 或伪造身份会被拒绝。

`agent_actions.jsonl` 同时记录 `participant_model_message.observation`、
`participant_model_output.amount`、`router_action` 和 `environment_receipt.amount`，
可直接检查 participant 输出与环境实际接受值相等。低收入 participant 只完成等待，
不会提交 transfer。

## 真实 API（填写配置后再运行）

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend deepseek --participants 4 --full-session --seed 2026 \
  --request-limit 2000 --max-output-tokens 2048 \
  --timeout-seconds 900 --request-timeout 60
```

该命令会运行到真实随机终止，调用上限、单次输出上限和总时限由参数明确控制；
鉴权失败、余额不足或预算耗尽会停止，不会自动换供应商或循环重试。填写 `.env`
后再运行，并单独核对 `manifest.json` 中的 `success`、`stop_reason`、
`completed_periods` 和 `payment.json`。

## 测试

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m pytest -q
```

测试包括协议/环境单元测试、配置安全检查、离线 scripted smoke 链路和完整自然终止
链路；不把离线 scripted 结果当作 DeepSeek API 成功。
