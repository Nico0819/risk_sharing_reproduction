# Baseline 实验入口

这是当前 Baseline 的最小工程入口：四个独立 `PersonAgent`、两个固定 pair，
通过 AgentSociety2 2.9.0 的 `PersonAgent`、项目适配层和真实 `ReActRouter` 进入同一个
`RiskSharingEnv`。参与者选择金额，Router 模型只负责整理参数和调用工具。
入口有两种模式：scripted 模式用于快速验证，llm 模式连接参与者和 Router 模型；两种模式都
支持 smoke 和 full-session 运行。

## 配置

配置加载器固定从项目根目录读取：
`/home/zhicheng/risk_sharing_reproduction/.env`。请先复制/编辑 `.env.example`，
只填写：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
```

入口将同一份 LLM 配置用于参与者和 Router 的独立客户端，并强制关闭 thinking。真实密钥不会
进入 prompt、Agent 记忆、实验结果或 manifest。`.env` 已在 `.gitignore` 中排除。

## 离线 smoke 验证

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted --participants 4 --completed-period-limit 3 --seed 2026 \
  --agent-concurrency 4
```

该命令不联网，使用本地 scripted participant 和 router 模型响应，但仍运行真实
`PersonAgent -> ask_env -> RiskSharingRouterAdapter -> ReActRouter -> RiskSharingEnv`。
它完成指定的工程轮数后停止，不执行支付轮选择。每次运行在
`experiments/baseline/runs/` 生成独立目录，包含 `manifest.json`、
`agent_actions.jsonl`、`period_records.jsonl`、`summary.md` 和结构化诊断日志 `run.log`。

`--agent-concurrency` 控制同一轮最多同时运行多少个参与者。默认值是 `1`，方便复现
原来的逐个运行方式；设为 `2` 可以一次运行两个 Agent，设为 `4` 可以让四个 Agent
同时行动。每个 Agent 使用独立的 participant client 和调用上下文，四个 Agent 都完成
当前轮行动后才调用一次 `env.step()` 结算并推进下一轮。这个参数只改变
调度方式，不改变 pair、观察、身份校验或全局请求上限。

## 离线完整 Baseline

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted --participants 4 --full-session --seed 2026 \
  --request-limit 2000 --max-output-tokens 512 --timeout-seconds 300 \
  --agent-concurrency 4
```

`--full-session` 忽略 `--completed-period-limit`，持续运行到 `RiskSharingEnv.finished`。
入口会保留所有已结算轮次的 `period_records.jsonl`，并在实际终止于第 48、49 或 50
轮后，用不消耗环境随机数的独立选择器随机选出一个已实现轮次，写入 `payment.json`。
其中包含 `selected_period` 和该轮每位参与者的结算记录；这是研究者侧支付轮数据，
不表示已经接入外部支付系统。

## Participant 决策边界

每个 `PersonAgent` 的 step 先通过真实 `ask_env(readonly=true)` 和身份适配层读取自己的原始
`get_observation`。这一步是固定的只读请求，直接调用环境工具，不需要 Router API。
随后由该 Agent 的 participant client 选择 amount。参与者可以先用自然语言
说明理由，再在 `ask_env` 的 instruction 末尾各写一行 `decision_type: transfer`、
`decision_id: <本轮凭证>` 和 `amount: <自己选择的整数>`，无需拼写 JSON。
原来的合法 `RISK_SHARING_PARTICIPANT_DECISION` JSON envelope 仍可使用。
项目覆盖了框架默认的 `ask_env` 工具说明，要求直接填写实际值，避免默认的模板/variables
说明与参与者提示词冲突。为兼容旧响应，也接受参与者明确给出的 `variables`，以及同一行
上的三个带标签字段；缺失、冲突或非整数金额仍然拒绝，不从自然语言推测金额。

`RiskSharingRouterAdapter` 在 Router 调用前绑定这三个明确的值和真实身份。Router 模型
负责生成环境工具调用，环境拒绝参数缺失、改写 amount 或伪造身份。Router 工具成功后由
程序结束路由循环，直接返回原始 receipt，不再调用摘要模型重写结果。
当前入口关闭了主观记忆；确认 receipt 与原始选择一致后，程序直接结束该参与者的行动，
无需再调用参与者 API 生成一次 `finish`。低收入参与者仍由自己的模型观察后决定等待。
失败请求会返回给同一个参与者模型修正；不会自动补金额。重试受 `--agent-max-turns`、
`--router-max-steps` 和全局 `--request-limit` 限制。

`agent_actions.jsonl` 同时记录 `participant_model_message.observation`、
`participant_model_output.amount`、`router_action` 和 `environment_receipt.amount`，
可直接检查 participant 输出与环境实际接受值相等。低收入 participant 只完成等待，
不会提交 transfer。`participant_raw_request` 保存成功解析的参与者原文；解析失败的
原文和原因以 `participant_decision_rejected` 记录，并作凭据脱敏。
兼容旧模板时，`participant_variables` 只记录参与者填写的三个经济字段，便于核对来源。
观察记录标记为 `router_action=direct_observation`、`router_model_called=false`；
转账记录仍标记为 `router_forward`、`router_model_called=true`。
manifest 分别统计 `participant_model_requests` 和 `router_model_requests`；在 scripted
模式下这些是本地模拟调用，不是外部 API 请求。
`run.log` 的 `model_call_finished` 记录每次模型调用所属 Agent、participant/router 角色、
耗时和成功状态，不保存请求头或密钥。正常无重试时，每轮是 4 次参与者调用、2 次 Router
调用，共 6 次；优化前是 12 次。实际耗时还取决于服务延迟、输出长度和格式重试。

## 真实 API（填写配置后再运行）

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend llm --participants 4 --full-session --seed 2026 \
  --request-limit 2000 --max-output-tokens 2048 \
  --agent-concurrency 4 \
  --timeout-seconds 21600 --request-timeout 60
```

整场实验的默认运行时限是 6 小时（21600 秒），可通过 `--timeout-seconds` 调整，
例如 `--timeout-seconds 86400` 表示 24 小时。`--request-timeout` 单独控制一次 API
请求的超时，默认仍为 60 秒。延长总时限不会改变实验轮数或模型请求次数上限。

该命令会运行到真实随机终止，调用参与者和 Router 模型；请求上限涵盖两者，单次输出上限和总时限由参数明确控制；
鉴权失败、余额不足或预算耗尽会停止，不会自动换供应商或循环重试。填写 `.env`
后再运行，并单独核对 `manifest.json` 中的 `success`、`stop_reason`、
`completed_periods` 和 `payment.json`。

## 测试

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m pytest -q
```

测试包括协议/环境单元测试、配置安全检查、离线 scripted smoke 链路和完整自然终止
链路；不把离线 scripted 结果当作外部 LLM API 成功。
