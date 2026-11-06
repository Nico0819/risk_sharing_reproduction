# Baseline 实验入口

这是当前 Baseline 的实验入口：默认四个独立 `PersonAgent`、两个固定 pair，
通过 AgentSociety2 2.9.0 的 `PersonAgent`、项目适配层和真实 `ReActRouter` 进入同一个
`RiskSharingEnv`。参与者选择金额，Router 模型只负责整理参数和调用工具。
入口有两种模式：scripted 模式用于快速验证，llm 模式连接参与者和 Router 模型；两种模式都
支持 smoke 和 full-session 运行。`--participants` 接受不小于 2 的偶数，设为 `20`
时随机组成 10 个固定 pair；每个 Agent 的身份、模型上下文和本组历史仍独立。

参与者输入由 [`observation_presentation.py`](../../custom/agents/observation_presentation.py) 统一呈现，
原版和固定偏好版的三个实验都使用这套格式。当前任务、双方收入、decision ID 和合法范围放在前面，
历史按轮次列出发送者、接收者、金额和双方最终收入；完整 observation 以 JSON 对象同时保留。
有转账机会时只给出自选金额 `x` 的收支关系，不建议具体金额。呈现由本地代码完成，
无需额外参数或模型调用，也不向原版注入固定偏好。`observation_presentation=readable_v1`
在运行记录中表示呈现版本，不是可切换的选项。

## 配置

配置加载器固定从项目根目录读取：
`/home/zhicheng/risk_sharing_reproduction/.env`。请先复制/编辑 `.env.example`，
只填写：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
```

入口将同一份 LLM 配置用于参与者和 Router 的独立客户端，并关闭供应商原生 thinking。真实密钥不会
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
`agent_actions.jsonl`、`period_records.jsonl`、`summary.md`、交互图表 `report.html` 和诊断日志 `run.log`。

每次运行都会自动生成 `report.html`，在 summary 顶部点击“打开交互图表报告”，用浏览器查看。
报告可离线使用，支持按组或 Agent 搜索、分页浏览和切换逐轮转账/收入曲线。“组间对比”可多选
组别、选择轮次范围，比较平均转账、绝对收入差、每人平均收入和有转账的轮次比例，并导出当前
对比表的 CSV。组数超过 6 时用热图保留各组数据，趋势图展示每轮组间均值和实际取值范围；
勾选不超过 6 组时可直接叠加各组曲线。数据按真实配对生成，展示层没有两组上限。
历史运行可单独更新图表，不重新调用模型：

```bash
.venv/bin/python -m experiments.report --run-dir experiments/baseline/runs/<运行目录>
```

历史刷新保留原始记录，并在首次添加图表入口前保存 `summary.before-visual-report.md`。

`--agent-concurrency` 控制同一轮最多同时运行多少个参与者。默认值是 `1`，方便复现
原来的逐个运行方式；最大可设为参与者总人数。`--participants 20 --agent-concurrency 20`
可让 20 个 Agent 同时行动。每个 Agent 使用独立的 participant client 和调用上下文，所有 Agent 都完成
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

## 可选的决策审视

在运行命令中加上 `--participant-deliberation`，参与者会在转账或投保前多调用一次自己的
模型：先根据当前 observation 和本组已结算历史写一段简短评估，再带着这段评估决定最终
行动。等待者不增加调用；提交格式错误时保留已有评估，由同一个参与者修正请求。

```bash
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted --completed-period-limit 2 --participant-deliberation \
  --agent-concurrency 4 --request-limit 100 --max-output-tokens 512
```

25V、25F 入口也接受这个参数。原有命令不加它就保留原来的行为，方便作为对照。
这不是启用供应商的原生推理模式，也没有增加跨轮的私人记忆；下一轮仍从环境收到本组历史。
评估提示词放在 `custom/agents/risk_sharing_deliberation.md`，不规定转账金额或投保策略。
评估阶段没有工具，文本中的金额不会被自动提交；最终选择仍由参与者的后续模型调用产生，
Router 只接收最终提交请求。空白、截断或包含工具调用的评估会报错，不会自动补决定。

四人正常一轮的调用从 6 次增加到 8 次，其中新增 2 次是高收入参与者的评估；25V 第 16 轮
投保还会额外增加 4 次评估。无重试时，完整 Baseline 为 384—400 次，25V 为 396—412 次。
这些调用都计入 `--request-limit` 和 token 用量；评估单次输出上限为
`min(512, --max-output-tokens)`。运行会更慢，实际费用取决于模型和历史长度。

开启后，`participant_deliberations.jsonl` 记录每次评估的 Agent、轮次、decision_id、
输入消息和评估文字，可与 `agent_actions.jsonl` 中相同凭证的最终选择核对。
该文件是研究者日志，不会提供给搭档或其他参与者；评估文字只回到该 Agent 当前任务的上下文。
`run.log` 的 `phase` 区分评估、决策和路由；`manifest.json` 保存开关、评估请求数与提示词哈希。
模型给出的评估只是决策说明，不等于完整的内部思考，也不能据此认定行为更接近人类。

## Participant 决策边界

参与者的工具执行入口只允许 `ask_env` 和 `finish`。模型即使自行输出未展示的工具名，
例如文件读写、技能脚本或消息发送，也会在执行前被拒绝。通用 skills 的可见和激活列表
在恢复 Agent 和每轮行动前均清空；不依赖 `disabled_skill_ids: ["*"]`，因为已安装的
AgentSociety2 2.9.0 不把这个值当作通配符。拒绝工具后，参与者仍可修正为合法行动。

参与者原文和私人评估不会作为搭档 observation 的内容；搭档只能看到协议允许公开的
收入、保险结果及本组已结算历史。转账金额带来的行为反馈是实验的一部分。
当前隔离针对模型消息和工具权限，所有 Agent 仍由同一个 Python 进程运行，
并非各自的操作系统沙箱。以后添加工具或技能时，需要重新检查这个执行边界。

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
耗时和成功状态，不保存请求头或密钥。正常无重试且不加决策审视时，每组每轮是 2 次参与者
调用、1 次 Router 调用；10 组为每轮 30 次。实际耗时还取决于服务延迟、输出长度和格式重试。

## 真实 API（填写配置后再运行）

以下命令运行 20 个 Agent / 10 个固定 pair，Agent 并发上限为 20，整场最多运行 24 小时：

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend llm --participants 20 --full-session --seed 2026 \
  --request-limit 20000 --max-output-tokens 2048 \
  --router-max-steps 4 --agent-max-turns 4 \
  --agent-concurrency 20 \
  --timeout-seconds 86400 --request-timeout 120
```

不加 `--participant-deliberation` 时，20 人完整 Baseline 无重试约需 1440—1500 次
模型调用；`20000` 是请求总数上限，不是要求调用这么多次。若启用额外决策审视，正常完整运行
约需 1920—2000 次。并发上限包含等待者，转账者内部的决定与 Router 提交仍依次执行。
如果服务端因并发返回限流错误，可将 `--agent-concurrency` 降为 `10` 或 `5`，人数仍保持 `20`。

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
