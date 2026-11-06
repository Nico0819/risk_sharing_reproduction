# 25V 自愿保险实验入口

第一版复用既有 `RiskSharingParticipant → RiskSharingRouterAdapter → ReActRouter → RiskSharingEnv`。
投保与转账都由各自的参与者模型选择，Router 只转发已绑定的原始值和身份。
未修改冻结的 `protocol/paper_spec.py` 或 AgentSociety2 平台包；25F 有[独立入口](../25f/README.md)。

参与者与 Baseline、25F 及固定偏好版共用[信息呈现](../baseline/README.md)：当前收入与任务放在前面，
历史逐轮标明谁给谁转了多少，完整 observation 以 JSON 对象保留。投保任务显示当前保险条款，
转账任务使用已经计入保费和赔付的收入。无需额外参数或模型调用；运行记录中的
`observation_presentation=readable_v1` 是呈现版本标记。

## 实验时序

| 论文轮次 | 参与者当前任务 | 环境处理 |
| --- | --- | --- |
| 1—15 | 转账或等待，与 Baseline 相同 | 不提供任何保险说明或保险工具 |
| 16：投保子阶段 | 每人独立选择购买或不购买 | 尚不抽收入；只保存决定，不能修改已接受的选择 |
| 16：转账子阶段 | 高随机收入者转账，其余等待 | 所有投保决定收齐后由 `step()` 抽收入、扣保费、计赔付；仍是第 16 轮 |
| 17—30 | 转账或等待 | 自动沿用第 16 轮的投保选择，每轮重新计算保费和赔付 |
| 31 起 | 转账或等待 | 第 31 轮才通知撤销；之后不扣保费、不赔付，配对与历史保留 |

依据论文 Section 2 / Table 1 和 Appendix C 的 Fig. C.4—C.8：

- 第 16 轮在收入冲击之前投保。`purchase` 必须是布尔值，缺交不等于拒保。
- 伙伴提交与否和选择在自己决定前均不可见；同一 pair 双方提交后才可见双方选择。提交接口始终只返回本人的确认。
- 投保成本每轮 25；低随机收入时赔付 50，高随机收入时赔付 0。投保者转账前收入因此为高收入 300、低收入 150；未投保者仍为 325、125。
- 25V 允许双方都不投保。不能套用 25F 的限制或猜测其分配概率。
- 参与者第 16 轮仅知道选择将自动延续，不获知第 30 轮结束保障、第 31 轮撤销。
- 真实结束机制不变：1—47 不结束，48、49 各以 0.78 概率结束，50 强制结束。分析阶段字段仅覆盖 1—45。
- 一个 `step()` 在第 16 轮可能仅推进子阶段，不结算、不抽结束、不增加论文轮次。收入、结束随机数序列不受决定 token 和保险选择影响。

提示词：基础说明是 `custom/agents/risk_sharing_participant.md`；仅当当前任务为
`insurance` 才附加 `risk_sharing_insurance.md`，格式为：

```text
decision_type: insurance
decision_id: <当前 observation 的凭证>
purchase: <true 或 false>
```

投保和转账使用不同凭证，支持同内容重试并拒绝改选、冒名、过期凭证以及 Router 改写。
金额和布尔值不互相转换。平台 2.9.0 缓存工具列表，项目级 ReActRouter 会在调用边界
刷新当前阶段的 schema；保险工具不会出现在前 15 轮或 Baseline 的模型上下文中。

## 离线运行

工作目录是 `/home/zhicheng/risk_sharing_reproduction`，不需要填写或使用 API 密钥。
默认是 4 个 Agent / 2 个固定 pair，也接受不小于 2 的偶数人数。
`--participants 20 --agent-concurrency 20` 可运行 10 个固定 pair，让 20 个 Agent 同时行动。
论文 25V 的样本为 36 人，使用其他人数时应在研究记录中注明。

先运行到 31 轮，覆盖保险引入和撤销（默认 smoke 目标也是 31）：

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python experiments/25v/run_smoke.py \
  --backend scripted --completed-period-limit 31 \
  --request-limit 400 --max-output-tokens 512 \
  --agent-concurrency 4 --timeout-seconds 300
```

完整运行至自然终止，并抽取一个实际结算轮次生成支付记录：

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python experiments/25v/run_smoke.py \
  --backend scripted --full-session --seed 2026 \
  --request-limit 400 --max-output-tokens 512 \
  --agent-concurrency 4 --timeout-seconds 300
```

`scripted` 只替换模型响应，仍运行已安装 AgentSociety2 的真实 Agent、Router 和工具调用。
测试策略按 Agent 编号给出确定性投保选择，只用于覆盖程序流程，不代表经济行为模型。
上述两个 pair 无重试时，每个转账轮次 6 次本地模拟模型调用，第 16 轮投保额外 8 次。
因此完整 48—50 轮共 296—308 次；`--request-limit` 同时约束参与者和 Router。

如需参与者先结合历史评估再决定，可在命令中加 `--participant-deliberation`。
每次转账或投保多一次参与者模型调用，等待不增加调用；完整 25V 无重试时需 396—412 次，
建议预留更高的请求上限，例如 `--request-limit 600`。评估只返回本人当前任务的上下文，
另存到 `participant_deliberations.jsonl`；不会把投保阶段的私人评估提供给伙伴。
这是额外的决策审视步骤，不是原生 thinking 开关或跨轮私人记忆。
说明与离线示例见 [Baseline 入口](../baseline/README.md#可选的决策审视)。

## 输出与检查

每次创建独立目录 `experiments/25v/runs/<时间戳-ID>/`：

- `manifest.json`：treatment、完成轮数、投保决定数、模型调用数、成功/停止原因。
- `insurance_decisions.jsonl`：接受的原始投保选择，供研究者核对；正常越过 16 轮每位参与者应有 1 条，20 人即 20 条。
- `agent_actions.jsonl`：参与者原文、选择、真实工具动作和 receipt；投保字段为 `purchase`，不是转账 `amount`。
- `period_records.jsonl`：每人每轮的收入、转账、保险购买历史、当轮有效状态、保费和赔付。`insurance_purchased` 在首次选择前为 `null`，撤销后保留原选择；`insured` 则在撤销后为 `false`。
- `run.log`：阶段切换、结算、错误和模型调用耗时，不包含密钥。
- `summary.md`：本次结果摘要；按 pair 列出成员的购买选择、购买人数、双方/单方购买情况及该组已结算保险期的保费和赔付。另保留逐个 Agent 的原始理由与回执、各阶段费用汇总和第 16 轮的收入明细。
- `report.html`：每次运行自动生成的交互图表报告，入口在 summary 顶部。用浏览器打开即可离线查看单组记录和组间对比，也可以下载图表与当前对比的 CSV。
- `payment.json`：仅自然终止的 full-session 生成，收益已包括保费、赔付和转账；不代表实际付款。

成功的完整运行应为 `success=true`、`stop_reason=natural_termination`，并有 48、49 或 50 个已结算轮次。
只运行到 31 轮则是 `smoke_period_limit`，不会生成支付抽样结果。

如果所有 Agent 都明确拒保，保险期内保费和赔付都为零是合法结果，不能将其误认为
没有引入保险。摘要会分别显示“已提交的拒保”和“没有提交/尚未到达保险阶段”。
每组购买情况沿用第 16 轮的原始选择，第 31 轮撤销保险后不会改写成“未购买”。
`scripted` 与 `llm` 的决策来源会明确标注；原始理由是模型的解释，不代表其经济判断正确。

历史运行也可以仅重新生成保险展示，不重新请求模型、不改变原始记录：

```bash
.venv/bin/python experiments/25v/report.py --run-dir experiments/25v/runs/<运行目录>
```

该命令更新 `summary.md` 的保险区块和图表入口，并生成 `report.html`；首次更新摘要前保存
`summary.before-insurance-report.md`。原始决定、结算和日志不变，不调用模型。

报告中的“组间对比”默认选中全部组，也可以搜索、勾选需要比较的组。四个指标分别是
平均每轮转账、平均绝对收入差、平均每人最终收入和有转账的轮次比例。轮次范围可以选择
1—15、16—30、31—45，也可以自行输入起止轮次；第 46 轮及以后单独列出，不并入第 31—45 轮。

选中不超过 6 组时，趋势图分别画出各组曲线；选中超过 6 组时，热图保留每组逐轮数据，
趋势图显示每轮各有效组的等权均值和实际最小—最大范围。这个范围只是数据的分布，不是置信区间。
下方对比表可以点击表头排序；“下载当前对比 CSV”导出当前所选组和轮次范围内的全部指标、
有效轮数，顺序与表格一致。

收入比较只使用同轮两位成员收入都完整的记录，缺失记录不补零，收入和转账分别按各自有效轮数
计算。单组记录仍可通过搜索和分页查看，保险引入、撤销会标在图上，撤销不会改写原购买选择。
报告按记录中的真实组数和成员生成，已覆盖 10 组及 40 组的数据测试；这套展示也用于 Baseline。
运行入口也支持按偶数人数配置多个 pair，报告直接使用本次实际配对。
旧目录迁移后，原始 manifest、日志内记录的旧路径保留为运行当时的信息，不改写历史数据。
Python 普通 `from` 语句不能使用数字开头的包名，测试通过 `importlib.import_module`
加载 `experiments.25v`；命令行按文件路径或 `python -m` 运行均不受影响。

## 真实模型与待验收事项

使用同一入口的 `--backend llm` 可运行真实模型；它读取项目根目录已有 `.env` 中的 `LLM_API_KEY`、`LLM_BASE_URL`、
`LLM_MODEL`，不改变服务商或模型，不覆盖配置。真实调用会收费，离线通过不代表 API 已通过。

### 20 人 / 10 组的真实模型运行

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.25v.run_smoke \
  --backend llm --participants 20 --full-session --seed 2026 \
  --request-limit 20000 --max-output-tokens 2048 \
  --router-max-steps 4 --agent-max-turns 4 \
  --agent-concurrency 20 \
  --timeout-seconds 86400 --request-timeout 120
```

20 个 Agent 的行动可同时进行，每轮所有人完成后再统一推进；第 16 轮先收齐投保选择，
再进入本轮转账。总时限为 24 小时，单次请求时限为 120 秒。上例未开启额外决策审视，
无重试完整运行约需 1480—1540 次模型调用；20000 是预算上限。加 `--participant-deliberation`
后约需 1980—2060 次。实际速度取决于服务端延迟和限流；遇到限流可保留 20 人，
仅将 `--agent-concurrency` 降为 `10` 或 `5`。

仍待验证：真实模型的投保格式与运行稳定性、更多样本和重复 session、论文统计指标与回归。
目前没有完整 checkpoint/断点恢复；最终支付仅是抽轮与收益记录，未接外部支付。
参与者路径仅支持此处验证的 ReActRouter，不能将真实环境对象直接交给 CodeGenRouter。

```bash
.venv/bin/python -m pytest -q
```
