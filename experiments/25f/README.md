# 25F 强制保险实验

25F 沿用 Baseline 的固定配对和转账机制，在第 16 轮由环境随机决定每组谁有保险。
每组可以双方获保，也可以只有一方获保，不允许双方都没有保险。参与者不能购买、
拒绝或更改这次分配；转账金额仍由各自的参与者模型决定。

原论文的精确分配概率尚未从资料中确定，冻结的 `protocol/paper_spec.py` 仍将其记为
`None`。因此这个入口要求研究者明确提供三个概率，不设隐含默认值，也不把测试
配置标成论文参数。默认使用 4 个 Agent、2 个固定 pair，也接受不小于 2 的偶数人数；
例如 20 个 Agent 组成 10 个固定 pair。论文 25F 使用 40 人，样本设计和统计复现仍需后续工作。

已核对[本地发表版论文](../../papers/1-s2.0-S0167268119301192-main.pdf)第 2 节、Table 1 和
脚注 8，以及[作者公开的完整稿及在线附录](https://wengxi125.weebly.com/uploads/4/4/6/5/44658341/draft-0303.pdf)。
这些材料确认电脑随机分配和排除双方未投保，但未给具体抽样概率。Table 2 的投保率是
实现后的样本比例，不能当作分配概率。出版商补充附件目前未成功取得，原始抽样方法仍需核实。

## 时序与收入

| 轮次 | 参与者看到什么 | 环境如何处理 |
| --- | --- | --- |
| 1—15 | 与 Baseline 相同，只能转账或等待 | 不显示保险信息，不调用保险分配随机数 |
| 16 | 本人与搭档的分配结果、当前保险条款与收入 | 收入冲击之前，按配置概率为每个 pair 抽一次保险状态；没有投保决策子阶段 |
| 17—30 | 当前收入、本人和搭档的保险状态、已结算 pair 历史 | 保留原分配，每轮自动扣保费、计赔付 |
| 31 起 | 当轮才通知保险撤销 | 不再扣保费或赔付；保留原分配和历史 |

保险每轮扣费 25；低随机收入时赔付 50，高随机收入时不赔付。因此有保险者的
转账前收入为高收入 300、低收入 150；无保险者仍为 325、125。只有获得 200
随机收入的人有转账机会，金额必须是非负整数，不能超过其当轮转账前收入。
`submit_decision` 只校验并保存决定，`env.step()` 统一结算和推进。

参与者在第 16 轮不会预知第 31 轮撤销。终止机制保持不变：1—47 轮不结束，
48、49 轮各有 0.78 概率结束，50 轮强制结束。

分配使用独立的随机数流，`--assignment-seed` 只控制保险分配，不改变 `--seed`
控制的收入和终止抽样。三个分配概率依次对应双方获保、仅 pair 第一位成员获保、
仅第二位成员获保；成员顺序沿用环境的固定配对顺序，概率必须非负且合计为 1。
当前配置方式是按给定联合概率为各 pair 独立抽样，不声称已还原作者的原始抽样程序。

## 调用边界

参与者读取自己的 observation 后选择转账或等待。观察直接从经过身份验证的
适配层返回；转账仍经过真实 AgentSociety2 的 `PersonAgent → ask_env →
RiskSharingRouterAdapter → ReActRouter → RiskSharingEnv` 链路。
Router 只能提交参与者给出的原始金额和 decision_id，不能补金额或修改决定。
成功回执经核对后由程序结束当次行动。

当前收入与历史使用和 Baseline、25V、固定偏好版相同的[信息呈现](../baseline/README.md)：
历史按轮次标明发送者、接收者和金额，完整 observation 以 JSON 对象保留。保险由电脑分配这一来源
会明确显示，并区分当轮有效保障与原分配。呈现由本地代码完成，无需额外参数或模型调用；
运行记录中的 `observation_presentation=readable_v1` 是呈现版本标记。

保险分配由环境完成，不通过 participant 或 Router 模型，不制造投保决定、理由
或 receipt。模型提供的 agent_id 不能覆盖框架传入的身份，参与者只能看到自己
所在 pair 的合法信息。分配概率、实验 treatment 和研究者记录不进入 observation。

## 离线运行

下面的 `0.5 0.25 0.25` **只用于演示和工程测试，不是已经确认的论文概率**。
正式研究前需要从论文补充资料或作者信息中确定分配机制，并使用核实后的参数。
不提供概率时程序会报错，不会自行选择。

先运行至 31 轮，覆盖保险引入和撤销：

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.25f.run_smoke \
  --backend scripted --completed-period-limit 31 --seed 2026 \
  --forced-assignment-probabilities 0.5 0.25 0.25 \
  --assignment-seed 2026 --participants 4 --agent-concurrency 4 \
  --request-limit 400 --max-output-tokens 512 \
  --router-max-steps 6 --agent-max-turns 4 --timeout-seconds 300
```

完整运行至自然终止，只需将 `--completed-period-limit 31` 换为 `--full-session`：

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.25f.run_smoke \
  --backend scripted --full-session --seed 2026 \
  --forced-assignment-probabilities 0.5 0.25 0.25 \
  --assignment-seed 2026 --participants 4 --agent-concurrency 4 \
  --request-limit 400 --max-output-tokens 512 \
  --router-max-steps 6 --agent-max-turns 4 --timeout-seconds 300
```

以上命令显式选择 `scripted`，不调用外部 API；命令行必须指定 backend，不会自动启用
真实模型。Scripted 策略只替换模型响应，仍经过
真实框架和环境工具，用于检查调用、身份和结算边界。四个人无重试时，每轮共
6 次本地模拟模型调用：4 次参与者、2 次转账 Router；保险分配不增加模型调用。
完整 48—50 轮通常为 288—300 次，`--request-limit` 同时约束两类调用。

程序入口也可通过 `importlib.import_module("experiments.25f.run_smoke")` 加载，
向 `run_experiment` 传入 `forced_assignment_probabilities` 和可选的
`assignment_seed`。Python 普通 `from` 语句不能直接使用数字开头的包名。

## 运行记录

输出保存在 `experiments/25f/runs/<时间戳-ID>/`：

- `manifest.json`：本次配置、完成轮数、模型调用数、停止原因和分配记录数。
  `insurance_assignment_configuration` 保存三个概率、分配种子和来源；
  `probability_source=explicit_run_configuration`、`paper_probability_resolved=false`。
- `insurance_assignments.jsonl`：每位参与者一条原始分配，含 `agent_id`、`pair_id`、
  `partner_id`、`period=16`、布尔 `insured`、`assignment_source=environment_random`
  和 `treatment=25F`。正常进入 16 轮后每位参与者 1 条，20 人即 20 条；未到 16 轮时为空。
- `period_records.jsonl`：逐人逐轮的收入与转账，`insurance_assigned` 记录原分配
  （分配前为 `null`，撤销后保留）；`insured` 记录当轮有效保险；另有
  `insurance_cost` 和 `insurance_payout`。25F 不使用 `insurance_purchased`。
- `agent_actions.jsonl`：参与者原始转账选择、Router 实际调用和环境回执。
  随机保险分配另记为 `action=insurance_assignment`、`decision_source=environment_random`、
  `router_model_called=false`，不记作参与者的经济选择。
- `summary.md`：各 pair 双方的原分配、获保人数、实际保费和赔付，以及第 16 轮
  保险前、转账前和转账后的收入。缺少分配记录不会显示为拒保。
- `report.html`：每次运行自动生成的离线交互报告，入口在 `summary.md` 顶部。
  显示各组原始保险分配、获保人数、保费和赔付，并支持轮次筛选、多组选取、转账和收入
  比较、热图、逐轮轨迹及 CSV 导出。保险状态来自电脑分配，不显示为参与者购买选择。
- `run.log`：模型调用、阶段推进和错误；不输出 API 密钥。
- `payment.json`：仅自然终止时生成抽样轮次与收益记录，不进行实际转账付款。

保费、赔付与收入报告只汇总已结算记录，不推测尚未完成的轮次。第 31 轮后无有效
保险，并不表示第 16 轮没有获保。报告保留两者的区别。
`env.step()` 结算第 15 轮后会准备第 16 轮并分配保险。因此只完成 15 轮的 smoke
也会保留每位参与者已发生的分配记录，但没有第 16 轮的转账、保费或赔付结算；完成 14 轮时
尚未分配。网页同样保留原始分配，缺失或异常分配会明确提示，不从撤销后的状态推测。

已有运行可补生成或刷新网页，不重新运行实验或调用模型：

```bash
.venv/bin/python -m experiments.report --run-dir experiments/25f/runs/<运行目录>
```

首次添加网页入口前会保存 `summary.before-visual-report.md`；原始分配、结算、模型日志、
支付记录和 manifest 保持不变。生成的 `report.html` 用浏览器打开即可离线查看。

目前未提供完整 checkpoint 恢复；论文统计分析、回归和外部支付也不在这个入口中。
真实 LLM 的运行稳定性和成本需另行验证，离线通过不代表已经完成真实 API 实验。

## 20 人 / 10 组的真实模型运行

先在当前 shell 中设置 `INSURANCE_PROB_BOTH`、`INSURANCE_PROB_FIRST`、`INSURANCE_PROB_SECOND`
三个变量，依次为双方、仅第一位、仅第二位获保的概率。必须使用本次研究明确采用的概率，
不能将上方工程测试分布当作已核实的论文参数。变量未设置时，下方命令会停止，避免误用默认值。

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.25f.run_smoke \
  --backend llm --participants 20 --full-session --seed 2026 \
  --forced-assignment-probabilities \
    "${INSURANCE_PROB_BOTH:?请先设置双方获保概率}" \
    "${INSURANCE_PROB_FIRST:?请先设置仅第一位获保概率}" \
    "${INSURANCE_PROB_SECOND:?请先设置仅第二位获保概率}" \
  --assignment-seed 2026 \
  --request-limit 20000 --max-output-tokens 2048 \
  --router-max-steps 4 --agent-max-turns 4 \
  --agent-concurrency 20 \
  --timeout-seconds 86400 --request-timeout 120
```

整场时限为 24 小时，单次请求时限为 120 秒。参与者和 Router 使用根目录已有的 LLM 配置；
保险分配由环境完成，不增加模型调用。上例未开启额外决策审视，无重试完整运行约需
1440—1500 次模型调用；20000 是预算上限，不是固定调用次数。加 `--participant-deliberation`
后约需 1920—2000 次。若服务端限流，可保留 20 人，仅将 `--agent-concurrency` 降为 `10` 或 `5`。
