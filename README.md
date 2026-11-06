# AgentSociety2 风险分担实验复现

这是我目前在做的一个实验原型：用 AgentSociety2 和外部 LLM 模拟两个人在收入不确定时如何分担风险。当前实现了 Baseline、25V 自愿保险和 25F 强制保险的流程。25F 的原论文分配概率仍待确认，统计结果的复现也还没有完成。

## 我现在主要在检查什么

原来的调用方式里，参与者通过 `ask_env` 请求环境，但实际的 observation 读取和转账金额选择可能落到 `ReActRouter` 背后的模型上。这样最后看到的金额，无法确定是某个参与者自己决定的，还是 Router 代替它决定的。

这个版本希望把边界固定成：

```text
Agent 1 的 PersonAgent
    -> Agent 1 的 participant model 看到自己的 observation
    -> Agent 1 选择 amount
    -> RiskSharingRouterAdapter 绑定身份和原始决定
    -> ReActRouter 的模型整理参数并调用工具
    -> RiskSharingEnv 校验并保存
```

其他 Agent 走同样的流程。`llm` 模式下，参与者模型和 Router 都会调用 API。
金额由参与者选择，Router 负责工具调用；环境会拒绝 Router 改写金额、凭证或身份的请求。
读取 observation 是适配层校验身份后的直接环境调用，不再请求 Router 模型。转账提交仍使用
Router API；收到并核实成功回执后由程序结束行动，省去只为说“结束”而调用模型的步骤。

## 目前已经能运行的部分

三个实验入口默认使用 4 个 Agent，也接受不小于 2 的偶数人数。例如 `--participants 20` 会随机组成 10 个固定 pair，配对在整场实验中保持不变。每个 pair 每轮有一个人拿到随机收入 200，另一个人拿到 0；两个人的固定收入都是 125。只有拿到 200 的人可以转账，金额必须是严格的非负整数，并且不能超过自己的当轮转账前收入。

25V 在第 16 轮、随机收入产生之前增加一次自愿投保选择。每个 Agent 独立选择，自己提交前看不到伙伴的选择；双方提交后只在本 pair 内公开。该选择持续用于第 16—30 轮，每轮投保者支付 25、随机收入为 0 时获得赔付 50。第 31 轮才通知保险撤销。前 15 轮的参与者 observation、提示词和可用工具均不预告保险，第 16 轮也不预告撤销时间。

25F 的保险条款和生效时段与 25V 相同，但第 16 轮由电脑分配保险，参与者不做投保选择。每组至少一人有保险，分配保持到第 30 轮；转账仍由各自的参与者模型决定。原论文可确认的规则没有给出三个合法分配组合的概率，因此入口要求显式提供概率，测试配置会与论文参数区分记录，代码不设置默认概率。

`PersonAgent` 使用项目自己的 `RiskSharingParticipant` 类。参与者先拿到当前 observation，
可以用自然语言说明自己的考虑，再通过 `ask_env` 的 `instruction` 提交明确选择：

```text
这里可以写参与者自己的理由。
decision_type: transfer
decision_id: <当前 observation 中的 id>
amount: <参与者选择的整数>
```

参与者不再需要手写 JSON，原来的合法 JSON envelope 仍然兼容。格式不合法时不会提交，
同一个参与者模型可以根据错误反馈修正，重试受行动轮数和请求预算限制。
工具说明已改为填写实际值，不再要求模型同时填写模板和变量。旧响应如果通过 `variables`
明确给出了三个经济字段，也可以解析；字段缺失或彼此冲突时不会由程序补齐。

`RiskSharingRouterAdapter` 从 AgentSociety2 的 context 读取真实身份，固定参与者明确给出的三个值，再让 Router 调用工具。环境还会检查 pair、period、`decision_id`、金额范围和重复提交。`submit_decision` 只记录决定，统一结算和推进 period 由 `env.step()` 完成。

参与者能看到自己的收入、partner 的有限信息、本 pair 的历史和当前合法转账范围。`treatment`、`analysis_phase`、`session_id` 以及其他 pair 的数据不会放进 participant observation。低收入参与者没有 transfer opportunity，只会等待。

原版和[固定偏好版](experiments/fixed_preferences_v1/README.md)共用同一套信息呈现：先列当前任务和双方收入，再按轮次列出谁给谁转了多少、双方最终收入。完整 observation 以 JSON 对象保留在同一条输入中，历史不截断。保险扣费和赔付已经计入转账前收入；有转账机会时只展示自选金额 `x` 对应的收支关系，不提供建议金额。这一步由本地代码完成，不增加模型调用，也不改变两套实验各自的偏好设置。

所有实验入口直接使用这一格式，无需额外参数。运行记录中的 `observation_presentation=readable_v1` 是呈现版本标记，便于核对实际输入。

离线验证时，`ScriptedLLMClient("participant")` 根据各自 observation 生成测试金额；
`ScriptedLLMClient("router")` 只转发已有参数，两者都不联网。运行日志分别记录参与者原文、
选择的金额、Router 工具动作和环境 receipt，并统计两类模型的调用次数。

当前虚拟环境实际安装的是 **`agentsociety2==2.9.0`**，项目声明和 `uv.lock` 也已固定为这个版本。

## 目录说明

```text
custom/agents/risk_sharing_participant.py   PersonAgent 子类
custom/agents/risk_sharing_participant.md   参与者 prompt
custom/agents/risk_sharing_insurance.md     仅投保任务开放的补充 prompt
custom/agents/observation_presentation.py   所有参与者共用的信息呈现
custom/envs/risk_sharing_env.py             Baseline / 25V / 25F 环境和 Router 适配器
custom/insurance_assignment.py             25F 联合分配概率的显式校验
custom/config.py                            scripted/LLM 配置
experiments/baseline/run_smoke.py           smoke 和 full-session 入口
experiments/25v/run_smoke.py                25V 入口，复用既有调度、配置和支付记录
experiments/25v/report.py                   保险决策与扣费赔付报告，可更新历史摘要
experiments/25f/run_smoke.py                25F 入口，要求明确提供分配概率
experiments/25f/report.py                   电脑分配与扣费赔付报告
protocol/paper_spec.py                      冻结的论文协议
tests/test_baseline_integration.py         AgentSociety2 集成测试
tests/test_risk_sharing_env.py             环境、身份和隔离测试
tests/test_paper_spec.py                   协议测试
tests/test_25v.py                          保险规则和真实框架离线集成测试
tests/test_25f.py                          强制分配、隔离与真实框架离线集成测试
```

## 安装

项目使用 Python 3.11 及以上版本。已经有 `.venv` 时可以直接运行；重新准备环境可以使用锁文件：

```bash
cd /home/zhicheng/risk_sharing_reproduction
uv sync --locked
```

真实 API 配置放在项目根目录的 `.env` 中。可以从模板开始：

```bash
cp .env.example .env
```

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
```

`.env` 不会提交到 Git。`scripted` 模式不调用外部 API；`llm` 模式使用这里配置的服务。

## 离线运行

单轮验证：

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted \
  --participants 4 \
  --completed-period-limit 1 \
  --seed 2026 \
  --request-limit 100 \
  --max-output-tokens 512 \
  --router-max-steps 6 \
  --agent-max-turns 4 \
  --agent-concurrency 4 \
  --timeout-seconds 180
```

跑到 Baseline 自然结束：

```bash
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted \
  --participants 4 \
  --full-session \
  --seed 2026 \
  --request-limit 2000 \
  --max-output-tokens 512 \
  --router-max-steps 6 \
  --agent-max-turns 4 \
  --agent-concurrency 4 \
  --timeout-seconds 300
```

`--agent-concurrency` 是每一轮的并发上限。默认值为 `1`，最大可设为本次参与者人数，
例如 `--participants 20 --agent-concurrency 20` 让 20 个 Agent 同时完成自己的行动。
每个 Agent 有独立的 participant client 和模型调用上下文；
所有 Agent 都结束当前轮后，环境才统一结算和推进，所以并发不会把不同 Agent 的
observation 或经济决定混在一起。全局 `--request-limit` 对参与者和 Router 模型调用共同生效。
正常无重试、未启用额外决策审视时，每组每轮有 2 次参与者调用和 1 次转账 Router 调用，
10 组共 30 次。并发上限约束同时行动的 Agent 数；每个 Agent 内部仍先观察、后决定、再提交，
实验轮次仍按顺序推进。模型服务自身的限流也会影响实际速度。

整场实验默认最多运行 6 小时（21600 秒）。`--timeout-seconds` 可以覆盖这个默认值，
例如设为 `86400` 可运行最多 24 小时；上面的离线验证命令显式使用了较短时限。
单次 API 请求的超时由 `--request-timeout` 单独控制，默认是 60 秒。

每次运行会在 `experiments/baseline/runs/` 下创建一个目录，里面有：

- `manifest.json`：运行参数、信息呈现版本和最终状态；
- `agent_actions.jsonl`：每次环境工具调用，以及 participant、适配器、环境三方的记录；
- `period_records.jsonl`：环境结算后的数据；
- `summary.md`：摘要，顶部提供交互图表入口；
- `report.html`：Baseline、25V、25F 每次运行自动生成的离线交互报告，可按轮次范围比较各组的转账、收入差和转账频率，支持多选、热图、排序和 CSV 导出，也可查看单组轨迹；25V 展示购买选择，25F 展示电脑原始分配，两者均汇总保费和赔付；
- `run.log`：脱敏后的诊断日志，包括每次模型调用的 Agent、角色、耗时和成功状态；
- `payment.json`：只有 `--full-session` 成功结束时才会生成，表示随机选中的研究支付轮，不是外部支付系统。

看金额来源时，重点对照 `agent_actions.jsonl` 里的这些字段：

```text
decision_source
participant_model_message.observation
participant_model_output.amount
submitted_amount
environment_receipt.amount
router_action
router_model_called
```

一次成功的 transfer 应满足 `decision_source == "participant_model"`、`router_model_called == true`，
并且 participant 输出、Router 提交参数和环境 receipt 中的金额相同。`participant_raw_request`
保留参与者提交的原文；格式失败记录的 `action` 为 `participant_decision_rejected`。
旧模板响应中明确填写的经济字段另存为 `participant_variables`。观察读取记录的
`router_model_called` 为 `false`，表示这一步没有请求模型。

## 测试

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m pytest -q
```

测试覆盖了协议常量、pair 和 history 隔离、真实 AgentSociety2 集成链路、身份伪造、旧 `decision_id`、非法金额、低收入 Agent、格式错误后的修正，以及 Router 改写参与者金额或凭证时被环境拒绝等情况。

## 后续工作

三个实验均可配置多个固定 pair，20 人 / 10 组的运行命令分别见 [Baseline](experiments/baseline/README.md#真实-api填写配置后再运行)、[25V](experiments/25v/README.md#20-人--10-组的真实模型运行) 和 [25F](experiments/25f/README.md#20-人--10-组的真实模型运行)。25F 还需要原始实验程序或作者资料来确认分配概率；目前的离线验证使用显式测试参数。后续还需按研究设计准备样本，进行重复实验和统计回归。

参与者提示词和策略设定还在调整。后续会通过重复实验检查 LLM 的运行稳定性、调用成本和实验结果，再补充统计分析、回归和结果图表。`scripted` 模式中的策略用于测试程序流程。

支付部分目前只生成随机选中的轮次及对应收益记录，尚未接入实际支付系统。

论文规则统一记录在 [`protocol/paper_spec.py`](protocol/paper_spec.py) 中，保持冻结。运行说明见 [Baseline](experiments/baseline/README.md)、[25V](experiments/25v/README.md) 和 [25F](experiments/25f/README.md)。
