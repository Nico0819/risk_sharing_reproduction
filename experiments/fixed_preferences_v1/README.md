# 固定偏好版本：风险厌恶与优势不平等厌恶

这个版本给每个参与者增加两项整场保持不变的个人偏好。偏好随本人的 observation 一起传给 participant model，具体金额和保险选择仍由模型产生。

`experiments.baseline.run_smoke`、`experiments.25v.run_smoke` 和 `experiments.25f.run_smoke` 是没有个人偏好配置的基准入口。这里使用单独的 runner、Agent 子类、提示词和运行目录，与基准共用信息呈现、环境规则、身份验证、Router 和报告模块。两套 runner 是独立文件，后续修改调度或日志时需要同步核对；代码指纹用于记录实际运行版本。

## 更清楚的 observation 呈现

本入口及不带固定偏好的三个入口统一使用 [`observation_presentation.py`](../../custom/agents/observation_presentation.py)，运行命令无需增加参数。

新格式先展示当前任务、本人和搭档的收入账户、当前 decision ID 和合法范围，再按轮次列出发送者、接收者、转账金额及双方最终收入。发送者由真实转账资格确定，金额为零时也能区分“选择不给”和“没有转账机会”。

有转账机会时，输入只列出由本人选择的金额 `x` 对应的会计关系，例如本人收入 `300 - x`、搭档收入 `125 + x`；不提供建议金额，也不要求均分。保险扣费和赔付已计入当前收入，会明确提醒不要重复计算。保险引入之前不会展示未来的保险安排。

完整 observation 以普通 JSON 对象保存在消息的 `<risk_sharing_turn>` 块中。完整历史没有截断；表格只是同一信息的另一种显示方式。原始数据使用紧凑 JSON，减少重复排版带来的长度。呈现由本地代码完成，不增加模型调用次数。

`manifest.json`、`summary.md` 和 `participant_contexts.jsonl` 中的 `observation_presentation=readable_v1` 记录呈现版本，不是配置开关。呈现代码也有源文件指纹；核对模型实际输入时，应同时查看消息日志和代码指纹。已有运行产物保留原样。

信息呈现不引入私人记忆或新偏好，也不改变 Router、金额选择、环境规则和结算。离线测试验证的是信息保留和调用兼容性；真实 LLM 能否正确理解收入和历史，还需要独立的理解测试。

## 偏好怎么进入决策

| 字段 | 含义 | 可选值 |
|---|---|---|
| `risk_aversion` | 对自己最终所得的不确定性有多在意 | `low`、`medium`、`high` |
| `advantageous_inequality_aversion` | 对自己转账后收入高于搭档有多在意 | `low`、`medium`、`high` |

每个等级对应固定的英文描述，见 [`profiles.py`](../../custom/agents/fixed_preferences_v1/profiles.py)。这些等级是本版本的研究配置，不是论文估计出的参数，也不代表已验证的人类偏好分布。配置不会直接计算转账金额，不要求均分，也不规定是否投保。

每人的配置保存在自己的 `agents/agent_XXXX/config.json` 中。框架恢复 Agent 时读取配置；每次构造模型消息时校验配置指纹，再将本人偏好放入 system 消息的 `<fixed_preferences>` 块。模型请求发送前还会检查该块是否与客户端绑定的本人配置一致。

私人记忆保持关闭，`enable_memory=False`。客观的本 pair 历史仍由环境提供。与基准相比，偏好注入本身不增加额外模型请求，但会增加输入文本和本地日志。Router 沿用当前项目的行为：`llm` 模式会调用 Router 模型，它只能转发已明确提交的经济变量；`scripted` 模式不调用外部模型。

偏好配置只自动注入本人的 participant 输入，其他参与者无法读取该配置。Router 沿用原有的请求转发路径，会收到参与者提交的原文理由；如果参与者在理由中主动提到自己的偏好，Router 可能看到这些文字，但不能据此修改金额。程序不会把全员偏好配置交给 Router。

新提示词明确要求参与者希望提高自己的最终收益，同时按个人偏好权衡收益的不确定性和自己收入高于搭档的程度。参与者自行决定是否愿意、以及愿意放弃多少自身收益来降低风险或缩小优势收入差距。支付方式和环境规则没有改变。原版与新版的差异同时包括目标措辞和个人配置，不能仅凭两个版本的结果差异识别某一个偏好的作用。

## 先跑一轮

以下命令均在项目根目录执行，直接使用项目 Python，无需定义 `run_llm` 等 shell 函数。

```bash
.venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --treatment baseline \
  --backend scripted \
  --participants 4 \
  --agent-concurrency 4 \
  --completed-period-limit 1 \
  --seed 2026 \
  --preference-seed 2026 \
  --request-limit 100 \
  --max-output-tokens 512 \
  --router-max-steps 6 \
  --agent-max-turns 4 \
  --timeout-seconds 180
```

默认配置会尽量均衡分配两项偏好的九种组合，再随机分给 Agent ID。人数不是九的倍数时，各组合人数相差至多一人。`--preference-seed` 只影响偏好配置，使用独立随机数生成器，不消耗收入、配对、终止或保险分配的随机数。相同人数和偏好种子可在不同 treatment 中保持相同的个人配置。

这是一种合成实验设计。20 个 Agent 并不代表从真实人口分布中抽取的 20 个人。

## 指定每个人的偏好

[`profiles.example.json`](profiles.example.json) 提供了 4 人示例。运行时加入：

```text
--profiles-file experiments/fixed_preferences_v1/profiles.example.json
```

例如：

```bash
.venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --treatment baseline --backend scripted \
  --participants 4 --agent-concurrency 4 \
  --completed-period-limit 1 --request-limit 100 \
  --profiles-file experiments/fixed_preferences_v1/profiles.example.json
```

文件必须包含 `schema_version: 1` 和 `profiles`，后者完整列出 `"1"` 到 `"participants"` 的所有 Agent。每人只能有上述两个字段；等级、未知字段、缺失 Agent 和重复 JSON 键都会被校验。示例文件只能配合 4 人运行；20 人运行需要包含 20 人的文件。

文件配置优先于自动生成，使用文件时 `--preference-seed` 不再参与分配。输入文件不接受自由提示词、金额或模型填写的身份。可信身份仍由框架传递。

每次运行导出的 `participant_profiles.json` 也符合这个输入格式。可以把它直接作为下一次运行的 `--profiles-file`，在不同 treatment 中复用完全相同的个人配置；配置的来源和生成方法另存于 `manifest.json`。

## 10 组、20 人的完整离线实验

下面三条命令分别运行到自然终止；20 表示同时执行的 Agent 上限，每轮仍需等参与者完成后统一结算。输出目录自动生成，互不覆盖。

Baseline：

```bash
.venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --treatment baseline --backend scripted \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 \
  --request-limit 5000 --max-output-tokens 512 \
  --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 86400
```

25V：

```bash
.venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --treatment 25v --backend scripted \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 \
  --request-limit 5000 --max-output-tokens 512 \
  --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 86400
```

25F：

```bash
.venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --treatment 25f --backend scripted \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 \
  --forced-assignment-probabilities 0.5 0.25 0.25 \
  --request-limit 5000 --max-output-tokens 512 \
  --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 86400
```

25F 的三个概率依次表示双方投保、仅 pair 第一位投保、仅第二位投保。上面的 `0.5 / 0.25 / 0.25` 是示例研究设定，不是已经从论文确定的分配概率。程序没有自动默认值；运行者需要明确提供。

`scripted` 沿用基准中的确定性 participant policy，故意不根据偏好等级改变金额。它用于验证真实框架调用、配置传入、身份与金额校验、结算和日志。不同 Agent 的 scripted 金额不是偏好产生了经济效果的证据。

## 使用外部 LLM

外部模型仍读取项目已有的 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL` 配置。偏好版本不修改 `.env`，也不自动更换模型。建议先把下一条命令中的 `--full-session` 换成 `--completed-period-limit 1`，检查结果后再运行完整 session。

这条命令临时清除可能与项目配置冲突的 AgentSociety 框架变量，只作用于本次进程：

```bash
env -u AGENTSOCIETY_LLM_API_KEY \
  -u AGENTSOCIETY_LLM_API_BASE \
  -u AGENTSOCIETY_LLM_MODEL \
  -u AGENTSOCIETY_CODER_LLM_API_KEY \
  -u AGENTSOCIETY_CODER_LLM_API_BASE \
  -u AGENTSOCIETY_CODER_LLM_MODEL \
  -u AGENTSOCIETY_LLM_THINKING \
  -u AGENTSOCIETY_CODER_LLM_THINKING \
  -u AGENTSOCIETY_LLM_EXTRA_BODY \
  -u AGENTSOCIETY_CODER_LLM_EXTRA_BODY \
  .venv/bin/python -m experiments.fixed_preferences_v1.run_smoke \
  --treatment baseline --backend llm \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 \
  --request-limit 20000 --max-output-tokens 2048 \
  --router-max-steps 4 --agent-max-turns 4 \
  --timeout-seconds 86400 --request-timeout 120
```

运行 25V 时将 `--treatment baseline` 改为 `--treatment 25v`；运行 25F 时改为 `--treatment 25f`，并加入明确的 `--forced-assignment-probabilities`。并发上限是否适合实际服务，需要结合请求延迟和服务端限流判断。

## 运行记录

默认目录为 `experiments/fixed_preferences_v1/runs/<treatment>/<运行编号>/`。也可用 `--run-dir` 指定尚不存在的目录。这个版本的 `runs/` 由本目录的 `.gitignore` 忽略。

| 文件 | 内容 |
|---|---|
| `summary.md`、`report.html` | 本次运行概况及沿用项目报告模块生成的图表 |
| `participant_profiles.json` | 全员配置，可直接复用为输入文件；供研究者审阅，不作为全员上下文传给参与者 |
| `participant_contexts.jsonl` | 实际交给 participant client 的模型消息、本人偏好和配置指纹，可按 Agent、轮次和 decision ID 核对 |
| `agent_actions.jsonl` | 参与者原始选择、Router 动作、环境 receipt，以及关联的个人偏好指纹 |
| `agents/agent_XXXX/config.json` | 该 Agent 的固定配置；每轮恢复后继续使用 |
| `manifest.json` | 实验及呈现版本、运行参数、配置来源、配置文件和源代码指纹、请求次数及结果 |
| `baseline_reference.json` | 新版本创建时原项目相关文件的指纹，便于核对基准 |
| `period_records.jsonl` | 环境已经结算的逐人逐轮数据 |

25V、25F 继续分别保存保险选择或保险分配记录。完整 session 继续保存支付轮次和支付记录；这些记录不是外部付款操作。

输入日志会随人数和轮数增长，因为每次输入都可能包含本 pair 的历史。日志仅供研究者读取，不进入其他参与者的 observation。

## 文献与后续验证

风险偏好的选择任务参考 Holt & Laury（2002），[Risk Aversion and Incentive Effects](https://doi.org/10.1257/000282802762024700)。优势不平等厌恶的概念参考 Fehr & Schmidt（1999），[A Theory of Fairness, Competition, and Cooperation](https://doi.org/10.1162/003355399556151)。本版本没有实现两篇论文的参数估计，也没有声称完整实现 Fehr–Schmidt 效用函数。

当前版本把文献中的概念转成可重复配置的模型输入。它尚未通过真实 LLM 选择任务验证：`high` 是否稳定表现得比 `low` 更风险厌恶，或者更愿意付出自身收益来减少优势收入差距。正式解释实验结果前，需要完成这些独立检查。

建议先保持模型、采样参数、其他提示词和收入安排一致，使用显式配置文件，每次只改变一个偏好等级。风险偏好用已知概率的彩票选择检查；优势不平等厌恶用没有历史和后续互动的分配任务检查。确认行为方向后，再把组合配置用于风险分担实验，避免依据正式实验的预期结果反复调整提示词。

工程回归测试可以单独运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_fixed_preference_profiles.py \
  tests/test_fixed_preferences_integration.py \
  tests/test_observation_presentation.py \
  tests/test_observation_presentation_integration.py
```

这些测试检查配置和调用边界。模型的经济行为是否符合设定，以及是否更接近人类实验，是另外的验证问题。

2026-10-25 的完整测试结果、三类 20 人离线运行和产物路径见 [验证记录](VALIDATION.md)。
