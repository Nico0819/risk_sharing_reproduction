# 固定偏好与私人记忆

这个版本保留风险厌恶、优势不平等厌恶两项固定偏好，让每个参与者另外保存一份自己的主观记录。记录会随经历更新，供之后的决定参考。

现有 `experiments.fixed_preferences_v1.run_smoke` 继续作为没有私人记忆的基准。新版本有独立的 Agent 类、runner 和运行目录，复用相同的偏好配置、信息呈现、环境规则、身份验证、Router 和结果报告。原有代码与历史运行没有被覆盖。

已通过 482 项测试和三类各 20 人的完整离线实验，详见[验证记录](../../artifacts/fixed_preferences_memory_v1_20261026/README.md)。尚未调用外部 LLM 验证行为效果。

## 记录什么

每个人的 `agents/agent_XXXX/private_memory.json` 包含：

- 对搭档的暂时判断；
- 预计搭档下次有转账机会时可能给多少，以及自己对此有多大把握；
- 自己认为搭档可能期待自己给多少，以及把握程度；
- 尚不清楚、需要修正的地方；
- 最多六个已结算轮次的引用及环境提供的原始事实。

最初没有预设信任、合作意愿或金额预期。模型可以回答“不知道”，也可以修正以前的判断。预测金额不是当轮选择，环境不会自动拿它提交转账。

事实与判断分开处理：模型只能引用已经看见的本 pair 历史，程序从合法 observation 中附上对应记录；当前收入、保险安排和结算历史仍以环境为准。程序不把自由文字中的判断当成事实，也无法保证模型的每项判断都正确。

## 每轮怎么运行

```text
本人当前 observation ＋ 本人上次的私人记录
    → 本人的 participant model 更新记录
    → 校验并保存到本人的 workspace
    → 同一个 participant model 根据 observation 和新记录选择行动
    → Router 转发明确的选择
    → 环境校验；所有人完成后统一结算
```

低收入参与者也会更新自己的记录，随后等待，不提交转账。25V 在同一轮的投保任务和之后的转账任务是两份不同 observation，因此会分别更新。因格式错误重试同一个行动时，已完成的记忆更新不会重复执行。

每份新 observation 通常增加一次没有工具权限的 participant 记忆调用。若输出未通过校验，同一个人的模型会收到具体错误反馈，最多再修正两次；包括初次请求在内，总共最多三次。每次尝试都计入原有的请求数、token 和时间上限。它和经济决定使用同一个人的客户端；Router 不参与生成或修正记忆。脚本后端只验证读取、写入和隔离，金额策略仍沿用基准，不用来证明记忆产生了经济效果。

私人状态通过独立文件在框架每轮恢复 Agent 时继续使用。`private_memory_enabled=True` 表示本版本的私人记忆；`enable_memory=False` 和 `framework_memory_enabled=False` 表示 AgentSociety2 通用记忆仍关闭。本版本不使用额外 embedding 或记忆整理服务。

文件绑定本人、搭档、运行范围和偏好指纹。文件缺失、损坏或属于其他人时会停止，不会静默初始化一份替代。模型输出的字段、类型、引用轮次不合法时，旧记录保持不变，先由该参与者模型在有限次数内修正。只有完整校验通过的输出才会保存一次并进入经济决策；三次尝试仍不合法时停止，不使用无效记录，也不跳过记忆更新继续行动。程序不会替模型删除多余字段、挑选历史依据、补记忆或补金额。预算耗尽、网络异常和取消不属于格式修正范围，不会在这里自动重试。

记忆输出接受纯 JSON，也兼容完整、单一的 Markdown JSON 代码块（`json` 标签或无标签）。程序只去掉最外层代码块标记，仍检查重复字段、类型、金额预测范围和历史引用；不会从解释文字中抽取 JSON，也不会自行修补不完整的内容。需要修正时由同一个参与者模型重新输出。日志中的 `raw_output` 保留模型原文，包括代码块标记，已保存的状态文件仍为标准 JSON。

其他 Agent 无法通过工具访问这份记录。程序不会把它自动加入 Router 请求；参与者如果在自己的决定理由中主动提到某项判断，Router 仍可能看到那段理由。记忆不是模型完整的内部思维记录。

## 先跑两轮检查

在项目根目录运行：

```bash
.venv/bin/python -m experiments.fixed_preferences_memory_v1.run_smoke \
  --backend scripted --treatment baseline \
  --participants 4 --agent-concurrency 4 \
  --completed-period-limit 2 --seed 2026 --preference-seed 2026 \
  --request-limit 100 --max-output-tokens 512 \
  --router-max-steps 6 --agent-max-turns 4 --timeout-seconds 180
```

## 三种完整离线实验

下面各运行20人、10组，并发上限20，到自然终止。25F 的概率沿用显式测试设置，不表示已经确认论文原始分配概率。

```bash
.venv/bin/python -m experiments.fixed_preferences_memory_v1.run_smoke \
  --backend scripted --treatment baseline \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 --request-limit 5000 \
  --max-output-tokens 512 --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 86400
```

```bash
.venv/bin/python -m experiments.fixed_preferences_memory_v1.run_smoke \
  --backend scripted --treatment 25v \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 --request-limit 5000 \
  --max-output-tokens 512 --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 86400
```

```bash
.venv/bin/python -m experiments.fixed_preferences_memory_v1.run_smoke \
  --backend scripted --treatment 25f \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 \
  --forced-assignment-probabilities 0.5 0.25 0.25 --request-limit 5000 \
  --max-output-tokens 512 --router-max-steps 6 --agent-max-turns 4 \
  --timeout-seconds 86400
```

## 使用外部 LLM

继续读取已有的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 配置，不需要修改 `.env`。同名 `LLM_*` 终端变量优先于 `.env`；启动时会用解析后的项目配置覆盖本进程中残留的 `AGENTSOCIETY_*` 密钥、地址和模型别名，因此这些旧别名不会再引起配置冲突。缺少项目配置时仍会报错，不会借用旧别名。

下面的兼容命令还会清除可能残留的框架 thinking 设置，然后运行 Baseline；它不会修改 `.env` 或父终端的变量：

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
  .venv/bin/python -m experiments.fixed_preferences_memory_v1.run_smoke \
  --backend llm --treatment baseline \
  --participants 20 --agent-concurrency 20 --full-session \
  --seed 2026 --preference-seed 2026 --request-limit 20000 \
  --max-output-tokens 2048 --router-max-steps 4 --agent-max-turns 4 \
  --timeout-seconds 86400 --request-timeout 120
```

运行25V时改为 `--treatment 25v`。运行25F时改为 `--treatment 25f`，并加入 `--forced-assignment-probabilities 0.5 0.25 0.25`。没有额外的信息格式参数。

记忆更新增加了模型请求和输入文本，运行时间与费用可能高于固定偏好基准。默认没有另外开启 `--participant-deliberation`；如果显式启用，它在记忆更新后进行，还会增加请求。

## 文件与核对

默认产物目录：

```text
experiments/fixed_preferences_memory_v1/runs/<treatment>/<运行编号>/
```

| 文件 | 用途 |
| --- | --- |
| `report.html`、`summary.md` | 结果图表、运行状态、记忆调用数量 |
| `participant_profiles.json` | 与基准相同格式的个人偏好，可复用为 `--profiles-file` 输入 |
| `agents/agent_XXXX/private_memory.json` | 对应参与者当前的持久记录 |
| `private_memory_updates.jsonl` | 成功更新前后的状态、模型原始输出、依据、调用序号和成功时的 `attempt` |
| `participant_contexts.jsonl` | 实际 participant 输入；区分 `memory_update`、`decision` 和可选的 `deliberation`，记忆输入记录 `memory_update_attempt` |
| `agent_actions.jsonl` | 原始选择、环境回执，以及本次决定读取的记忆指纹、修订号和调用序号 |
| `manifest.json` | 参数、私人记忆实现、更新次数、代码和提示词指纹 |
| `baseline_reference.json` | 新版本建立时的基准文件指纹 |

记忆输出被拒绝时，`run.log` 中的 `private_memory_update_rejected` 事件记录每次原始输出、拒绝原因、此前修订号、`attempt` 和 `max_attempts`，并沿用密钥脱敏。拒绝内容不会出现在成功更新日志中，也不会覆盖此前有效的私人记录。修正过程中保存的记忆修订号不变；成功后只增加一次，后续经济调用只看到成功保存的版本。

`manifest.json` 和 `summary.md` 分别列出记忆请求数 `private_memory_update_requests`、成功保存次数 `private_memory_update_count`、被校验拒绝次数 `private_memory_update_rejections`，以及每次更新的尝试上限 `private_memory_update_max_attempts=3`。因此出现修正时，请求数会大于成功保存次数；若发生网络错误，还可能存在没有进入内容校验的请求。

研究日志不进入其他参与者的输入。记忆文件中的运行范围用于程序校验，不交给模型作为实验信息。

## 和基准比较

基准命令使用 `experiments.fixed_preferences_v1.run_smoke`，本版本使用 `experiments.fixed_preferences_memory_v1.run_smoke`。保持人数、配对和收入种子、偏好配置、模型设置相同；也可以让两版通过 `--profiles-file` 读取同一份导出的配置。

这项改动同时加入了跨轮的主观状态和决策前的记忆更新调用。因此，即使行为发生变化，也不能仅凭两次运行把差异全部归因于“记住了过去”，还需要考虑额外调用和提示内容的影响。是否更接近人类或原论文结果，需要独立重复运行和分析。
