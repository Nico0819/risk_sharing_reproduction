# 固定偏好与私人记忆

这个版本保留风险厌恶、优势不平等厌恶两项固定偏好，让每个参与者另外保存一份自己的主观记录。记录会随经历更新，供之后的决定参考。

`experiments.fixed_preferences_v1.run_smoke` 是没有私人记忆的对照入口。本入口有独立的 Agent 类、runner 和运行目录，复用相同的偏好配置、信息呈现、环境规则、身份验证、Router 和结果报告。当前代码已经统一开启参与者原生推理并保存接口返回的轨迹；历史运行文件保留原样。

最初版本的工程验证见 [2026-10-26 验证记录](../../artifacts/fixed_preferences_memory_v1_20261026/README.md)。该记录不代表后来修改的验证状态；开启原生推理后的行为和成本需要重新实验。

自由笔记、终局补记和匹配对照的验证见 [2026-11-09 验证记录](../../artifacts/free_memory_20261109/README.md)。

## 记录什么

每个人的 `agents/agent_XXXX/private_memory.json` 使用 schema 2。模型只输出两个字段：

```json
{"note": "自己选择保留的短笔记", "evidence_periods": [1, 2]}
```

`note` 最多 2400 个字符，可以为空、保持原样，也可以修正或撤回旧判断。不要求预测搭档的金额、填写信任分数或制定某种策略。2400 字符是当前工程上限，不代表经验证的人类记忆容量。`evidence_periods` 最多引用六个已结算轮次，程序附上对应事实；决策仍然可以看到本 pair 的完整合法历史，并非只看到这六轮。

身份、搭档、运行范围、偏好指纹和修订号由程序维护。最初笔记为空。旧 schema 1 的运行记录保留原样，新程序不会自动转换或继续载入旧状态；可读报告工具兼容两种历史格式。

事实与判断分开处理：模型只能引用已经看见的本 pair 历史，程序从合法 observation 中附上对应记录；当前收入、保险安排和结算历史仍以环境为准。程序不把自由文字中的判断当成事实，也无法保证模型的每项判断都正确。

## 每轮怎么运行

```text
本人当前 observation ＋ 本人上次的私人记录
    → 检查新结算反馈或新披露信息；有新反馈则由本人更新，没有则沿用笔记
    → 校验并保存到本人的 workspace
    → 同一个 participant model 根据 observation 和新记录选择行动
    → Router 转发明确的选择
    → 环境校验；所有人完成后统一结算
```

第一轮没有历史反馈时，直接使用空笔记。低收入参与者收到新反馈也会更新，随后等待。25V 的保险报价、双方完成投保后公开的状态，以及第 31 轮公开的撤销信息按实际披露时机进入更新。单纯改变 decision_id、任务或当轮收入不触发记忆调用；重试同一行动也不会重复更新。

成功完成指定轮数或自然结束后，通过真实 PersonAgent → ask_env → 可信 Adapter → 环境再读取一次仅含本人身份与本 pair 已结算历史的反馈，更新末轮笔记后本地结束。不生成经济决定，不推进环境，也不展示下一轮收入或保险信息。终局更新标记为 `memory_update_kind=closing`，不能用来解释之前的决定。

每次新增合法反馈通常增加一次没有工具权限的 participant 记忆调用。若输出未通过校验，同一个人的模型会收到具体错误反馈，最多再修正两次；包括初次请求在内，总共最多三次。每次尝试都计入原有的请求数、token 和时间上限。它和经济决定使用同一个人的客户端；Router 不参与生成或修正记忆。脚本后端只验证读取、写入和隔离，金额策略仍沿用基准，不用来证明记忆产生了经济效果。

记忆调用会额外加载 [`memory_update_phase.md`](../../custom/agents/fixed_preferences_memory_v1/memory_update_phase.md) 作为系统级阶段说明，明确当前只更新笔记，观察中的行动说明用于后续决策。该说明不会进入正常经济决策调用。项目的 `finish` 工具也明确区分结束回合所需的简短事件与另行保存的私人笔记，避免模型反复判断应当转账还是更新记忆。模型、原生推理开关、输出预算和合法历史保持原设置。

私人状态通过独立文件在框架每轮恢复 Agent 时继续使用。`private_memory_enabled=True` 表示本版本的私人记忆；`enable_memory=False` 和 `framework_memory_enabled=False` 表示 AgentSociety2 通用记忆仍关闭。本版本不使用额外 embedding 或记忆整理服务。

文件绑定本人、搭档、运行范围和偏好指纹。文件缺失、损坏或属于其他人时会停止，不会静默初始化一份替代。模型输出的字段、类型、引用轮次不合法时，旧记录保持不变，先由该参与者模型在有限次数内修正。只有完整校验通过的输出才会保存一次并进入经济决策；三次尝试仍不合法时停止，不使用无效记录，也不跳过记忆更新继续行动。程序不会替模型删除多余字段、挑选历史依据、补记忆或补金额。预算耗尽、网络异常和取消不属于格式修正范围，不会在这里自动重试。

记忆输出接受纯 JSON，也兼容完整、单一的 Markdown JSON 代码块（`json` 标签或无标签）。程序只去掉最外层代码块标记，仍检查重复字段、类型、笔记长度和历史引用；不会从解释文字中抽取 JSON，也不会自行修补不完整的内容。需要修正时由同一个参与者模型重新输出。日志中的 `raw_output` 保留模型原文，包括代码块标记，已保存的状态文件仍为标准 JSON。

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

当前入口也会覆盖本进程残留的框架 thinking 设置：参与者开启，Router 关闭。下面的兼容命令显式清除旧别名后运行 Baseline，不修改 `.env` 或父终端变量：

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
  --max-output-tokens 8192 --router-max-steps 4 --agent-max-turns 4 \
  --timeout-seconds 86400 --request-timeout 120
```

运行25V时改为 `--treatment 25v`。运行25F时改为 `--treatment 25f`，并加入 `--forced-assignment-probabilities 0.5 0.25 0.25`。没有额外的信息格式参数。

记忆更新增加了模型请求和输入文本，运行时间与费用可能高于固定偏好基准。默认没有另外开启 `--participant-deliberation`；如果显式启用，它在记忆更新后进行，还会增加请求。

参与者在记忆更新、经济决策和可选评估中都使用原生推理。单次输出预算默认 8192 token，这些阶段统一使用 `--max-output-tokens`，不再把记忆或评估输出另行压到 1024/512。预算需要容纳推理与最终输出，较小的值可能导致没有完整行动。实际服务必须支持当前请求格式；这次修改没有自动更换模型，也没有用真实 API 验证网关兼容性。

## 查看推理轨迹

每次运行都会创建 `participant_reasoning.jsonl`。每一行对应一次 participant client 调用的最终返回或异常，按本人身份记录，不记录 Router 的推理，也不把这些研究日志交给其他 Agent。

```bash
.venv/bin/python -m experiments.view_reasoning <运行目录> \
  --agent-id 1 --period 16 --phase decision
```

去掉 `--period` 可查看该 Agent 的多轮记录；默认最多显示 10 条，加上 `--limit 0` 显示全部匹配记录，每条正文不截短。`--phase memory_update` 查看更新私人记忆时的记录，`--phase deliberation` 查看显式开启的额外评估。查看命令只读本地文件，不请求模型。旧运行没有这个文件时，不能事后恢复原始推理。

实验结束后会自动生成 `reasoning_text/all_agents.txt` 和每个 Agent 的独立文本，`summary.md` 中有汇总文件的链接。原版、固定偏好版和私人记忆版的 Baseline、25V、25F 入口都适用。导出会等待当前运行的调用结束，包含最后一次记忆更新；Agent 报错或程序处理的超时退出也会导出已保存记录。尚未产生调用记录时跳过；导出失败会写入 `run.log`，不改变实验成功与否。

旧运行、导出失败后重试，或进程被强制杀死而未执行收尾时，可以手动导出：

```bash
.venv/bin/python -m experiments.export_reasoning <运行目录>
```

也可以传入 `participant_reasoning.jsonl` 的文件路径。程序在运行目录下生成 `reasoning_text/all_agents.txt`，以及 `reasoning_text/agent_0001.txt` 等个人文件，按 Agent、轮次和请求顺序整理全部记录。只提取推理正文，保留原文、换行和空值／截断状态，不包含最终回答、工具参数或密文签名。不会调用模型，也不修改原始日志。重新导出会更新同名文本；`--output-dir <目录>` 可另选保存位置。

主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `agent_id`、`pair_id`、`period`、`task`、`phase` | 这次调用属于谁、哪轮、什么任务 |
| `request_index`、`decision_id` | 与输入日志、提交选择及环境回执关联的凭据 |
| `reasoning_content` 等接口实际返回的字段 | 接口暴露的推理，保持原文长度；敏感凭据会脱敏 |
| `content`、`tool_calls` | 模型的最终文字及工具调用参数 |
| `reasoning_status` | `available`、`empty`、`missing` 或 `request_failed` |
| `readable_reasoning_present` | 返回的字段中是否包含可读推理文本；只有签名等不透明信息不算 |
| `finish_reason`、`response_truncated` | 响应为何结束，是否触及长度上限 |
| `redaction_applied` | 日志脱敏是否改动了返回文本 |
| `response_source` | `provider` 是真实接口；`scripted` 是离线模拟 |

`manifest.json` 汇总推理返回状态。`participant_reasoning_capture_complete` 只有真实接口的所有已记录调用都正常结束、有可读推理且没有脱敏改动时才为真；缺失、截断、异常或无法判断结束原因时不会声称完整。经济实验成功和推理记录完整是两个独立指标，应同时检查。

推理字段会在同一 Agent 的后续工具对话中原样回传，避免框架重建消息时丢失。字段缺失时不伪造空白推理来通过后续校验。日志不保存 API 密钥、请求头或完整网络响应对象。

真实参与者请求使用 `tool_choice=auto`，只提供当前任务允许的工具；金额、凭据、身份和是否可以转账仍由程序检查。这样适配推理模式不接受强制工具调用的接口限制，Router 关闭推理后仍可强制转发指定工具。接口依据见[工具选择参数](https://api-docs.deepseek.com/api/create-chat-completion/)和[推理字段回传规则](https://api-docs.deepseek.com/guides/thinking_mode/)。

这里的“轨迹”仅指供应商实际公开的推理与输出，不等于模型全部内部计算，也不保证文字能忠实解释决策。框架内部可能重试网络请求，本文件只记录一次框架客户端调用最终拿到的响应或异常，不覆盖每次底层重试。开启推理可能改变行为、输出长度、耗时和费用，与旧实验比较时需要记录这一条件变化。

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
| `memory_text/all_agents.txt`、`memory_text/agent_XXXX.txt` | 按人阅读每次更新前后、依据和使用该记忆的决定；终局补记单独标记 |
| `private_memory_updates.jsonl` | 成功更新前后的状态、模型原始输出、依据、调用序号和成功时的 `attempt` |
| `participant_contexts.jsonl` | 实际 participant 输入；区分 `memory_update`、`decision` 和可选的 `deliberation`，记忆输入记录 `memory_update_attempt` |
| `participant_reasoning.jsonl` | 对应请求的接口推理、最终输出、工具参数、完成与截断状态 |
| `agent_actions.jsonl` | 原始选择、环境回执，以及本次决定读取的记忆指纹、修订号和调用序号 |
| `manifest.json` | 参数、私人记忆实现、更新次数、代码和提示词指纹 |
| `baseline_reference.json` | 此版本最初建立时的历史文件指纹；当前代码以 manifest 中本次运行的源文件指纹为准 |

记忆输出被拒绝时，`run.log` 中的 `private_memory_update_rejected` 事件记录每次原始输出、拒绝原因、此前修订号、`attempt` 和 `max_attempts`，并沿用密钥脱敏。拒绝内容不会出现在成功更新日志中，也不会覆盖此前有效的私人记录。修正过程中保存的记忆修订号不变；成功后只增加一次，后续经济调用只看到成功保存的版本。

`manifest.json` 和 `summary.md` 分别列出记忆请求数 `private_memory_update_requests`、成功保存次数 `private_memory_update_count`、被校验拒绝次数 `private_memory_update_rejections`，以及每次更新的尝试上限 `private_memory_update_max_attempts=3`。因此出现修正时，请求数会大于成功保存次数；若发生网络错误，还可能存在没有进入内容校验的请求。

研究日志不进入其他参与者的输入。记忆文件中的运行范围用于程序校验，不交给模型作为实验信息。

## 和基准比较

基准命令使用 `experiments.fixed_preferences_v1.run_smoke`，本版本使用 `experiments.fixed_preferences_memory_v1.run_smoke`。保持人数、配对和收入种子、偏好配置、模型设置相同；也可以让两版通过 `--profiles-file` 读取同一份导出的配置。

这项改动同时加入了跨轮的主观状态和决策前的记忆更新调用。因此，即使行为发生变化，也不能仅凭两次运行把差异全部归因于“记住了过去”，还需要考虑额外调用和提示内容的影响。是否更接近人类或原论文结果，需要独立重复运行和分析。


## 查看记忆与匹配对照

新运行会自动生成 `memory_text/all_agents.txt` 与每人的 `agent_XXXX.txt`。它展示的是已保存笔记的演变和关联决定；推理全文仍在单独的 `reasoning_text` 导出中。也可对旧运行只读导出：

```bash
.venv/bin/python -m experiments.export_memory <运行目录>
```

默认模式为 `--memory-condition persistent`，更新时读取本人旧笔记。使用 `--memory-condition fresh` 则在相同触发时机、提示词和请求预算下生成当轮笔记，更新调用看不到旧笔记和它引用的附加记录，但仍能读取完整合法历史；当轮经济决定使用本次新笔记。两种模式都保留磁盘版本和研究日志，`fresh` 不把它们反馈给后续更新调用。身份和审计元数据不被清空。

比较时仅切换这个参数，保持 treatment、种子、偏好、模型、并发和输出预算一致。固定偏好无私人记忆入口仍可作为直接决策条件。scripted policy 不读取笔记来选择金额，因此只验证调用边界；是否改变真实 LLM 行为需要另做实验。

`manifest.success` 继续表示经济实验是否完成；`closing_memory_complete` 表示终局补记是否全部完成；`memory_experiment_complete` 要求两者同时成立。终局模型失败不会抹去已结算经济结果，错误单列在 `closing_memory_failures`。命令行仅在完整记忆实验完成时返回退出码 0，终局调用同样受到请求和运行时间预算限制。

日志可以验证笔记被输入，以及具体决定读取了哪个修订号，不能单独证明笔记导致了该决定。当前没有强制收集每个行动的简短理由，也不把完整推理日志回灌；模型不能把后来推测的理由伪称为当时记忆。
