# 九种人格的配对实验

这里使用现有的“固定偏好＋私人记忆”参与者。风险厌恶和优势不平等厌恶各有低、中、高三个等级，合成九种偏好配置。这里把这些配置简称为“人格”，不表示已经验证了完整的人类人格模型。

| 编号 | 风险厌恶 | 优势不平等厌恶 |
| --- | --- | --- |
| P1 | 低 | 低 |
| P2 | 低 | 中 |
| P3 | 低 | 高 |
| P4 | 中 | 低 |
| P5 | 中 | 中 |
| P6 | 中 | 高 |
| P7 | 高 | 低 |
| P8 | 高 | 中 |
| P9 | 高 | 高 |

配对保留顺序：`P2__P8` 和 `P8__P2` 是两种组合，也包含 `P2__P2` 这样的同类配对。默认每类实验有 81 × 10 = 810 场独立两人实验；Baseline、25V、25F 合计 2,430 场，创建 4,860 个参与者实例。

每次重复都是一个新 session，只有一对参与者，记忆从空开始。每场使用不同的环境种子；25F 另有独立的保险分配种子。三个 treatment 也使用不同的随机冲击，不是匹配收入路径的设计。种子可以复现环境随机过程，不能保证外部 LLM 的输出完全一致。

## 配对方向与隔离

“第一位／第二位”指环境中 pair 的实际顺序，不固定等于 Agent 1／Agent 2。计划生成时按环境的 shuffle 规则分配人格，因此 25F 的 first_only/second_only 与这里的方向一致。每场的 `job.json`、`pairing_metadata.json` 和 CSV 都保存实际 Agent ID。

人格、组合与重复编号用于研究侧标注。参与者仍只能看到自己的偏好与合法 pair 信息，不会收到搭档人格或批次计划。经济规则、参与者提示词、私人记忆更新和 Router 的职责沿用现有版本。

## 先生成计划，再开始运行

在项目根目录执行。`plan` 只生成任务清单和每场的偏好文件，不读取模型凭据、不发 API 请求。目录必须不存在。

下面 25F 的 `0.5 0.25 0.25` 是显式示例分布，分别对应双方获保、仅第一位获保、仅第二位获保；不是已经从论文确认的概率。代码没有为 25F 猜一个默认值，正式实验应明确采用的分布。

```bash
.venv/bin/python -m experiments.preference_grid.run_batch plan \
  --batch-dir experiments/preference_grid/runs/grid_01 \
  --backend llm \
  --treatments baseline 25v 25f \
  --repetitions 10 \
  --seed 2026 \
  --forced-assignment-probabilities 0.5 0.25 0.25 \
  --full-session \
  --request-limit 1000 \
  --max-output-tokens 8192 \
  --router-max-steps 6 \
  --agent-max-turns 4 \
  --timeout-seconds 86400 \
  --request-timeout 180
```

检查 `plan.json` 后，用下面的命令启动：

```bash
.venv/bin/python -m experiments.preference_grid.run_batch run \
  --batch-dir experiments/preference_grid/runs/grid_01 \
  --concurrency 10
```

这里并发的是 10 场独立 session，每场两个 Agent 内部也并发，最多同时处理 20 个 Agent 的行动。不同 session 使用独立子进程，避免框架的全局工作区和配置互相干扰。`--concurrency` 可调整；并发上限改变调度，不减少推理预算或改变记忆机制。

`--timeout-seconds` 和 `--request-limit` 都是每场两人实验的上限，整个批次没有额外总时限。任务的模型调用继续使用项目现有 LLM 配置；这里不会更换模型。`execution.json` 记录不含密钥的配置与代码指纹，恢复时条件发生变化会拒绝混入同一批数据。

可以先只运行少量任务：

```bash
.venv/bin/python -m experiments.preference_grid.run_batch run \
  --batch-dir experiments/preference_grid/runs/grid_01 \
  --treatments baseline --combinations P2__P8 \
  --limit-jobs 2 --concurrency 2
```

再次执行 `run` 会跳过已完成和已有失败记录的任务，继续尚未运行的部分。需要重跑失败或中断的任务时显式加 `--retry-failed`；仍使用原种子，但新建一次 attempt、重新创建 Agent 和记忆，保留先前文件。它是同一重复的重试，不是新增一个独立重复。每个任务最多在一次启动中尝试一次，不会无限重试到成功。

## 查看数据

终端实时进度可以用独立的显示入口。它只读取状态和日志，不改批次的代码指纹或实验设置。对于已经启动的实验，在另一个终端执行：

```bash
.venv/bin/python -m experiments.preference_grid.live watch \
  --batch-dir experiments/preference_grid/runs/grid_01 \
  --interval 10
```

显示每场当前轮次、已结算轮次、请求数、最近调用阶段、日志更新时间，以及任务数量。`watch` 不启动任务，不调用模型；按 Ctrl+C 只退出查看。加 `--once` 可只显示一次，`--treatments baseline --combinations P2__P8` 可缩小查看范围。

以后也可以直接用带进度的入口启动。它调用相同的批量运行器，支持原来的筛选、并发、`--limit-jobs` 和 `--retry-failed` 参数：

```bash
.venv/bin/python -m experiments.preference_grid.live run \
  --batch-dir experiments/preference_grid/runs/grid_01 \
  --concurrency 10 --interval 10
```

这里 `run` 中按 Ctrl+C 会中断该次批量执行。终端标记“进程完成”只表示子进程已结束，正式可用的重复数以报告的完整性校验为准；请求仍在等待时也会刷新，不把暂时没有新日志判为卡死。

每次批量运行结束后自动生成以下文件，也可以不调用模型地重新汇总：

```bash
.venv/bin/python -m experiments.preference_grid.run_batch report \
  --batch-dir experiments/preference_grid/runs/grid_01
```

| 文件 | 内容 |
| --- | --- |
| `report.html` | 可离线查看总体分布、轮次曲线、配对热图、逐次重复、偏好分组、阶段变化、保险、互惠条件和场次明细 |
| `pair_sessions.csv` | 每个计划任务一行；双方人格、重复编号、种子、完成状态、投保或保险分配及原始记录路径 |
| `period_records.csv` | 每个 Agent 每轮一行；附本人和搭档人格、配对组合、重复编号及原始结算值 |
| `combination_summary.csv` | 每种 treatment × 配对 × 阶段的重复间均值、标准差与有效重复数 |
| `cases/<任务编号>/job.json` | 该场的固定设计与人格映射 |
| `pair_sessions.csv` 的 `run_dir` 列所指目录 | 该场采用的报告、结算、输入日志、私人记忆和推理文本 |

### 扩展分析

运行同一条 `run_batch report --batch-dir ...` 会根据已有结算记录重新计算扩展分析。正常批量运行结束时也会自动生成。页面默认选择 1–45 轮，便于比较相同长度的历史；可以切换三个固定阶段、全部轮次及随机延长阶段。点击人格热图中的格子，可看到该组合在三个实验中每次重复的散点、标准差、四分位数和均值区间。

新增文件：

| 文件 | 用途 |
| --- | --- |
| `analysis_summary.md` | 可直接阅读的完成情况、阶段结果、变化差和方法说明 |
| `analysis.json` | 报告使用的完整计算结果 |
| `analysis_provenance.json` | 分析源文件哈希及本次使用的经济记录、身份和状态文件哈希 |
| `pair_stage_metrics.csv` | 每场每阶段的 8 项配对指标及平分比例的有效轮数 |
| `agent_stage_metrics.csv` | 每个 Agent 的阶段指标，包含真实发送者机会数及双方偏好 |
| `treatment_statistics.csv` / `period_statistics.csv` | 总体与逐轮均值、SD、四分位数、有效样本数和区间 |
| `personality_statistics.csv` | 本人、搭档偏好的分组关联及场次聚类区间 |
| `phase_changes.csv` / `treatment_contrasts.csv` | 同场前后变化、实验间水平与变化差、Welch 检验及 Holm 校正 |
| `insurance_summary.csv` | 第 16 轮保险人数与分母，以及无人、仅第一位、仅第二位、双方获保的场数与占比 |
| `insurance_statistics.csv` / `insurance_rates.csv` | 按获保人数的描述性比较，以及各人格的自愿购买率或环境获保率 |
| `termination_distribution.csv` | 第 48–50 轮进入、结束、继续的场数，以及当轮条件终止率和占全部自然结束场次的比例 |
| `record_coverage.csv` | 各实验已结算的发送者机会数、可读推理响应数、常规记忆更新和终局补记数，以及各项计数的覆盖场次 |
| `reciprocity_statistics.csv` | 按此前最近搭档发送者选择分组的本次行为 |
| `orientation_comparisons.csv` | A__B 与 B__A 的独立样本比较 |
| `preference_regressions.csv` | 控制本人和搭档两项偏好的分类变量回归，按场次聚类标准误 |

计算使用当前 AgentSociety2 环境已有的 NumPy、SciPy，不需要额外模型服务。配对阶段指标先对场内轮次平均，再对场次等权；个人指标先对每个 Agent 平均，再对 Agent 等权，并按 `job_id` 聚类。没有转账机会不计为选择 0。收入波动是个人阶段内实际收入的总体标准差，不是风险厌恶系数。

配对指标包括平均转账、正转账率、零转账率、每人平均收入、最终收入绝对差、个人收入波动、最终收入相等比例和转账相对于平分所需金额的比例。最后一项逐轮计算为 `2 × 实际转账 /（发送者转账前收入 − 接收者转账前收入）`，再取场内平均。比例为 1 表示当轮刚好平分，超过 1 表示转账超过平分所需金额。转账前收入缺失或发送者的收入优势不为正时，该轮无有效比例；比较两个阶段的比例变化时，各自都须有 15 个有效轮次。进入“全部已结算轮次”可以读取整场结果，包括第 46 轮以后的记录。

保险总体表的个人分母是有效两人账本覆盖的参与者人数，配对分母是有效账本对应的场数。25V 和 25F 分别报告自愿选择与环境分配；第一位、第二位使用计划中的配对顺序。自然终止表按第 48、49、50 轮列出进入该轮、在该轮结束、继续进入下一轮的场数。条件终止率用当轮结束数除以进入该轮数，结束轮次分布用当轮结束数除以全部自然结束场数。短程测试只报告其结算轮数。

证据覆盖表把每个已结算配对轮计为一次发送者机会。推理计数使用参与者响应，含决策和记忆调用。常规记忆更新与最终决策后的补记单列；每类计数给出实际提供元数据的场数，缺项和已观测的零可以分辨。

均值区间为 Student t 95% 区间；个人分组均值使用按场次聚类的截距模型区间。回归以个人阶段平均转账或正转账比例为因变量，本人和搭档偏好均按类别编码，低为参照；标准误采用 CR1 小样本修正与 t(G−1)。样本不足或设计矩阵不满秩时跳过，不用伪逆制造唯一系数。回归是探索性加性模型，尚未包含偏好交互项。

阶段变化先为每场计算后期减前期，再跨实验比较；要求两个阶段都恰有 15 轮。不同 treatment 的重复号并非匹配随机路径，因此跨实验使用独立样本 Welch 比较，不做按 r 编号配对的检验。所有水平与变化差检验共享一个 Holm 校正族，方向比较另成一个族；图中的区间仍是逐项区间，回归不标显著性星号。小样本或全零重复不能证明总体行为没有不确定性。

25V 的购买选择存在自选择，25F 的获保率是环境分配结果，不能解释为购买意愿。报告中的前期保险分组也按第 16 轮选择或分配追溯形成。保险分组、互惠条件和阶段变化均按描述性关联解释，未识别保险或记忆的因果效果。互惠条件严格使用此前已结算的搭档发送者选择，没有把无机会的记账零当作拒绝分享。报告不自动读取推理和私人记忆正文，原始文本仍可从每场链接查看。

汇总先在每场 session 内取均值，再对各次重复等权平均。固定阶段为 1–15、16–30、31–45，46 轮起的随机延长单列。轮次不是独立重复。短程 smoke 有独立标记，不能与完整实验当作同一条件混用。

组合均值只纳入通过经济结算、终局记忆和账本完整性校验的独立场次。每个计划的重复编号贡献一场。各表分别报告有效场次和分母，缺失字段显示为“—”。

## 离线验证

把计划的 `--backend llm` 改成 `--backend scripted`，把 `--full-session` 换成 `--completed-period-limit 3`，并使用新的批次目录，就可以小规模验证排程、数据标签和导出。要覆盖投保和保险撤销阶段，使用 `--completed-period-limit 31`。scripted 的金额不是人格行为研究结果。
