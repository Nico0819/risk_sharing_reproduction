# Baseline 实验入口

这是当前 Baseline 的最小工程入口：四个独立 `PersonAgent`、两个固定 pair，
通过 AgentSociety2 2.8.3 的 `ReActRouter` 和项目适配层进入同一个
`RiskSharingEnv`。它只完成转账阶段的三轮工程试跑；论文最终随机单轮支付尚未实现，
因此输出不能解释为完整论文复现。

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

## 离线验证

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend scripted --participants 4 --completed-period-limit 3 --seed 2026
```

该命令不联网，使用本地 scripted 模型响应，但仍运行真实
`PersonAgent -> ReActRouter -> RiskSharingRouterAdapter -> @tool -> RiskSharingEnv`。
每次运行在 `experiments/baseline/runs/` 生成独立目录，包含 `manifest.json`、
`agent_actions.jsonl`、`period_records.jsonl`、`summary.md` 和结构化诊断日志 `run.log`。

## 真实 API（填写配置后再运行）

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m experiments.baseline.run_smoke \
  --backend deepseek --participants 4 --completed-period-limit 3 --seed 2026
```

请求上限为 80 次、单次最大输出 2048 tokens、总时限默认 900 秒；鉴权失败、余额不足
或预算耗尽会停止，不会自动换供应商或循环重试。当前尚未执行真实联网试跑；请在填写
`.env` 后明确通知“API 配置已填写，可以按原定上限进行真实试跑”，再运行此命令。

## 测试

```bash
cd /home/zhicheng/risk_sharing_reproduction
.venv/bin/python -m pytest -q
```

测试包括协议/环境单元测试、配置安全检查和一次离线真实框架组件链路检查；不把离线
scripted 结果当作 DeepSeek API 成功。
