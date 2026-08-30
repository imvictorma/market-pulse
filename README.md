# Market Pulse

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2ea44f)](https://maqianxiong.github.io/market-pulse/)
[![Daily Update](https://github.com/maqianxiong/market-pulse/actions/workflows/daily.yml/badge.svg)](https://github.com/maqianxiong/market-pulse/actions/workflows/daily.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Market Pulse 是一个面向长期投资者的纳斯达克 100 市场温度仪表盘。它在美股收盘后自动拉取行情，计算透明的五维温度模型，生成移动端优先的静态网页，并可通过 Server酱推送到微信。

它不是涨跌预测器。温度越高，表示估值、情绪、趋势、仓位与利率环境的“逆向配置吸引力”越高；页面中的定投倍率只是模型输出，不构成投资建议。

## 已实现

- 自动行情：纳指100、标普500、QQQ、VXN、美债10年期收益率
- 自动指标：1/5/20 日涨跌、Wilder RSI(14)、MA200 偏离、VXN 756 个交易日分位
- 手工指标：Forward PE、TTM PE、CNN Fear & Greed、NAAIM、利率环境评分
- 五维透明评分、温度区间、参考定投倍率和事件提醒
- `data/history.csv` 按数据日 upsert，同日重跑不重复
- 失败时复用上一份成功行情并标为 `stale`；没有旧值则显示“不可用”，不会造数
- 自包含的 HTML/CSS/JS 仪表盘，不依赖前端框架或 CDN
- GitHub Actions 工作日自动更新、提交历史、部署 Pages、发送 Server酱通知

## 本地运行

需要 Python 3.12。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate.py --no-push
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
python generate.py --no-push
```

生成结果为 `site/index.html`。在仓库根目录运行下面的命令即可本地预览：

```bash
python -m http.server 8000 --directory site
```

浏览器打开 `http://localhost:8000`。

## 配置真实手工数据

首次运行前，请编辑 `config.json` 的 `manual_inputs`。默认值全部是 `null`，这是有意设计：在你填入真实数据前，系统会显示“数据不足”，不会拿示例数冒充市场数据。

每个手工项都建议填写：

- `value`：当前绝对值；PE 项必须填写
- `percentile`：只有两项 PE 需要，范围 0–100
- `as_of`：数据日期，格式 `YYYY-MM-DD`
- `source`：数据来源名称或链接说明
- `percentile_window`：PE 分位窗口，例如 `10Y`
- `max_age_days`：超过多少天后标为过期

示例结构（数字仅说明格式，请替换为你核验过的真实值）：

```json
"forward_pe": {
  "value": 25.0,
  "percentile": 60,
  "as_of": "2026-08-06",
  "source": "你的数据来源",
  "percentile_window": "10Y",
  "max_age_days": 45
}
```

模型阈值、五维权重、MA200 分段、温度倍率和提醒阈值都位于 `config.json`，无需修改 Python 代码。修改权重后，程序会校验每组权重之和是否为 1。

## 从零部署到 GitHub Pages

1. 在 GitHub 新建仓库，将本目录全部文件提交并推送到默认分支。
2. 打开仓库 `Settings → Pages`，在 `Build and deployment` 中将 Source 设为 **GitHub Actions**。
3. 打开 `Settings → Secrets and variables → Actions`，新建 Repository secret：
   - 名称：`SERVERCHAN_SENDKEY`
   - 值：从 Server酱后台重新生成的 SendKey
4. 不要复用曾经贴进聊天或截图里的旧 Key，也不要把 Key 写入 `config.json`、代码或提交历史。
5. 打开 `Actions → Daily Market Pulse → Run workflow` 手工执行一次。
6. 工作流成功后，Pages 地址会出现在部署任务和仓库 Pages 设置页。

工作流使用 `30 23 * * 1-5`，即 UTC 周一至周五 23:30，对应北京时间次日 07:30。GitHub 的定时任务可能延迟几分钟。美股休市时通常会 upsert 最近的数据日，而不会伪造当天行情。

> 如果仓库启用了默认分支保护，GitHub Actions 机器人可能无法直接推送生成文件。请为工作流授予写入权限，或调整分支规则允许 `github-actions[bot]` 写入。仓库 `Settings → Actions → General → Workflow permissions` 需选择 **Read and write permissions**。

## 数据失败与新鲜度

页面使用四种状态：

- `最新`：本次成功取得的自动行情
- `手工`：来自 `config.json` 且仍在配置的新鲜期内
- `已过期`：抓取失败后沿用旧行情，或手工数据超过 `max_age_days`
- `不可用`：既没有本次数据也没有可回退的旧值

行情的上一份完整快照存放在 `data/latest.json`。若单个 ticker 抓取失败，其他 ticker 仍会继续生成；推送中会列出失败项。任何一个五维评分缺失时，总温度会明确显示不可用，不会按剩余权重偷偷重算。

## 模型公式

```text
估值 = (100 - Forward PE 分位) × 62.5%
     + (100 - TTM PE 分位) × 37.5%

情绪 = VXN 分位 × 60%
     + (100 - CNN Fear & Greed) × 40%

趋势 = (100 - RSI) × 50%
     + MA200 偏离分 × 50%

资金 = 100 - NAAIM
宏观 = 手工利率环境评分

总温度 = 估值 × 40% + 情绪 × 25% + 趋势 × 20%
        + 资金 × 10% + 宏观 × 5%
```

默认温度映射：冰点 ≥80（3x）、偏冷 ≥65（2x）、正常 ≥50（1x）、偏热 ≥35（0.6x）、过热 ≥20（0.3x）、极热 <20（0x）。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖指标计算、模型公式、关键阈值边界、缺失数据行为和历史记录 upsert。
