"""实时训练监控面板：读取日志生成 HTML 可视化。"""
import os, re, json, time
from datetime import datetime

LOG_FILE = "logs_cfg1-5_v3.txt"
OUTPUT = "training_monitor.html"

def parse_log():
    configs = []
    current_config = None
    epochs = []
    results = {}
    
    if not os.path.exists(LOG_FILE):
        return {"configs": [], "current": None, "epochs": [], "results": {}}
    
    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 配置开始
            m = re.search(r'\[(\d)/5\]\s+(.+?)\s+\(model=(.+?),\s+cfg(\d)\)', line)
            if m:
                if current_config:
                    configs.append({"name": current_config, "epochs": epochs})
                current_config = "cfg%s" % m.group(4)
                epochs = []
                continue
            
            # epoch 完成
            m = re.search(r'\[epoch (\d+)/30\]\s+avg_loss=([\d.]+)\s+lr=([\d.e-]+)\s+time=([\d.]+)s', line)
            if m and current_config:
                epochs.append({
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "lr": float(m.group(3)),
                    "time": float(m.group(4))
                })
                continue
            
            # RESULT
            m = re.search(r'RESULT:\s+AUC=([\d.]+)\s+Acc=([\d.]+)%\s+Sep=([\d.]+)', line)
            if m and current_config:
                results[current_config] = {
                    "auc": float(m.group(1)),
                    "acc": float(m.group(2)),
                    "sep": float(m.group(3))
                }
                continue
    
    if current_config:
        configs.append({"name": current_config, "epochs": epochs})
    
    return {
        "configs": configs,
        "current": current_config,
        "results": results,
        "updated": datetime.now().strftime("%H:%M:%S")
    }

def generate_html(data):
    # 准备图表数据
    all_series = []
    for cfg in data["configs"]:
        epochs = [e["epoch"] for e in cfg["epochs"]]
        losses = [e["loss"] for e in cfg["epochs"]]
        all_series.append({"name": cfg["name"], "epochs": epochs, "losses": losses})
    
    # 当前进度
    current_cfg = data["current"]
    current_epochs = data["configs"][-1]["epochs"] if data["configs"] else []
    current_epoch = current_epochs[-1]["epoch"] if current_epochs else 0
    current_loss = current_epochs[-1]["loss"] if current_epochs else 0
    
    # 已完成配置
    completed = list(data["results"].keys())
    
    # 预计剩余时间
    avg_time = 180  # 秒/epoch
    if current_epochs:
        avg_time = sum(e["time"] for e in current_epochs[-5:]) / min(5, len(current_epochs))
    
    remaining_epochs = (30 - current_epoch) + (5 - len(data["configs"])) * 30 + (len(data["configs"]) - len(completed) - 1) * 30
    remaining_minutes = remaining_epochs * avg_time / 60
    
    series_json = json.dumps(all_series)
    results_json = json.dumps(data["results"])
    
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>训练实时监控</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
h1 { color: #00d4ff; margin-bottom: 5px; }
.subtitle { color: #888; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card { background: #16213e; border-radius: 12px; padding: 20px; border: 1px solid #0f3460; }
.card h2 { color: #00d4ff; font-size: 16px; margin-top: 0; }
.stat { font-size: 36px; font-weight: bold; color: #e94560; }
.stat-label { color: #888; font-size: 14px; }
.progress-bar { background: #0f3460; border-radius: 10px; height: 24px; overflow: hidden; margin: 10px 0; }
.progress-fill { background: linear-gradient(90deg, #00d4ff, #e94560); height: 100%; transition: width 0.5s; }
.config-list { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
.config-badge { padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; }
.done { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff; }
.running { background: #e9456022; color: #e94560; border: 1px solid #e94560; animation: pulse 1.5s infinite; }
.pending { background: #333; color: #666; border: 1px solid #444; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #0f3460; }
th { color: #00d4ff; }
#chart { background: #16213e; border-radius: 12px; padding: 20px; margin-top: 20px; }
</style>
</head>
<body>
<h1>人脸识别人脸消融训练监控</h1>
<div class="subtitle">更新时间: __UPDATED__ | 自动刷新中...</div>

<div class="grid">
  <div class="card">
    <h2>当前配置</h2>
    <div class="stat">__CURRENT_CFG__</div>
    <div class="stat-label">Epoch __CURRENT_EPOCH__/30 | Loss __CURRENT_LOSS__</div>
    <div class="progress-bar"><div class="progress-fill" style="width: __PROGRESS__%"></div></div>
    <div class="stat-label">__PROGRESS__%</div>
  </div>
  <div class="card">
    <h2>预计剩余</h2>
    <div class="stat">__REMAINING__</div>
    <div class="stat-label">分钟 | 约 __REMAIN_HOURS__ 小时</div>
  </div>
</div>

<div class="card" style="margin-top: 20px;">
  <h2>配置进度</h2>
  <div class="config-list">
    __CONFIG_BADGES__
  </div>
</div>

<div class="card" style="margin-top: 20px;">
  <h2>已完成结果</h2>
  <table>
    <tr><th>配置</th><th>AUC</th><th>Acc</th><th>Sep</th><th>状态</th></tr>
    __RESULT_ROWS__
  </table>
</div>

<div id="chart">
  <h2 style="color: #00d4ff; margin-top: 0;">Loss 曲线</h2>
  <canvas id="lossChart" height="100"></canvas>
</div>

<script>
const series = __SERIES__;
const colors = ['#00d4ff', '#e94560', '#00ff88', '#ffaa00', '#ff66ff', '#88ff00'];
const datasets = series.map((s, i) => ({
  label: s.name,
  data: s.losses.map((l, idx) => ({x: s.epochs[idx], y: l})),
  borderColor: colors[i % colors.length],
  backgroundColor: colors[i % colors.length] + '22',
  tension: 0.3,
  pointRadius: 3
}));

new Chart(document.getElementById('lossChart'), {
  type: 'line',
  data: { datasets },
  options: {
    responsive: true,
    scales: {
      x: { title: { display: true, text: 'Epoch', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#0f3460' } },
      y: { title: { display: true, text: 'Loss', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#0f3460' } }
    },
    plugins: { legend: { labels: { color: '#eee' } } }
  }
});

setTimeout(() => location.reload(), 30000);
</script>
</body>
</html>"""
    
    # 配置徽章
    badges = ""
    all_cfgs = ["cfg0", "cfg1", "cfg2", "cfg3", "cfg4", "cfg5"]
    for cfg in all_cfgs:
        if cfg in data["results"]:
            badges += '<span class="config-badge done">%s ✓</span>' % cfg
        elif cfg == current_cfg:
            badges += '<span class="config-badge running">%s 运行中</span>' % cfg
        elif cfg == "cfg0":
            badges += '<span class="config-badge done">cfg0 ✓ (89.45%%)</span>'
        else:
            badges += '<span class="config-badge pending">%s 等待</span>' % cfg
    
    # 结果表格
    result_rows = ""
    if "cfg0" not in data["results"]:
        result_rows += "<tr><td>cfg0</td><td>0.9576</td><td>89.45%</td><td>0.3443</td><td style='color:#00d4ff'>已完成</td></tr>"
    for cfg, r in sorted(data["results"].items()):
        result_rows += "<tr><td>%s</td><td>%.4f</td><td>%.2f%%</td><td>%.4f</td><td style='color:#00d4ff'>已完成</td></tr>" % (cfg, r["auc"], r["acc"], r["sep"])
    if not result_rows:
        result_rows = "<tr><td colspan='5' style='color:#666'>暂无完成结果</td></tr>"
    
    html = html.replace("__UPDATED__", data["updated"])
    html = html.replace("__CURRENT_CFG__", current_cfg or "等待中")
    html = html.replace("__CURRENT_EPOCH__", str(current_epoch))
    html = html.replace("__CURRENT_LOSS__", "%.4f" % current_loss)
    html = html.replace("__PROGRESS__", "%.0f" % (current_epoch / 30 * 100))
    html = html.replace("__REMAINING__", "%.0f" % remaining_minutes)
    html = html.replace("__REMAIN_HOURS__", "%.1f" % (remaining_minutes / 60))
    html = html.replace("__CONFIG_BADGES__", badges)
    html = html.replace("__RESULT_ROWS__", result_rows)
    html = html.replace("__SERIES__", series_json)
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    
    print("监控面板已生成: %s" % OUTPUT)
    print("当前: %s epoch %d/30 loss=%.4f" % (current_cfg, current_epoch, current_loss))
    print("已完成: %s" % ", ".join(completed) if completed else "无")

if __name__ == "__main__":
    data = parse_log()
    generate_html(data)
