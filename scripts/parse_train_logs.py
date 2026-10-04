"""
解析训练日志，提取 6 配置的 epoch 级 loss/lr 曲线，保存为 JSON。

日志文件映射（通过验证确认）:
  cfg0 (baseline):   logs_final_fix.txt   [1/6] baseline, 30 epochs
  cfg1 (+channel):   logs_cfg1-5_v3.txt   [1/5] +C1 channel, 30 epochs (first run)
  cfg2 (+spatial):   logs_cfg1-5_v3.txt   [2/5] +C1 spatial, 30 epochs (second run)
  cfg3 (+full SAB):  logs_cfg3-5_full.txt   [1/3] +C1 full SAB, 30 epochs
  cfg4 (+distill):   logs_cfg4-5_v4.txt     [1/2] +C1 +C2 distill, 30 epochs (first run)
  cfg5 (+aug):       logs_cfg5_v2.txt     [1/1] +C1 +C2 +C3 aug, 30 epochs
"""
import re
import json
import os

OUTPUT_DIR = "results/training_curves"

EPOCH_PATTERN = re.compile(
    r"\[epoch\s+(\d+)/(\d+)\]\s+avg_loss=([\d.]+)\s+lr=([\d.e\-+]+)\s+time=([\d.]+)s"
)

LOG_FILES = [
    "logs_final_fix.txt",
    "logs_cfg1-5_v3.txt",
    "logs_cfg3-5_full.txt",
    "logs_cfg4-5_v4.txt",
    "logs_cfg5_v2.txt",
]

CONFIG_MAP = {
    "baseline": "cfg0",
    "channel": "cfg1",
    "spatial": "cfg2",
    "full SAB": "cfg3",
    "full": "cfg3",
    "distill": "cfg4",
    "aug": "cfg5",
}

CONFIG_NAMES = {
    "cfg0": "baseline",
    "cfg1": "+channel",
    "cfg2": "+spatial",
    "cfg3": "+full SAB",
    "cfg4": "+distill",
    "cfg5": "+aug",
}


def extract_epochs(lines, start=0):
    """从指定行开始提取 epoch 数据，直到遇到下一个 config header 或文件末尾。"""
    epochs = []
    for i in range(start, len(lines)):
        if i > start and re.search(r"\[\d+/\d+\]", lines[i]) and "epoch" not in lines[i]:
            break
        m = EPOCH_PATTERN.search(lines[i])
        if m:
            epochs.append({
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "avg_loss": float(m.group(3)),
                "lr": float(m.group(4)),
                "time_s": float(m.group(5)),
            })
    return epochs


def parse_log_file(filepath):
    """解析单个日志文件，返回 {cfg_id: epochs_list}。"""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    results = {}
    for i, line in enumerate(lines):
        header_match = re.search(r"\[\d+/\d+\]\s+(.+?)\s+\(model=(\w+)", line)
        if header_match:
            desc = header_match.group(1).strip()
            model_key = header_match.group(2)

            cfg_id = None
            for keyword, cid in CONFIG_MAP.items():
                if keyword in desc:
                    cfg_id = cid
                    break

            if cfg_id is None:
                continue

            epochs = extract_epochs(lines, i + 1)
            if epochs:
                if cfg_id not in results or len(epochs) > len(results[cfg_id]):
                    results[cfg_id] = {
                        "config_id": cfg_id,
                        "config_name": CONFIG_NAMES.get(cfg_id, desc),
                        "description": desc,
                        "model_key": model_key,
                        "source_file": os.path.basename(filepath),
                        "epochs": epochs,
                    }
    return results


def main():
    all_configs = {}
    for log_file in LOG_FILES:
        if not os.path.exists(log_file):
            print(f"[skip] {log_file} not found")
            continue
        parsed = parse_log_file(log_file)
        for cfg_id, data in parsed.items():
            if cfg_id not in all_configs or len(data["epochs"]) > len(all_configs[cfg_id]["epochs"]):
                all_configs[cfg_id] = data
                print(f"  {cfg_id} ({data['config_name']}): {len(data['epochs'])} epochs from {log_file}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nParsed {len(all_configs)} configs\n")

    summary = []
    for cfg_id in sorted(all_configs.keys()):
        data = all_configs[cfg_id]
        epochs = data["epochs"]

        first = epochs[0]
        last = epochs[-1]
        min_loss = min(e["avg_loss"] for e in epochs)
        max_lr = max(e["lr"] for e in epochs)
        min_lr = min(e["lr"] for e in epochs)
        total_time = sum(e["time_s"] for e in epochs)

        print(f"  {cfg_id} ({data['config_name']}):")
        print(f"    source: {data['source_file']}")
        print(f"    epochs: {len(epochs)}, loss: {first['avg_loss']:.2f} -> {last['avg_loss']:.2f} (min={min_loss:.2f})")
        print(f"    lr: {max_lr:.2e} -> {min_lr:.2e}, total_time: {total_time/60:.1f} min")

        summary.append({
            "config_id": cfg_id,
            "config_name": data["config_name"],
            "description": data["description"],
            "model_key": data["model_key"],
            "source_file": data["source_file"],
            "num_epochs": len(epochs),
            "loss_first": first["avg_loss"],
            "loss_last": last["avg_loss"],
            "loss_min": min_loss,
            "lr_max": max_lr,
            "lr_min": min_lr,
            "total_time_min": round(total_time / 60, 1),
            "epochs": epochs,
        })

        out_path = os.path.join(OUTPUT_DIR, f"{cfg_id}_curve.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
