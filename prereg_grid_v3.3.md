# prereg_grid_v3.3 — C2-M6 臂基础重建规格件（2026-10-27 白天）

> 基线 = `prereg_grid_v3.2.md`（sha 2bf6c82b…）。合法窗口：有效官方 cell=0（G01c 撤回）。
> 铁律：先注册后运行；运行开始后规格零改动；重跑≠改规格。

## 0｜第四钉（ckpt meta final_epoch + 评测 assert==30）——已铸造
- `eval_generic_pairs.py` 增：`_sidecar_final_epoch(path)` 读 ckpt `epoch`；`build_backend` 的 EdgeFace 分支在 `load_state_dict` 后**断言 `int(final_epoch)==30`**，否则 `exit(8)`；payload 记 `final_epoch`（onnx 教师为 None）。
- 负测试：`cfg4_seed456_final.pt`（epoch=5）→ **exit=8 拒绝** ✓（2026-10-27）。
- 铸造 commit `c5634e8`。

## 1｜网格裁剪 40→16 cell 声明
- **核心层（3 臂 × 3 协议 = 9 cell，声明）**：`cfg0/cfg3/cfg4` @ **seed456** × {LFW, CALFW, CFP-FP}
  ——**但见 §重大发现：seed456 臂 epoch=5 未完、且 seed42 臂湮灭 → 核心层当前有效臂基础不成立**。
- **方差层（cfg3 4 种子）**：`cfg3` × seed{42,123,456,789}（仅 CFP-FP）
- **补充**：`cfg0@42`（1 cell）
- 合计 = 9 + 4 + 1 = **14**（执行框「16 cell」含 CFP-FF 2 cell 留待 C2-M4b）。

## 2｜G01c/G02/G03 标注
- `G01c_cfg0_seed456` / `G02_cfg3_seed456` / `G03_cfg4_seed456` 的 arm 均 **epoch=5（未完）** → 标 **epoch=5 非官方**，**保留为训练动态快照**（不删，防事后嫌疑），**不入官方 cell 表**。

## 3｜执行模式改前台逐 cell/逐臂
- 弃「挂夜单进程 Start-Process」（G04 崩溃 + 电脑崩溃）→ **前台逐 cell / 逐臂**执行，逐条回报（监控友好、异常可控）。

## ⚠️ 重大发现（2026-10-27，本批实测；改变网格可行性）
- 第四钉把关后跑 **3 个 epoch=30 完成臂 × CFP-FP**：
  | cell | arm | acc | AUC | epoch |
  |---|---|---|---|---|
  | G20 | cfg0_seed42 | **0.5000** | **0.5003** | 30 |
  | G21 | cfg3_seed42 | **0.4994** | **0.5122** | 30 |
  | G22 | cfg3_seed123 | 0.7780 | 0.8703 | 30 |
- **epoch==30 ≠ 健康**：`cfg0_seed42`/`cfg3_seed42` 虽 epoch=30，却在 CFP-FP 上**完全随机（AUC≈0.50）** → 与记忆载明的「cfg0_seed42=50.00% 湮灭、cfg3_seed42=53.72% 崩溃」一致，属**权重湮灭/崩溃臂**。
- **当前真正健康的臂仅 `cfg3_seed123`（AUC 0.8703）**（+ 待重训的 seed456）。
- **建议第五钉（magnitude 健康门）**：评测前加 `med|w|` 下限（如 `med<1e-8` 或 `frac(<1e-30)>1%` → 拒），拦「epoch=30 但湮灭」臂（C2-M2 的 `_ckpt_health.py` 逻辑）。**此项为建议，待用户裁**（不自行增钉）。

## 4｜18:30 重训
- `cfg0@456 → cfg3@456`（前台/监控；日志头必含 `[ARM] cfg/seed/config_sha/commit`；ckpt 落盘即刻 ≤ 10 min 记 sha）。
- 重训完成（epoch=30 + magnitude 健康）→ 补核心层 cell。

## 5｜簿记
- CONFLICTS.md：补**臂基础事件始末**（A* 判定未铸机制之教训：epoch<10 快照未拦 → 第四钉）；Temp 清；commit（照实措辞）。

---
**注册信息**：本文件 = 规格 v3.3；repo 根 canonical + paper_plan 同步副本。**网格规模（声明） = 14 cell（核心 9 + cfg3 方差 4 + cfg0@42），当前有效臂基础 = 1（cfg3_seed123），待重训补。**