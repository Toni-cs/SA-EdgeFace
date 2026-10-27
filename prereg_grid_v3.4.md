# prereg_grid_v3.4 — C2-M7 第五钉 + 臂普查 + 核心重训规格件（2026-10-27 午后）

> 基线 = `prereg_grid_v3.3.md`（sha a60a4062…）。铁律：先注册后运行；运行开始后规格零改动；重跑≠改规格。

## 0｜第五钉（湮灭门）——已铸造
- `eval_generic_pairs.py` 增 `_sidecar_weight_health(path)` → `(med|w|, frac(<1e-30))`；
  `build_backend` 的 EdgeFace 分支在第四钉后**断言 `med|w| ≥ 1e-8 ∧ frac ≤ 0.01`**，否则 `exit(9)`；payload 记 `weight_med` / `weight_frac`。阈值**沿用 C2-M2 `_ckpt_health.py` 标定**。
- 铸造 commit `0b4a41b`。

### 负测试三连（实测，含偏差）
| 臂 | 预期 | 实测 | 判定 |
|---|---|---|---|
| `cfg0_seed42` | 必 FAIL | `exit=9`（med=2.44e-10 湮灭）→ 被拦 | ✅ 符合 |
| `cfg3_seed42` | 必 FAIL | **`exit=0` 未拦**，ACC=0.4994 / AUC=0.5122（med=6.78e-03 正常） | ❌ **偏差** |
| `cfg3_seed123` | 必 PASS | `exit=0`，ACC=0.7780 / AUC=0.8703 | ✅ 符合（**G22 证书补齐**） |

**⚠️ 第五钉局限（须裁）**：magnitude 门拦得住「**数值湮灭**」（cfg0@42），**拦不住「数值正常但功能崩坏」**（cfg3@42 实测随机）→ **`cfg3@42 必 FAIL` 预期不成立**。**建议第六钉（功能门）**：评测前对「功能健康」加实测判据（如最小 AUC/rank1 门槛），或对 magnitude 通过的臂强制**小样本功能预检**。**此项为建议，待用户裁。**

## 0.1｜seed42 预声明条款
- `seed42` 系列臂（cfg0/cfg3_seed42）**已知存在湮灭/崩坏前科** → 其任何出数**须先过双钉 + 功能预检**方可入官方表；未过者标「seed42 崩坏·非官方」。

## 0.2｜崩溃续训条款
- 训练/评测遇 `exit=-1073741819`（0xC0000005）或机器崩溃 → 判**瞬时**（G04 已证不复现），**允许从最近 checkpoint 续训/重跑**，须记 `resume` 时间戳与源 ckpt sha。

## 0.3｜G 件标注
- `G20_cfg0_seed42`（ACC 0.5000/AUC 0.5003）→ 标 **湮灭·非官方**；`G21_cfg3_seed42`（0.4994/0.5122）→ 标 **湮灭·非官方**。
- **`G22_cfg3_seed123`（0.7780/0.8703）→ 入 ledger 为官方 cell**（首批有效官方数字）。

## 1｜臂普查（全 ckpt × {final_epoch, med|w|, frac}）
**通过双钉（epoch=30 ∧ 非湮灭）= 9 臂**：
```
cfg0_final(30,7.84e-03) / cfg1_final(30,9.12e-03) / cfg2_final(30,1.04e-02)
cfg3_final(30,7.34e-03) / cfg3_seed123(30,2.03e-02) / cfg3_seed42(30,6.78e-03)
cfg4_final(30,1.05e-02) / cfg4_seed42_s20_l01(30,1.91e-02) / cfg5_final(30,5.49e-03)
```
**重训队列 = 11+1 = 12**：
- epoch=5 未完（10）：cfg0_seed123/456/789、cfg3_seed2024/456/789、cfg4_seed123/2024/456/789
- **数值湮灭**（1）：cfg0_seed42
- **功能崩坏（magnitude 正常）**（1）：**cfg3_seed42**（← 第五钉漏拦，功能预检补捉）

## 2｜块二铸造（照 M6 框）
- LFW `_eval_calib_mirror_v1.py` 加 `--ckpt` → 门一锚认证 **97.0833 逐位**
- CALFW `eval_generic_pairs.py` `DATASET_CHOICES` + `calfw`（manifest sha 65879bca）→ 门一 + 锚认证 **0.967338**
- → `cfg3@123 × LFW/CALFW` 首批 cell；canary ② 读码 1 分钟（遗留项，非阻塞）

## 3｜18:30 重训（前台监控）
- `cfg0@456 → cfg3@456`；`[ARM] cfg/seed/config_sha/commit` 行 + **每 epoch 尾记 `med|w|`** + ckpt 即刻 sha + **当场双钉自查**。

## 4｜簿记
- CONFLICTS.md 补「**已知≠已执行**」模式（第七例始末）；Temp 清；commit（照实措辞）。

---
**注册信息**：本文件 = 规格 v3.4；repo 根 canonical + paper_plan 同步副本。**网格有效臂基础（双钉通过）= 9；官方 cell 实得 = 1（G22 cfg3_seed123）。**