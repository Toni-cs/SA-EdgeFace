# prereg_grid_v3.1 — C2-M4v3.1 机制修订规格件（2026-10-26 深夜）

> 基线 = `prereg_grid_v3.md`（sha 7137eb99…）。本版**只修订「机制」层（评测通道与三道门）**，
> **cell 集与判读规则零改动**（合法窗口：0 cell 已跑）。铁律：先注册后运行；运行开始后规格零改动；重跑≠改规格。

## 0｜机制条款（替换 v3 的「既往已评 arm 同一机制」）
### 0.1 同构律
- 学生 arm 与教师件**同作物、同 manifest、同预处理、同评测栈**；唯一变量 = 模型权重。
- 逐协议 canonical 通道：
  | 协议 | 作物根 | manifest | 教师锚 | 评测脚本 |
  |---|---|---|---|---|
  | CFP-FP | `datasets\cfp_aligned_fid2` | `eval_manifests_fid2\cfp_fp_manifest.csv`（sha 09013db6） | 95.5429 | `eval_generic_pairs.py --ckpt` |
  | LFW | `datasets\lfw_mirror_112` | LFW pairs（内嵌） | 97.0833 | `_eval_calib_mirror_v1.py --ckpt`（本批铸造） |
  | CALFW | `datasets\calfw_aligned` | `calfw_manifest.csv`（sha 65879bca） | 0.967338 | `eval_generic_pairs.py --dataset calfw`（本批铸造） |

### 0.2 三道门（每通道铸造/发射前必过）
- **门一（教师回归）**：canonical 通道跑**教师**，与注册锚**逐位对比**——CFP-FP=95.5429／LFW=97.0833／CALFW=0.967338；不逐位 → 停、报回。
- **门二（canary 语义）**：每 cell 必带 canary ①=**100%**（同图自比，须精确 100%）、②≈**50%**（随机对照，逐 cell 记实测值，容差内即过）。
- **门三（卫生）**：缺钉（model_sha/manifest_sha/crop_chain 任一）/ AUC<0.90 → cell 无效重跑（重跑≠改规格）。
- （注：门二/门三的**逐位判据细则**若与本框上文有异，以上文为准；本落盘为可见全文。）

### 0.3 CALFW 锚认证/重铸条款
- 门一须认证教师 CALFW **AUC=0.967338**（v3 C2-M3d 新锚）；不达标 → 按「重铸」处理：重跑教师、记新锚、旧锚转历史注记、报回。

### 0.4 canary 语义条款
- canary ①=100%（同图自比）、②≈50%（随机对照）；**逐 cell 记实测**（非固定 49.625%，仅作参考带）。

### 0.5 重训条件预注册（cfg1/2/5）
- 条件：`cfg1/2/5 @ seed456`，**钉当前 `ABLATION_CONFIGS` sha + git HEAD**；
- 日志头必含 `[ARM] cfg=<i> seed=<s> config_sha=<..> commit=<..>`（出身法证教训）；
- GREEN（seed 落定 + config 可验证）→ **自动补 9 cell**（3 臂 × 3 协议），主检验**扩全 6 臂集**；
- **干净 3 臂结果保留为中期报告**（不因扩集而作废）。

## 1｜CFP-FP 通道（今晚发射 11 cell）
- a. `--ckpt` 实存 ✅（`eval_generic_pairs.py:543-551` 学生 .pt 分支完整）→ **无需改脚本**
- b. 门一（教师逐位 95.5429）
- c. 冒烟 `G01_cfg0_seed456_cfp_fp`（canary 齐 + 数字合理）→ `Start-Process` 发射其余 10（核心 3 → 方差 8）；cell 失败 = 停报不跳

## 2｜LFW/CALFW 通道（机制铸造 → 认证 → 发射）
- CALFW：`DATASET_CHOICES` 加 `calfw`（manifest sha 65879bca）→ 门一（教师 CFP-FP 回归）+ 锚认证（vs 0.967338）→ 学生 3 cell
- LFW：`_eval_calib_mirror_v1.py` 加 `--ckpt` → 门一（教师逐位 97.0833）→ 学生 3 cell

## 3｜簿记
- CONFLICTS.md：M4v3 机制阻塞始末 + 本批裁定 + v3.1 sha；LFW 旧学生件（6000 对口径、临时脚本产物）标 **legacy**
- commit（照实措辞，未发生项不入 message）

---
**注册信息**：本文件 = 规格 v3.1；repo 根 canonical = `face_recognition_research\prereg_grid_v3.1.md`，paper_plan 同步副本 = `paper_plan\prereg_grid_v3.1.md`。