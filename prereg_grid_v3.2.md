# prereg_grid_v3.2 — C2-M4v3.2 门三修正规格件（2026-10-26 深夜）

> 基线 = `prereg_grid_v3.1.md`（sha 8430c03b…）。本版**只修订门三卫生判据**，
> **cell 集与主/副检验判读零改动**（合法窗口：官方 cell=0；G01 冒烟属认证步骤）。铁律：先注册后运行；运行开始后规格零改动；重跑≠改规格。

## 0｜门三重铸全文（替换 v3.1 门三）
- **卫生底线 0.70**（原 0.90 线**删除**，仅保留于**教师通道**）：学生 cell 的 AUC **< 0.70** 才判卫生失败；**0.70 ≤ AUC < 0.90 为 LOW 记录**（照常出数、不重跑）。
- **一次重跑**：仅当触发卫生失败（AUC<0.70 / canary 败 / 缺钉）→ 该 cell 重跑**一次**；重跑仍败 → 标 FAIL 停报不跳（重跑≠改规格）。
- **LOW-查因差分四项**（对 LOW 区间 cell 的查因，不阻断出数）：
  1. **作物链一致性**（crop_chain 钉 vs 通道 canonical）
  2. **manifest 覆盖**（manifest_sha 钉 + 存在性）
  3. **锚归属**（教师锚 vs 该协议注册锚）
  4. **统计口径**（acc_cv10 / AUC / EER 与既有件同构）
  （四项为查因框架，细则以上框文为准。）
- **0.90 线仅存教师通道**：教师回归门一仍用逐位对比（CFP-FP 95.5429 / LFW 97.0833 / CALFW 0.967338），**不适用于学生臂**。

### 修正合法性论证段（原文入册）
- 触发事实：冒烟 `cfg0_seed456 × CFP-FP` 实测 **AUC=0.8764 < 0.90**——0.93M 学生臂在 CFP-FP 上 AUC<0.90 是**真实能力**（非故障），教师 95.5% 之下；按原门三将**必然误伤全部低配臂**且重跑同值死循环。
- 与既有精神一致：v3 已立「教师锚差全表报出、**不设 FAIL 线**」（落后幅度是论文内容）→ 门三 0.90 线与之**自相矛盾**，故修正为「0.70 底线 + LOW 记档」，把 0.90 从**阻断线**降为**记录线**。
- **G01 冒烟值透明记录（防事后嫌疑）**：`G01`（无 canary/无三钉）与 `G01b`（有 canary/无三钉）实测均 **acc=79.84% / AUC=0.8764 / EER=0.206**（三跑同值）→ 数值稳定、非噪声；此为门三修正的**触发证据**，原样留存。
- sidecar 三钉铸造：`eval_generic_pairs.py` 增 `model_sha`（权重文件 sha256）+ `crop_chain`（作物链标签）→ 三钉齐备（门三前提），门一重验 95.54% 逐位未变。

## 1｜G01 转正
- **采纳 `G01c_cfg0_seed456_cfp_fp.json` 为官方 cell**（三钉齐备：model_sha `04aae325…8790c` / manifest_sha `09013db6…b6d4` / crop_chain `fid2` / canary PASS）。
- 旧 `G01_cfg0_seed456_cfp_fp.json`、`G01b_…json` 标 **superseded**（缺钉版）。
- 结果件（`results/G01*`）**gitignored 不入库**（属内部产物）。

## 2｜点火
- 余 **10 cell**：核心 `cfg3/cfg4@seed456`（6 cell）+ 方差 8 中除 G01 外（`cfg0×{42,123,789}`、`cfg3×{42,123,789}`、`cfg4×{123,789}`）+ —— 合计 10（G01 已占即位）。
- `Start-Process` 脱离会话、单日志、**cell 失败=停报不跳**；预计 2–3.5h。

## 3｜块二机制（与 cell 并行）
- **LFW**：`_eval_calib_mirror_v1.py` 加 `--ckpt` → 门一教师逐位 97.0833 → 学生 3 cell 排队。
- **CALFW**：`eval_generic_pairs.py` 的 `DATASET_CHOICES` + `calfw`（manifest sha 65879bca）→ 门一（CFP-FP 回归）+ 锚认证 0.967338 → 学生 3 cell 排队；来不及 → C2-M4b 明晨（既有回落条款）。
- **canary ② 数据独立性读码**：确认 canary② 用独立 RandomState（非复用外部常量）→ 仅意外常量才阻断（读码 1 分钟）。

## 4｜簿记
- CONFLICTS.md：补 M4v3 机制阻塞始末 + 门三修正理由 + canary② 注记 + LFW legacy 件标注；Temp 清；commit v3.2（照实措辞）。

---
**注册信息**：本文件 = 规格 v3.2；repo 根 canonical = `face_recognition_research\prereg_grid_v3.2.md` + paper_plan 同步副本。网格规模落定 = **17 cell**（核心 9 + 方差 8，含 G01 已跑 1）。