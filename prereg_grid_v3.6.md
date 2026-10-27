# prereg_grid_v3.6 — C2-M9 块二决定性批次 + 主检验封卷规格件（2026-10-27 午后）

> 基线 = `prereg_grid_v3.5.md`（sha b0c63826…）。铁律：先注册后运行；运行后规格零改动；重跑≠改规格。

## 0a｜主检验封卷（判据冻结 + 主张终结）
- **判据冻结（此后零改动）**：主检验 = 「**全体健康种子对一致 ≥ +1.0pp**」（即每一对「配置臂 vs cfg0」在同协议下 acc 差均 ≥ +1.0pp，才算主张成立）。
- **census 判决**：CFP-FP 上最优臂 `cfg3`(0.8101) − `cfg0`(0.8029) = **+0.72pp < +1.0pp** → **主检验主张终结，等级 = 趋势描述**（数据照报不删）。
- **反翻案条款**：主检验封卷后，**不得**因后续新臂/新协议/新种子而**重开**该主张；新臂只作**独立描述**，除非**新增预注册**并**全新判据**（不得追溯改判已封卷结论）。

## 0b｜KD 跨域判读预注册
- **叙事判据**：`cfg4` 系（含变体 `cfg4_seed42_s20_l01`）在 **LFW + CALFW 两列均 > cfg0**（**同号为正**）→ 「**域内保持 / 跨域受损**」叙事句**成立**（描述级）。
- **主张级仍需 +1.0pp**（与 0a 同判据）；**CFP-FP 列差照报**（不作叙事句依据，仅报数）。

## 0c｜TTA 政策（锚定义管线，全臂统一）+ CALFW fold 归一授权
- **TTA 政策**：以**锚定义的管线**为准（LFW 镜像腿 = flip-TTA 求和 + L2；CFP-FP 腿 = 单次前向 + pairwise cosine），**全臂统一**——同协议同管线，**学生 arm 与教师锚同管线**；JSON 必记 `tta` 字段。
- **CALFW fold 归一授权**：授权重生成 `calfw_manifest.csv`，**fold 归一到 0..9**（same: label−1；diff: 顺序 300/fold）→ **新 sha 入 ledger**；**旧件（sha 65879bca）标 legacy**（防误引）。

## 0d｜census 官方化
- **8 臂 CFP-FP 列入册**（三钉普查：cfg0/cfg1/cfg2/cfg3/cfg3_seed123/cfg4/cfg4_seed42_s20_l01/cfg5，acc 0.7780–0.8101 / AUC 0.8703–0.8885）。
- **臂册补种子列**（cfg0–cfg5 默认名 final 的种子归属须标；未定者标 `seed?`）。
- **健康臂计数勘误 7→8**：`cfg4_seed42_s20_l01` 为**变体臂**（非 cfg4 的种子臂），单列计；三者（cfg4 / s20_l01）并列示。

## 1｜LFW 铸造（`_eval_calib_mirror_v1.py --ckpt`）
- 加 `--ckpt`：EdgeFace 加载（`build_edgeface(DEFAULT_CONFIGS[cfg])` + 第四/五/六钉自查）+ **.pt 的 flip-TTA 求和 + L2**（与锚同管线）。
- **门一（教师）逐位 97.0833**；不过 → **不跑学生**。

## 2｜CALFW 铸造（fold 归一 + `calfw` dataset）
- fold 归一新 manifest（新 sha 入 ledger）→ `DATASET_CHOICES`+`calfw`（4 常量 + `load_manifest_rows` 分派）。
- **门一（CFP-FP 回归）+ 锚认证 0.967338**；不过 → 停。

## 3｜student cells（门一后，18:00 前收 GPU）
- **8 健康臂 × {LFW, CALFW}**（约半小时 GPU）；JSON 记 `tta` / 管线 / 第四·五·六钉字段。

## 4｜18:30 重训
- `cfg0@456 → cfg3@456`；纪律照 M8（`[ARM]` 行 / 每 epoch 心电图 / 钉自查 / 即刻 sha）。

## 5｜簿记清欠（第三轮不得再挂）
- CONFLICTS.md：M7 五块 + 第八例（负测试预期按规格范围写）+ `cfg3@42` 30s 验尸（正负对嵌入分布是否重叠，一行结论）+ 封卷始末 + 臂册种子列。
- **canary ② 读码 1 分钟**（已挂三批，本批必清）；Temp 清；commit（照实措辞）。

---
**注册信息**：本文件 = 规格 v3.6；repo 根 canonical + paper_plan 同步副本。**主检验封卷 = 趋势（+0.72pp < 1.0pp）；块二两腿门一后放 student cells；CALFW manifest 归一获授权（新 sha）。**
## 勘误（2026-10-27，B2 诊断收束）
- **CALFW 锚值勘误（第九例·口径差坐实）**：CALFW 锚 **0.967338**（C2-M3d 用 Mann-Whitney 全局 AUC）与 `eval_generic_pairs.py` 实测 **0.968887**（sklearn `roc_curve`+`auc`）**差 +0.0015**，经 D1/D2 诊断确认为**口径实现差**（非数据/故障）：
  - D1：B2 读脚本**重生成**的 `calfw_manifest.csv`（sha `fc2df448…`，LF）；与独立名 `calfw_manifest_v2.csv`（sha `dd8aa594…`，CRLF，差 6001≈6000 行尾）**内容一致**（同 fold 归一）。**旧 v1（sha 65879bca）已被 `write_manifest` 同名覆盖**（现不存在）—— 不变量受损，记之。
  - D2：以 C2-M3d 原法（Mann-Whitney）跑该 manifest → **AUC=0.967338（逐位同旧锚）**，`same 0.4357 / diff 0.0528`（逐位同 C2-M3d）→ **数据一致、两实现差坐实**。
- **裁定**：**CALFW 新基准锚 = `0.968887`（eval 口径）**；`0.967338` 标 legacy（Mann-Whitney 口径，不可与 eval 口径直接比）；**学生放行**。