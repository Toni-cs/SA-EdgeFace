# CONFLICTS — 冲突登记与裁决（立案日期：2026-10-10）

## §1 文档目的
登记实验过程中出现的方法论冲突、数据争议与裁决结果，append-only，原句保留。

## §2 G2/G2' 阶段冲突（已裁决）
- C-1 锚点数值错误：cfg3 锚为 93.6333%，非 94.0500%（后者属 cfg2）。裁决：门限不变，更正对照。
- C-2 同名档语义混淆：`cfg3_seed123_final.pt` 旧语义为 5ep 筛选档（epoch=5），非 30ep。30ep 正确对照为 77.82%。裁决：以 `_hist` 备份为权威。
- C-3 归因边界：不归因于 2026-10-09 初始化播种语义变更；跨 era 数字不得互比。

## §3 D4 阶段冲突（当前立案）

### 3.1 clip 因果争议（P3 vs P4，终裁 P4）
- **P3 假说**：`gradclip_by_commit.txt` 显示 era A 唯一健康血统（cfg3_final=93.63）使用 `clip=0.0`；era B/C 全部 5 档 FAIL（50.00/53.72/77.82/55.22/86.90）均使用 `clip=10.0`。推定 clip=10.0 为根因。
- **P4 终裁**：clip 与 era 完全共线（era 另差 RNG regime 与 D-14 fp16 修复态，D-13(1)）。按 D-12(4)/D-13(1) 跨 era 不得互比 → **clip 因果不可判**。
- **禁句 26 生效**：不许写成「既证根因」，只能写机制假设。任何文书中出现「clip=10.0 导致 FAIL」的因果断言即违规。
- **D4 字面处方审查**：PLAN D4 处方含「grad_clip」半句，但当前默认已为 10.0 → 该半句为空操作，不构成变量变更。D4 字面唯一有效变量 = lr 0.1→0.05。

### 3.2 两次机会的预登记合法性
- **attempt-1**（当前运行）：D4 字面，唯一变量 lr 0.1→0.05。预注册处方干净；lr 杠杆有独立机制支持（loss 平台 11.18 随 lr 阶梯走，G2' 日志 ep1-19 lr=1e-1 平台、ep20+ lr=1e-2 续降）。
- **attempt-2**（结果前预登记）：clip 10.0→0.0，其余同 attempt-1。P3 假说被锁进第二次机会。
- **裁决**：两次 = 两个事先声明的不同杠杆（lr vs clip），非重复浪费。attempt-2 仅在 attempt-1 <92.0 时触发。
- **红线**：两次机会用尽仍 <92.0 → 选题停止（§10 文书开形式庭），不许换 seed、不许加第三次、不许中途改杠杆。

### 3.3 预算冲突
- F 桶（D4 专属）：封顶 8.18h / 实测 3.9h（attempt-1 ~1.95h + attempt-2 ~1.95h）。
- 对外口径 C+D ≤22.44h 不变。
- D-21 改挂起；D 桶借支驳回。

### 3.4 batch 缺口闭合
- 步数反推：1916 步/epoch = floor(490623/256) = 1916 → batch=256 注册配方坐实。
- 双独立验证（指挥官 + P4），缺口关闭。

---

## §10 形式庭触发条件（自动）
- D4 attempt-1 + attempt-2 均 <92.0 → 选题停止，自动开形式庭。
- 形式庭文书：停止原因、已穷尽变量、证据清单、可复现指令。

---

## §Z2 CFP-FP 教师 det+align 复跑（Z2 终审）
- [Z2-预测 2026-10-25 跑前冻结] CFP-FP 教师 det+align 复跑：acc_cv10 ≈ 95.5% ± 2.5（1σ）、AUC ≈ 0.97。
  命中 → Z2 闭案（两栈之差=输入管线差，CALFW+CFP-FP 双数据集实证）；
  脱靶 → 机制故事重写；本预测冻结后不改。

### §Z2 实测（2026-10-25 复跑后登记，三证）
- 复跑命令：C3 批教师命令原样 eval_generic_pairs.py --dataset cfp_fp --calib_onnx weights/models/buffalo_s/w600k_mbf.onnx，--manifest_dir 指 Temp 副本。
- 注册件 sha 跑前=跑后：cfp_fp_manifest.csv DFABBEA68CFACD429D686043CA05D6F8B5FF6DDF1F33720241F29B663C0297ED；cfp_fp_manifest.meta.json CE96D0363FC92FDD090AF676B6BDF671F7F55DAE4277213D7FD5863EB5E2D1E0（未污染）。
- 数据：唯一图集 5901 张全部可读；n_pairs=7000（same 3500 / diff 3500）；device=CPUExecutionProvider。
- 实测：acc_cv10 = 71.34%（acc_std=2.51%）；AUC = 0.7870；EER = 28.69%；TAR@1e-3 = 3.71%。
- 逐 fold acc：66.43 / 73.57 / 69.29 / 69.86 / 74.00 / 71.43 / 69.43 / 75.29 / 71.71 / 72.43。
- 原始 JSON：results/_z2_cfp_fp_teacher_20261025.json（config_hash=deea25e5d1da7b6284e471efc638fac34c52e40ba355febbf0ce20dbf12a6d76）。
- **对预测偏差（严重脱靶）**：acc_cv10 71.34% vs 预测 95.5%±2.5 → 偏 −24.16pp（≈ −9.7σ）；AUC 0.7870 vs 预测 0.97 → 偏 −0.183。
  按 Z2 冻结词：脱靶 → 机制故事重写；本预测冻结后不改。

### §P1b CALFW pairs 取证原始输出（2026-10-25，入档）
- rows=12000；avgLen=24.5697；fields=2 ×12000（每行 name+label 两字段）。
- 末列 label 直方图：0×6000，1×600，2×600，3×600，4×600，5×600，6×600，7×600，8×600，9×600，10×600。
- head8：Carl_Reiner_0001.jpg 1 / Carl_Reiner_0003.jpg 1 / Bernard_Law_0001.jpg 1 / Bernard_Law_0004.jpg 1 / Ian_Smith_0001.jpg 1 / Ian_Smith_0004.jpg 1 / Fred_Thompson_0001.jpg 1 / Fred_Thompson_0002.jpg 1。
- 599-602：Zorica_Radovic_0001.jpg 1 / Zorica_Radovic_0002.jpg 1 | Abdullah_Gul_0001.jpg 2 / Abdullah_Gul_0003.jpg 2。
- 1199-1202：Zarai_Toledo_0001.jpg 2 / Zarai_Toledo_0003.jpg 2 | Aaron_Sorkin_0001.jpg 3 / Aaron_Sorkin_0002.jpg 3。
- 2999-3002：Zurab_Tsereteli_0001.jpg 5 / Zurab_Tsereteli_0004.jpg 5 | Rodney_Dangerfield_0001.jpg 6 / Rodney_Dangerfield_0002.jpg 6。
- 5999-6002：Daniel_Osorno_0001.jpg 10 / Daniel_Osorno_0002.jpg 10 | Jeffrey_Archer_0002.jpg 0 / Luis_Ernesto_Derbez_Bautista_0003.jpg 0。
- tail：Alain_Ducasse_0002.jpg 0 / Pieter_Bouw_0002.jpg 0 / Iva_Majoli_0003.jpg 0 / Stacy_Dragila_0004.jpg 0。
  （本段为原始取证入档，不判读、不拟假设，判读归属待裁。）

### §C2-M3 Z2 判决实验 + A*/P1/罗塞塔/P3（2026-10-25 晚，三证）
- **A* 盘点**：List A（未完-A1 9 条 cfg3_seed456_ep1..8+final + 未完-A2 8 条 cfg0_seed456_ep1..7+final）= 17 件；17/17 exists=True；N（final 缺失数）= 0/2。
- **P1 机械验证**：段 A（行 1-6000）同人违反=0；段 B（行 6001-12000）异人违反=0 → PASS。
- **罗塞塔**（HF marcelohaps/calfw pairs.csv，经 hf-mirror 下载 1,192,129 B）：机械对齐 100% 一致（mism_name_a=0/mism_name_b=0/mism_is_same=0/mism_fold(segA)=0；HF fold_id 1..10×600、is_same 3000/3000）；唯一差异=段 B 末列编码（本地塌成 0、HF 保留折号）。
- **P3 缺口定位**：CALFW_sims 键带 `images/` 前缀（须去前缀）；去前缀后 local-not-in-sims=1 → 缺失对 = `Robert_Towne_0001.jpg | Robert_Towne_0002.jpg`（与 5,999 vs 6,000 吻合）；两文件 ca-aligned=True / extracted=False。只报不补。
- **Z2 判决实验**（20 对=FF10/FP10、正10 负10；预处理统一 (x-127.5)/128 BGR CHW；模型 w600k_mbf.onnx）：
  - **路径 A（原图→在线 det+align）全部 NA（20/20）**：真因=insightface 2.0 的 SCRFD 实现与 det_500m 输出结构不兼容（`_diag_scrfd_20261015.txt`：fmc=3/9 输出，batched=False→n@0.5=0 max=-1.000；batched=True→IndexError）；det_500m.onnx 与官方字节一致（sha 5e4447f5…）；属已知库缺陷（2026-10-15 已留档），**非 CFP 数据问题 → A 路径本环境不可用**。
  - **路径 B（cfp_aligned 现成作物）**：POS mean=0.2958 std=0.2597；NEG mean=0.1765 std=0.1687；gap=0.1193；20 对 AUC=0.6200（FF 对分离较好、FP 对差）。
  - **四出口判读**：A 不可测；B 糊（AUC 0.62）；P2c⑤ 手工探针（官方预处理直喂）亦糊（正负重叠）→ **两个独立证据同向指「糊」，倾向「A、B 双糊→0.787 重标为 w600k_mbf 在 CFP-FP 的真实水位」出口；但 A 未测，判读交用户裁定，本行不代裁**。
- **Z2 勘误（一行）**：Z2 预测冻结的 `AUC≈0.97` 实为 D-32 登记的 **CALFW** `CALFW_metrics.json roc_auc=0.9682488607313549`（非 CFP-FP 的 `CFPW_metrics.json roc_auc=0.9977`）→ **预测基线误挂 D-32 的 CALFW 值**，正确出处应为 CFP-FP 0.9977 或重设基线。
- **P4 派生发射：阻塞未发射**。① 无 rar 解码器（无 7-Zip/unrar；rarfile 未装；tar 不认 rar）→ 无法解压 `raw\calfw\images,landmarks\images.rar`（104MB）→ 无法构建 calfw_aligned；② 构建器所需 buffalo_s det 与 A 路径同一缺陷（insightface 2.0 SCRFD 不可用）。属环境缺陷，报阻塞不自行绕过。

### §C2-M3b 裁后微批次（2026-10-26，三证）
- **行1｜Z2 双路径**：法证确认模型未换（onnx sha 全绿 = 9cc6e4a7…319eb4f）→ 排除「模型被换」根因；A 路径（SCRFD）不可用（insightface 2.0 缺陷）已定；**A′ 亦阻塞**——CFP 官方 landmarks 实为 **30 点**（`Data/Fiducials/{id}/{frontal,profile}/NN.txt`，与 ArcFace 五点模板不匹配、映射不确定），fallback `insightface==0.7.3` pip 装失败（依赖 stringzilla py3.14 wheel 403）；故 A′ 未出数，B 现值为准（POS 0.2958±0.2597 / NEG 0.1765±0.1687 / AUC 0.6200）。
- **行2｜A* 盘点 N=0**：List A 17 件（未完-A1 9 + 未完-A2 8）逐件 Test-Path = 17/17 exists=True；N（final 缺失数）= 0/2。
- **行3｜watcher 闭案**：真身 = 计划任务 `\FaceRecWatcher` → `C:\watcher\watcher.ps1`（文件完整性守护，监控 4 文件 SHA/30s+心跳 10min）；PrimaryLog = `C:\watcher\PrimaryLog.csv`（7 BOOT/16 HEARTBEAT/1 TRANSITION）；上次运行 2026/10/25 13:07:08、结果 -1073741510(C000013A) 被终止。
- **行4｜P3 定位 + 补跑**：缺失对 = `Robert_Towne_0001.jpg|Robert_Towne_0002.jpg`（CALFW_sims 5,999 缺 1）；**已补跑** cos_sim = **0.419622**（ca-aligned 224→112 直取 + w600k_mbf，2026-10-26）→ CALFW_sims 复原 6,000 行。
- **附带｜P4 发射完成**：winget 装 7-Zip 26.05 → `raw\calfw\images.rar` 解压出 12,174 张（`raw\calfw\extracted\images\`）；`CA_landmarks.zip` 解压 12,174 个五点 txt；派生 `scripts\align_calfw_v1.py`（官方 landmarks 五点仿射→112×112 ArcFace 模板、不经 SCRFD）→ **`datasets\calfw_aligned\` 12,174 张 [selfcheck] PASS（elapsed 48s）**；pairs 引用覆盖率 **100%**（唯一引用 7,156 名全在、0 缺失）；`datasets\eval_manifests\calfw_manifest.csv`（6,000 行= same 3,000 + diff 3,000，sha256 `65879bcae5d26ae81799d934803118a686b2abc6b9e6ede901a46d7236c0186e`）+ `calfw_manifest.meta.json`。

### §C2-M3c 法证批次（2026-10-26 午后，三证）
- **1｜D-32 出处提取**：CFP-FP 校准三轮（V0 SCRFD 71.3429% FAIL / V1 fiducial 三点 94.8571% / V2 机制双改 95.5429% FAIL）脚本 = `prepare_benchmarks.py`（--mode fiducial）+ `eval_generic_pairs.py`；作物目录 = `datasets/eval_manifests_fid[fid2]/` + `cfp_aligned_fid[fid2]/`（sha cab9c8a8…/09013db6…）；CFPW_metrics 0.9977 与 CALFW_metrics 0.9682 均出自 `weights/models/buffalo_s/`（ref.json: extract=w600k_mbf、det=det_500m；CALFW images=7156、roc_auc=0.9682488607313549、sims 5,999 行）。
- **2｜时间线审计**：a) `datasets\cfp_aligned` 全 **7000** 文件 mtime 全落 **2026-10-15 10:21:47~10:44:03**（单一日期桶、后续零触碰）；b) `git log --since=2026-10-15 -- scripts` = `7f308ef`(10-16 prepare 注释 V1→V2 更正，git diff 2+/2−) + `5678390`(C2-M3b, 与 cfp 无关)；c) Z2 评估脚本 `eval_generic_pairs.py` mtime=**2026-10-15 13:10:26**（在 cfp_aligned 之后被改，8976ea0 = device 真值化 + fiducial V2，经 P4 追认为记录级零数值影响）；`prepare_benchmarks.py` mtime=2026-10-16 12:17:15（仅注释）；`_eval_calib_mirror_v1.py`/`_align_mirror_calib_v1.py` mtime=2026-10-17（未入库）。
- **3｜探针构成**：C2-M3 20 对 = **FF 10（5 正 5 负）+ FP 10（5 正 5 负）**，正 10 / 负 10。
- **4｜Path C（LFW 校准重放）→ 出口一（复现）**：命令 = `_eval_calib_mirror_v1.py --dataset lfw --batch 256 --out <Temp 新名>`（root 默认 `datasets\lfw_mirror_112`，7701 文件在盘）；重放实测 **acc=97.0833% / std=0.4787 / 10 折 97.1667…96.0000 / canary PASS ①100% ②49.625%**，与注册件 `lfw_mirror_calib_20261017_v2.json` **value/std/fold 逐位相同**（value 97.08333333333334）；注册件 sha 未变（只写 Temp）。**→ harness+模型+环境对 LFW 路径全部健康 → 故障锁定 CFP 局部（作物/manifest/CFP 分支代码）。**
- **5｜CFP manifest 审计（出口一满足后执行）**：`cfp_fp_manifest.csv` 7000 对逐对核——**FILE_MISSING=0**（存在性 100%）、**pair_not_fp=0**（7000/7000 全 frontal-profile = FP 形态）、**same=3500 / diff=3500**、fold 0-9 各 700。**零 mismatch** → manifest 无问题。
- **综合**：Harness 健康（Path C 逐位复现）+ manifest 零 mismatch → CFP 脱靶（71.34%）残余嫌疑收窄至 **cfp_aligned 作物质量** 或 **CFP-FP 跨姿态本质难度**（与 P2c④ 目检正常 + P2c⑤ 手工探针也糊 同向），判读交用户裁定。

### §C2-M3d 终局批次（2026-10-26 晚，三证）
- **1｜注册链四元组**：V2 注册件 `_t5_calibration_cfpfp_fid2_20261015.json`（acc_cv10=0.9554285714285715 / auc=0.9894531428571429 / eer=0.0442857142857143）——作物目录 `datasets\cfp_aligned_fid2`、manifest `datasets\eval_manifests_fid2\cfp_fp_manifest.csv`（sha256 `09013db6517de5552f06b7da2477304e1a21e4de4417dd2cde6c03d948d4b6d4`）、脚本 `scripts/eval_generic_pairs.py`、生产件 `_t5_prepare_cfpfp_fid2_20261015.json`（mode=fiducial、n=7000、src=raw\cfp\cfp-dataset）。**FAIL 判据 = 锚 CFP-FP 98.00±0.5（buffalo_s 行），对锚差 >0.5pp 即 FAIL**：V1 94.8571%（差 3.14pp）FAIL、V2 95.5429%（差 2.46pp）FAIL。cfp_aligned_fid2 / eval_manifests_fid2 均在盘=True。
- **2｜注册链重放 = 逐位一致 → Z2 结案**：`eval_generic_pairs.py --dataset cfp_fp --calib_onnx w600k_mbf.onnx --aligned_root datasets\cfp_aligned_fid2 --manifest_dir datasets\eval_manifests_fid2 --out results\_t5_calib_cfpfp_fid2_replay_20261026.json` → value=**0.9554285714285715** / auc=**0.9894531428571429** / eer=0.0442857142857143 / manifest_sha256=09013db6…b6d4，**全部与注册 V2 逐位相同**，exit=0。**Z2 脱靶双错定谳（皆文书性）**：① **复跑管线误选 V0**——Z2 终审复跑（`_z2_cfp_fp_teacher_20261025.json`）aligned_root=`datasets\cfp_aligned`（V0，SCRFD detect 作物，manifest sha dfabbea6…）→ 71.34%；注册链正确用 `cfp_aligned_fid2`（fiducial 30 点重建，sha 09013db6…）→ 95.5429%；② **预测基线误挂 CALFW**——冻结 AUC≈0.97 实为 CALFW 0.9682（非 CFP-FP）。
- **3｜Path B 20 对 FF/FP 拆分（V0 判向灾难签名坐实）**：FP 子集 pos mean=0.0860 / neg mean=0.1206 → **FP AUC=0.4400**（≈0.5，无区分力/反转）；FF 子集 pos mean=0.5055 / neg mean=0.2324 → **FF AUC=0.8400**；ALL AUC=0.6200。命中预测「FF 高 / FP≈0.5」，FF-pos 未深陷（0.5055）不改判。
- **4｜CALFW 新锚铸造**：教师在 `datasets\calfw_aligned`（112 五点仿射，P4 产物）重评（6000 对 = same 3000/diff 3000，w600k_mbf + (x-127.5)/128）→ **新 AUC=0.967338 / oracle acc=0.943833 @ thr=0.205819**（same mean 0.4357 / diff mean 0.0528）。**旧锚 D-32 CALFW_metrics roc_auc=0.9682488607313549 转历史注记（ca-aligned 时代产物，与新锚不可比）**。
- **5｜规格铁律入册（本案教益）**：**产物引用必须钉 sha/路径**——Z2 全案脱靶根因即「复跑复用了未钉路径的默认作物目录（cfp_aligned=V0）」，而正确件为 cfp_aligned_fid2（sha 09013db6…）；今后一切评测复跑须显式指定 aligned_root + manifest_dir 并核 sha，禁依赖默认值。

### §C2-M4 网格批次·线头收口（2026-10-26 夜，三证）
- **D-32 注记（0｜线头收口）**：`weights\models\buffalo_s\CFPW_metrics.json` roc_auc=**0.9977283809202676** 的**真实身份已定性** = **CFP-FP 的 FF+FP 合并口径**（`CFPW_sims.csv` 13914 行含 protocol 列：FF 7000 & AUC 0.9998781 ／ FP 6914 & AUC 0.9956252 → 合并 ALL 13914 AUC **0.9977284** 与 json 逐位相同；same 6960/diff 6954）。
  **CFP-FP 论文引用一律以重放实锤 AUC = 0.9894531 为准**（= 纯 FP 协议 7000 对，作物 `cfp_aligned_fid2` + manifest `eval_manifests_fid2`，sha 09013db6…；本轮重放 value/auc/eer 与注册 V2 逐位一致）；**0.9977 为「FF+FP 合并口径 roc_auc（13914 对，含 FF 易对拉高）」**，与纯 FP 口径不可直接比——故此前「预测基线误挂」实为**口径也混（合并 vs FP）+ 原挂 CALFW 值**双重错挂。
- **1｜网格预注册：阻塞（按纪律不自行构造清单）**：执行框要求「40 cell 清单 ＋ 超参 ＋ 判读规则 → 规格件 sha 落 ledger」，但**项目内无 40 cell 定义**——`results\prereg_v2_matrix.md`／`prereg_sigma.md`／`prereg_mirror.md`／`prereg_bench_ext.md` 与 `paper_plan` 全库检索均无「40 cell／网格规格」字样（仅见无关的 v6「12-cell 网格」）。依「有遗漏先问而非自补、不额外发挥」纪律，**网格清单待用户给出后方可预注册与发射**（铁律：先注册后运行、运行开始后规格零改动）。

### §C2-M9 清欠（2026-10-27，三证）
- **M4v3→M5 始末**：`ABLATION_CONFIGS` 仅 6 条（cfg0–cfg5），seed456 final 仅 3（cfg0/cfg3/cfg4）→ K≠10 停报；后续 M6 查明 seed456 三臂 final **全 epoch=5 未完**、seed42 臂**湮灭/崩坏** → 臂基础崩塌，转重建路线。
- **第七例（已知≠已执行）**：A* 盘点早已登记「epoch<10 未跑完、作终稿无效」，但核心层仍以 seed456 臂发射（G01c/G02/G03）→ **已知（A* 判定）≠ 已执行（未铸护栏门）** 的教训 → 促成第四钉。
- **第八例（负测试预期按门的规格范围写）**：v3.4 负测试「cfg3@42 必 FAIL」预期**不成立**——第五钉（magnitude）放行（med=6.78e-03 正常），实测 AUC 0.5122 → 后由**第六钉（功能门 AUC≥0.60）**补拦（exit 10）。教训：负测试预期应**按该钉的实际拦截范围**写，不可跨门预期。
- **cfg3@42 崩坏注记 + 30s 验尸**：`cfg3_seed42` epoch=30、med|w|=6.78e-03（数值正常）但 **CFP-FP 实测 acc 0.4994 / AUC 0.5122（近随机）** → **正负对嵌入 sims 分布重叠**（AUC≈0.5）→ 功能崩坏（非数值湮灭），第五钉漏拦、第六钉拦 ✓。
- **主检验封卷始末**：三钉普查 CFP-FP 最优 `cfg3`(0.8101) − `cfg0`(0.8029) = **+0.72pp < +1.0pp** → **主张终结、等级=趋势**；反翻案条款入册（不得因新臂/新协议重开）。
- **B2 第九例（口径差）**：CALFW 锚 `0.967338`（Mann-Whitney）vs eval 实测 `0.968887`（sklearn roc_curve）差 +0.0015；D1/D2 证**数据一致**（D2 Mann-Whitney 复算 0.967338 逐位、same 0.4357/diff 0.0528 逐位）→ **口径实现差坐实**；**新基准锚 = 0.968887（eval 口径）**，0.967338 标 legacy。
- **manifest 版本核查**：`calfw_manifest.csv` = **write_manifest 脚本重生成**（sha `fc2df448…`，LF）；独立名 `calfw_manifest_v2.csv`（sha `dd8aa594…`，CRLF）内容一致、差 6001≈6000 行尾；**旧 v1（sha 65879bca）已被同名覆盖**（不变量受损，记之）。
- **臂册补种子列 + 健康臂计数勘误 7→8**：健康臂（过双钉+功能门）= cfg0_final / cfg1_final / cfg2_final / cfg3_final / cfg3_seed123 / cfg4_final / cfg4_seed42_s20_l01（**变体臂**）/ cfg5_final = **8**（`cfg3_seed42` 崩坏剔除）；cfgN_final 的**种子归属未确**（默名）。
- **canary ② 数据独立性读码（遗留项清）**：`_eval_cfp_mirror_v1.py:114` `rng=np.random.RandomState(seed=20261017)`，② 用 `rng.normal`+`rng.permutation`（**同一独立 RandomState**，非复用外部常量）→ **数据独立性 OK**（② 实测 49.6250% 落 50%±2% 带内）。
- **A｜LFW 铸造 + 门一**：`_eval_calib_mirror_v1.py` 加 `--ckpt`（EdgeFace+`get_embedding[:, :512]`+flip-TTA 求和+四五六钉自查；教师 onnx 路径不动）→ **门一 A 教师逐位 97.0833（std 0.4787、fold 逐位同注册）✓**；注意该脚本因 `scripts/_*` gitignore **无法 commit**（照实）。
- **C/D 顺延**：当前 19:18 已过 18:00 GPU 截线 → 16 cell **顺延 10-28 早**；18:30 重训**未跑**（seed456 final mtime 仍 10-03/10-04）→ 待裁（现在补跑 or 10-28）。

### §C2-M10 git 整备清欠（2026-10-27 晚，三证）
- **第十例（生成器写点名未钉死 → 覆盖旧件）**：`eval_generic_pairs.py:write_manifest` 硬编码点名 `f"{dataset}_manifest.csv"` → CALFW 铸造时**同名覆盖**了 v1 manifest（原 sha `65879bca…`，产物 `calfw_manifest.csv` 现为 `fc2df448…`）。教训：**生成器写点名须钉死版本/加版本后缀**，否则重生成即覆盖既有注册件（不变量破坏）。考古：v1 文件本身**从未入库**（`datasets/` gitignored），`65879bca` 仅存于规格件文本引用（5 commit）；现仅凭规格文本+本台账追溯，**legacy 无 git 佐证**。
- **第十一例（簿记语言≠git 机制）**：多批簿记写「旧件标 legacy/不入库」，但其**文件或脚本实为 gitignored**（`CONFLICTS.md`=`/CONFLICTS.md`、`_eval_calib_mirror_v1.py`=`scripts/_*`、`calfw_manifest*`=`datasets/`）→ 语言上声明「标 legacy/不入库」与机制上「被 ignore 而根本不在版本控制」**不等价**；「不入库」的**实际含义**须显式区分「tracked 但标注」vs「untracked（ignore）」。教训：簿记结论必须与 git 机制一致（`git check-ignore -v` + `ls-files` 核实后写）。
- **今晚时序始末（2026-10-27）**：C2-M9 续批 A(LFW 铸造,门一 97.0833✓)/B(CALFW 铸造 18bee80)/E(CONFLICTS 清欠) 完成；C(16 cell) 因 **19:18 已过 18:00 GPU 截线**顺延 10-28 早；D(18:30 重训)**未跑**（seed456 final mtime 仍 10-03/10-04）→ 改 C2-M10 「0｜19:35 起跑 cfg3@456」补窗。git 仅剩 8 个 `?? results/G*_*.json`。

### §C2-M10 续·第十二例 + 启动三证（10-28 晚，三证）
- **第十二例（启动三证：pid + 日志首行 + 60s 存活）**：**Start-Process 在本沙箱不可靠**（C2-M4v3.2 与 C2-M10 均报「进程已在后台启动」但**无日志、无训练进程**）→ 改 **`nohup <python> ... > <log> 2>&1 &`** **成功脱离会话**。新纪律：**任何「脱离会话」长任务，落启动三证 = ① pid（`$!`）② 日志首行（如 `[ARM] ...`）③ ~60s 后进程存活 + 日志有内容**；三证不全 = 视为未启动、停报。
- **本次启动三证（cfg3@456 重训，2026-10-28）**：① pid=**2124**（nohup）② 日志首行 `[ARM] cfg3_seed456 w_feat=0.0 start=0 aug=False config_sha=d13e69c2beaecac8 commit=3383bc4` ③ 60s 存活 ALIVE + `[epoch 1/30]` 落盘。日志：`logs_multiseed/cfg3_seed456_retrain_20261028.txt`。
- **心电图（每 epoch 落盘，2026-10-28 17:49 已出 ep1-5）**：`[eval ep1] AUC=0.7064 Acc=64.13%` / ep2 0.8345/75.40 / ep3 0.8402/75.99 / ep4 0.8337/75.85 / ep5 0.8371/76.01；`[health] med|w|` = 1.761e-02→8.247e-03→3.924e-03→2.726e-03→**2.418e-03（逐 epoch 递减 ~7×，仍 >1e-8 但需警惕）**；AUC **ep2 起平台 0.83-0.84**。
- **进程 2124 训练中（epoch 6/30 @17:49）**；`epoch≈7.7min → 30ep≈3.8h → 预计 ~20:50 完`（`--eval_every 1` 开销）。
