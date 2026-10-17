# prereg_mirror.md —「CVLFace 官方管线镜像 · DG-R0a 两 FAIL 腿重校准」预注册（起草稿 v0 · 未冻结）

> 状态：**起草稿 v0，待冻结 commit 生效**（照 prereg_v2_matrix 先例：冻结 commit 之前不构成判据约束；先冻结后开跑，先跑后冻 = 违规）。
> 授权链：用户 2026-10-16 平台签署合并授权包五条款（问答原文照录 = CONFLICTS D-45①）；D-45 = 本件唯一授权依据。
> 关系条款：本件**不 amendment 任何既有件**（prereg_sigma / prereg_v2_matrix 各自 append-only，互不触碰）；只约束 DG-R0a 两条 FAIL 腿（CFP-FP、TinyFace）的管线重校准。

## §0 身份条款

| 项 | 值 |
|---|---|
| 文件名/路径 | `face_recognition_research\results\prereg_mirror.md` |
| 起草时间戳 | 2026-10-16（起草稿 v0） |
| 冻结 commit 哈希 | 冻结时登记（父提交 + 提交信息定位，不自引用） |
| 首臂开跑时间戳 | 开跑时登记 |
| 授权依据 | CONFLICTS D-45①（用户 2026-10-16 平台三答之合并授权包五条款） |
| 旧件对照件 | `results/prereg_v2_matrix.md`（四假设族与 §13/§16.1 约束照旧，本件不改其族） |

## §1 范围与不动项（冻结）

1. 范围 = **仅 DG-R0a 两条 FAIL 腿**重校准：CFP-FP（V2 实值 95.5429%，对锚 −2.4571pp）、TinyFace（直采实值 44.4474%，对判键 −27.9726pp）。
2. 不动项：
   - **锚维持 AMD-V2-2 注册值**：LFW 99.70 / CFP-FP 98.00 / AgeDB-30 96.58（buffalo_s 行，双源确认件）；TinyFace 判键 R1 72.42±0.5（CVLFace 板值行，D-38⑥）。锚不换、带宽 ±0.5pp 不变。
   - **LFW/AgeDB 两 PASS 腿不重跑**（已登记事实，无因重跑 = 无谓暴露）。
   - **学生侧零接触**（prereg_sigma §7 / v2_matrix §13 门2 照守：本件全部产出 = 已发表权重管线校准；两腿重校准结果一律入账、差也不剔）。
   - 既有判据字面一字不改；本件判据只增不改（§4 为新臂新判）。

## §2 授权边界（程序定性照实披露）

- 程序定性：镜像在程序上仍属**对齐栈第三次变更**；「向参照协议收敛」为申请方定性，本件照录不自裁其性质。
- **停止规则（仅此一锤）**：CFP-FP 镜像重校准对锚残差仍 >0.5pp 即停，转用户 + P4 裁量；不产生第四次对齐尝试、不迭代、不按结果调阈值。
- **检测器选型阶梯**（按序取首个可跑通者；选择规则先于结果，**不得按结果挑选**）：
  甲、修复 torchvision 安装后用 CVLFace 官方 eval 同款 MTCNN（镜像首选）；
  乙、纯 torch MTCNN 移植（同权重、同前处理参数）；
  丙、insightface SCRFD + arcface 模板对齐（末选；须在落册中披露为对板值管线的偏离）。
- **注（2026-10-17 源码取证，阶梯字面不动、仅照实披露）**：甲「同款 MTCNN」的「同款」在官方源码中**不可核**——全仓 MTCNN 命中=1（且为无关路径串）、eval 数据系预构建工具包而构建法未载、官方随仓对齐 app 用的是 DFA（`minchul/cvlface_DFA_mobilenet`）；toolkit 目录名 `tinyface_aligned_pad_0.1` 构成 pad 0.1 名证。阶梯三档照签执行，「同款」定性降为「按阶梯程序重建的近似」，证据 = `results/_cvlface_pin_20261017.txt`。**路线级三选**（A 官方工具包产物 / B 官方 DFA 程序复跑 / C 本阶梯重建）属条款级选择，待用户裁决；裁决落定前 §3 相关字段不完成、不开跑。**裁决已落（2026-10-17，D-48，问答照录）**：「A 并行 + C 兜底」（B 不走）——A 到货按 A 实填 §3；A 中断超窗口切 C（切换必在冻结 commit 之前并记 D-48+时间戳）。两路样本集恒等约束（probe 3,728 / gallery 157,871）通用。

## §3 口径冻结五字段（冻结 commit 前逐项实填 pin，空值不得开跑）

| 字段 | 冻结值（pin 时实填） |
|---|---|
| 检测器（型号 / 权重 sha256） | **源码半已 pin**：repo `https://github.com/mk-minchul/CVLface`，commit `308142aa50adf2e187711354f7524635d3414f1e`，zip sha256 `f76b0a752684e66c07f10cb1080dd619d9660b64b8dcc85276a6da06ab2f2d9d`，渠道 ghfast 镜像，取证 2026-10-17，取证件 `results/_cvlface_pin_20261017.txt`（源码 commit 与权重 hash 双钉规则不变）。**权重/检测器半（路线 D-48 = A 并行 + C 兜底）**：A 到货 = N/A（官方预对齐产物直用，工具包 zip sha256 作权重级 pin + 来源比对附卷）；A 中断切 C = 按阶梯甲乙丙 pin 对应权重 + 来源页比对后落值，缺一不得冻结 |
| 对齐模板（裁剪变换 / 模板点集） | 评测侧已 pin：**eval 阶段无模板变换**，对齐产物即数据集输入直通（`tinyface_evaluator.py:35` + transform 无几何项）。生成侧（D-48 裁决下）：A 到货 = N/A（官方产物）；切 C = MTCNN 5 点模板 + pad 0.1 外扩（名证：`tinyface_aligned_pad_0.1` 目录名） |
| 插值方式与位深 | 评测侧已 pin：eval transform = `ToTensor + Normalize(0.5,0.5,0.5 / 0.5,0.5,0.5)`，**无 resize、无重采样**（`models/iresnet/__init__.py:51-56`、`pipelines/infer_model_pipeline.py:26-27`）；色彩 RGB 8bit（full.yaml `color_space: RGB`）。生成侧插值：A = 官方产物整体 pin（其生成插值不可拆分即随卷登记）；切 C = 自选插值落值 + 披露 |
| 输入尺寸来源 | **已 pin：112×112 预对齐产物直入，eval 不再缩放**（官方 `apps/face_alignment/README.md:4,31` 载 resize 112x112；评测代码零 resize = 本行证据）。尺寸决定权在数据集侧，A/C 两路同此 |
| 漏检策略 | **结构披露已立（官方对漏检处置无明文）**；D-48 裁决下实填：A = N/A（官方产物含其漏检既成事实，样本集直接沿用）；切 C = 自定处置 + 落册披露「官方未载、非照搬」。**恒等约束不变**：样本集不得因漏检变动，probe 3,728 / gallery 157,871 计数恒等 |

- 镜像源码 pin：**已落**（2026-10-17，上表源码半 + 取证件 `results/_cvlface_pin_20261017.txt`，含协议七项行级证据、网络实测、HF 探测、原始数据在位清单）。
- 数据沿用 D-40 清单与 manifest（sha 84a64a9d…），不重切、不重建。

## §4 判据（事前写死，三分支，无第四支）

- 臂 1 = resize 直采实测 **44.4474%**（D-44 既有件，sha a658343d…，不重跑）。
- 臂 2 = 镜像对齐协议实测（本件产出，新名拒覆盖）。

**TinyFace 腿：**

| 分支 | 条件（互斥穷尽） | 后果 |
|---|---|---|
| ① | 臂2 − 臂1 ≥ 15pp **且** 臂2 ∈ [71.92, 72.92] | 协议口径错位假设成立；原 FAIL 归因口径，按对齐口径重判（判键带仍 ±0.5pp，不放宽） |
| ② | 臂2 − 臂1 ≥ 5pp 且不满足分支① | 协议为主因之一、模型低分辨率缺口并存；TinyFace 判键维持 FAIL，叙事吸收该发现 |
| ③ | 臂2 − 臂1 < 5pp | 协议假设不成立，低分辨率能力缺口确认；44.45% 口径下方可引用，判键带判定照机械执行 |

**CFP-FP 腿（镜像重校准单臂）：**

- |镜像值 − 98.00| ≤ 0.5pp → PASS（两腿齐 = DG-R0a 重估入口）；
- 残差 >0.5pp → 触发 §2 停止规则，转用户 + P4 裁量。

判后纪律：三分支机械执行；不换法、不改门、不二次重跑；落册按分支字面。

## §5 执行纪律

1. 顺序：源码 pin → §3 字段实填 → **冻结 commit（生效）→ 才开跑**。
2. 检测段：日内窗设计（用户侧 22:00 断电，**过夜不可行**）+ 断点续跑（预处理可续，判门不可断）；判门段（GPU 嵌入 + 计分约 20 min）单发连续，留足窗内余量。
3. 每件新名拒覆盖（`_mirror_v1/v2…`），证据件 sha 全留。
4. 分支边界与判键在数值产出前零改动；任何改字 = 新 amendment 带时间戳先登记后执行。
5. 检测实现强制件：checkpoint 机制（done_list 每 1,000 张追加 + 分片 .npy 落盘 + 断点续跑自检）为开检测**前置条件**——授权包条款⑤「断点续跑设计」的实现细则；断电不挑时间，无 checkpoint 不开检测。
6. 窗口匹配：开跑前以 500 张实测吞吐外推总时长（外部规划估计 6–10 h **仅作参考、不作事实**）；预计单日窗装不下则分片跨日、逐日续。CFP-FP 检测（约 7K 张）独立落盘，排 TinyFace 检测之后用余窗收掉，两份结果互不阻塞。
7. 当日收工判据（收工时以事实记入当日日报，判据本身以本节随冻结生效）：① 冻结 commit 时间戳**先于**检测启动；② 无论进度几何，done_list 与分片件已在盘。

## §6 文献论证（检索 2026-10-16）

- **原句一（主证，同源已验）**：arXiv:1811.08965v2 原文载「the CentreFace model yields 32.1% Rank-1 on TinyFace versus 65.2% on MegaFace, i.e. a 50+% performance degradation」——https://arxiv.org/html/1811.08965v2 （检索 2026-10-16）。同篇即 D-38⑤ 逐字核验件（表 3 数真实），引文保真零额外成本。
- **原句二（辅助，实验室仓库 README 非期刊文献）**：idiap 仓 README 载「TinyFace is reported under two alignment pipelines, and the choice moves absolute accuracy by a large margin」——https://github.com/idiap/Synthetic-Data-Generation-for-Low-Resolution-Face-Recognition （检索 2026-10-16；github 直连不通，取证走 ghfast 镜像）。
- 结论定位：对齐器选择系 TinyFace 的**文献已知敏感性变量**；本件协议假设 = 该已知现象在本项目的实例化，非我方事后推测。
