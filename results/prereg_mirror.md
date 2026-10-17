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
- **A 中断兑现（2026-10-17，D-49，问答照录「现在就切 C」）**：Google 系断（docs.google.com / drive 三域 TCP443=False、HTTP 超时）+ 机器零代理三证 → A 不设时限挂死，按 D-48 预置条款切 C；切换时戳 10:32:49+08 先于冻结。C 兑现 = 阶梯甲（torchvision 修复 → facenet-pytorch MTCNN 跑通，§3 检测器行实填）；乙丙不启用。冻结后 A 若再可得不回切（跑后不换法）。

## §3 口径冻结五字段（冻结 commit 前逐项实填 pin，空值不得开跑）

| 字段 | 冻结值（pin 时实填） |
|---|---|
| 检测器（型号 / 权重 sha256） | **C 终值（2026-10-17 冻结前实填；A 路已中断 = D-49）**：阶梯甲兑现件 = facenet-pytorch 2.6.0 内置 MTCNN 三权重——pnet 28,570 B `a2a71925e0b9996a42f63e47efc1ca19043e69558b5c523b978d611dfae49c8f` / rnet 403,147 B `bbb937de72efc9ef83b186c49f5f558467a1d7e3453a8ece0d71a886633f6a86` / onet 1,559,269 B `165bfbe42940416ccfb977545cf0e976d5bf321f67083ae2aaaa5c764280118d`（首登记，上游无文件级公布值；比对链 = wheel `facenet_pytorch-2.6.0-py3-none-any.whl` sha256 `ecb82b27beb226d106f2219efe8f829b01b87a8595badd01545679bdb9f19cca` PyPI JSON / 阿里镜像页 / 本地下载三方一致 2026-10-17 → 包内权重随件 pin）。运行参数（冻结 pin）：`min_face_size=5`、`thresholds=[0.6,0.7,0.7]`、`factor=0.709`、`keep_all=False`、多脸取置信度最高框、device cuda。msf=5 选定依据 = 500 张工程探针（`results/_probe500_cvlmirror_20261017.txt`：msf=20 于短边<20px 图（占样本 62.6%）金字塔空表必崩 313/500，msf=5 仅崩 4/500 且检出 165/500 > 127/500，44 img/s 外推检测段 ~61 min），选定先于评测、与评测结果无涉。前置修复：torchvision 0.27.1+cu126（与 torch 2.11.0+cu128 错配致 `torchvision::nms` 缺失）→ 0.26.0+cu128 官方源，NMS_OK。源码半 pin 照旧（repo commit `308142aa…` + zip sha `f76b0a75…` + 取证件 `results/_cvlface_pin_20261017.txt`） |
| 对齐模板（裁剪变换 / 模板点集） | **C 终值**：MTCNN 5 点 → Umeyama 闭式相似变换（含各向同性缩放、无反射）→ ArcFace 112 标准五点模板（38.2946,51.6963 / 73.5318,51.5014 / 56.0252,71.7366 / 41.5493,92.3655 / 70.7299,92.2041）以模板中心 (56.0262,71.9008) 整体收缩 ×(1/1.1) 后作为拟合目标 = pad_0.1 外扩语义操作化（人脸占幅 ÷1.1、四周留余量；名证 = `tinyface_aligned_pad_0.1` 目录名；官方实现未载 = 自定披露「官方未载、非照搬」）。评测侧照旧已 pin：eval 无模板变换，对齐产物直通（`tinyface_evaluator.py:35`） |
| 插值方式与位深 | **C 终值**：对齐 warp = `cv2.warpAffine(INTER_LINEAR + BORDER_CONSTANT 0)`；漏检回退 = PIL `BILINEAR` 整图 resize；产物 = 112×112 RGB uint8 **JPEG q95**（臂 1 tinyface_112 存储形态同族，自定披露）。eval 侧照旧已 pin：`ToTensor + Normalize(0.5)`（≡(x−127.5)/127.5，与 `egp._preprocess` 数值同式）、零 resize、RGB 8bit |
| 输入尺寸来源 | **已 pin（A/C 同此）：112×112 预对齐产物直入，eval 不再缩放**（官方 `apps/face_alignment/README.md:4,31` + 评测代码零 resize）。尺寸决定权在数据集侧；C 路对齐输出即 112×112 |
| 漏检策略 | **C 终值**：无框 / 无有效 5 点 / 检测异常（含短边<5px 图的金字塔空表 ValueError）→ PIL BILINEAR 整图 resize 112×112 回退（不裁剪、不丢弃）；检出/回退计数逐腿落册。**计数恒等硬断言**：tinyface 3,728+4,443+153,428=161,599、cfp_fp = 官方 FP 协议 5,000 对引用唯一图集 5,901（对齐器 build_tasks 前置 assert；对齐范围 = 协议消费集，不碰全库 7,000 中协议未引用者）。官方未载、非照搬（§2 结构披露照旧） |

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
5. 检测实现强制件：checkpoint 机制为开检测**前置条件**——授权包条款⑤「断点续跑设计」的实现细则（2026-10-17 冻结前与实现一致化：对齐产物 = 按 rel 路径落盘的 112×112 JPEG 文件树，非 .npy 分片）；细则 = 输出按文件原子生成 + **目标已存在即跳过（= 断点续跑语义，断电任意时刻可重启续跑）** + done_list 每 1,000 张追加 + 终局计数自检（tinyface 161,599 / cfp 协议引用集 5,901 硬断言）。无 checkpoint 不开检测。
6. 窗口匹配：开跑前以 500 张实测吞吐外推总时长（外部规划估计 6–10 h **仅作参考、不作事实**）；预计单日窗装不下则分片跨日、逐日续。CFP-FP 检测（约 7K 张）独立落盘，排 TinyFace 检测之后用余窗收掉，两份结果互不阻塞。
7. 当日收工判据（收工时以事实记入当日日报，判据本身以本节随冻结生效）：① 冻结 commit 时间戳**先于**检测启动；② 无论进度几何，done_list 与分片件已在盘。

## §6 文献论证（检索 2026-10-16）

- **原句一（主证，同源已验）**：arXiv:1811.08965v2 原文载「the CentreFace model yields 32.1% Rank-1 on TinyFace versus 65.2% on MegaFace, i.e. a 50+% performance degradation」——https://arxiv.org/html/1811.08965v2 （检索 2026-10-16）。同篇即 D-38⑤ 逐字核验件（表 3 数真实），引文保真零额外成本。
- **原句二（辅助，实验室仓库 README 非期刊文献）**：idiap 仓 README 载「TinyFace is reported under two alignment pipelines, and the choice moves absolute accuracy by a large margin」——https://github.com/idiap/Synthetic-Data-Generation-for-Low-Resolution-Face-Recognition （检索 2026-10-16；github 直连不通，取证走 ghfast 镜像）。
- 结论定位：对齐器选择系 TinyFace 的**文献已知敏感性变量**；本件协议假设 = 该已知现象在本项目的实例化，非我方事后推测。
