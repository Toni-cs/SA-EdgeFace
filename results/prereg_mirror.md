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

## §7 修订 A1（2026-10-17 12:45 +08:00 追加；带时间戳 amendment，§1-§6 冻结字面不动）

> 授权依据：用户 2026-10-17 平台指令（问答照录，CONFLICTS D-53）：「检测收工后的三个动作（按序）① 漏检统计落册 ② 嵌入+计分，顺序别乱：先 LFW / AgeDB 镜像忠实性校准（授权包条款，LFW 余量 0.0167pp 必须在统一栈下重验——这是镜像是否忠实的裁判）→ 再 CFP-FP → 再 TinyFace ③ 出数即落册，按 §4 三分支判据对号」「同时盯住 LFW：统一栈下 LFW 若跌破 99.70−0.5 下沿，镜像忠实性有问题，回滚排查移植代码——这个判读优先级高于一切 TinyFace 解读」。

1. **范围增补（用户明示授权，§1「两 PASS 腿不重跑」就此两腿开例外）**：新增两条**诊断腿**（LFW 镜像忠实性校准、AgeDB-30 镜像忠实性校准）。两 PASS 腿的既有登记事实（LFW 99.2167 / AgeDB 96.5667）**不动、不重判、不覆盖**；诊断腿产出 = 新名 JSON，只回答「统一镜像栈是否忠实」，不产生任何 PASS/FAIL 重判。
2. **诊断腿口径（事前写死）**：
   - 数据 = LFW 协议引用唯一图 7,701（datasets/pairs.txt，sha256 `ea42330c…`，6,000 对 fold-major 300same+300diff×10 已核；raw 全库 13,233，未引用者不碰，CFP 5,901 引用集原则同例）；AgeDB 协议引用唯一图 12,000（agedb_30_index_pairs.txt，6,000 对折聚集 600×10 已核，= variantA 全树）。
   - 对齐 = §3 C 终值同一实现（`_align_mirror_calib_v1.py`，process_one 直接 import 自冻结件 `_align_mirror_v1.py` sha `6b809d6f…`，MTCNN 参数逐字复写）。
   - 模型 = w600k_mbf.onnx（sha256 `9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f`，LFW/AgeDB 臂 1 同件同后端）。
   - 评测 = 统一镜像栈（flip-TTA 求和 + sklearn L2 + 官方 LFold 十折阈值折内择优，`calculate_roc` 自 `_eval_cfp_mirror_v1.py` 同源 import 零复制）。
   - **判据（事前写死）**：LFW 校准值 ≥ 99.20（= 锚 99.70−0.5）→ 栈忠实；< 99.20 → **镜像忠实性 FAIL** → 回滚排查移植代码，该判读优先级高于一切 TinyFace 解读（用户指令字面）。AgeDB 校准值 ≥ 96.08（= 锚 96.58−0.5）→ 同判。
   - 照实披露：臂 1 的 LFW/AgeDB 评测栈无 flip-TTA、无 L2 行归一；诊断腿为统一镜像栈，校准值与臂 1 值之差异同时含对齐+协议两变量；判据只对锚带、不对臂 1。
3. **执行顺序（用户指令字面，覆盖 §4 隐含的两腿并列）**：漏检统计落册 → LFW 诊断腿 → AgeDB 诊断腿 → CFP-FP 腿 → TinyFace 腿。
4. **配套实现件跑前纠错（0 张产出、判据零动，照 D-52 先例）**：`_eval_cfp_mirror_v1.py` v1 草稿误置 AdaFace IR-101 为评测模型——与 CFP 锚 98.00（AMD-V2-2 buffalo_s 行）及臂 1 实值 95.5429（w600k_mbf.onnx）不同件，判据失义；开跑前审出，纠正为 w600k_mbf.onnx + egp._preprocess（≡臂 1 栈）。§4 判据字面零动。新 sha 登记 D-53。

## §8 修订 A2（2026-10-17 18:41 +08:00 追加；带时间戳 amendment，§1-A1 冻结字面不动）

> 授权依据：用户 2026-10-17 平台指令（问答照录，CONFLICTS D-55）：「v2 重跑授权（附 D-55）：1. 定性：48.93% 为仪器病灶产物（配对全灭→随机水平，定量吻合），非镜像栈真实测量。v2 = 新候选（三新 sha），按 A2 冻结后跑；红线『新预注册+用户重新授权』两项均满足，本条即授权。2. v1 件与 FIDELITY_FAIL(v1) 原样留册不覆盖（已照拒覆盖红线执行）。3. 一锤封口：v2 是唯一重跑。v2 再 FAIL = 真实保真失败，照册收口，不进入第三次仪器修复循环（须回 user+P4）。4. 门限不动：LFW 99.70±0.5 锚带照旧。」；金丝雀条款（同指令）：「evaluate() 跑分前 30 秒预检：金丝雀两测：① 全同嵌入喂入 → 必须精确 100%（配对正确性的充分证明）② 随机嵌入喂入 → 必须落在 50%±2%（判别力健全）任一不过 → 拒绝出数，仪器回修」。

1. **v1 事实照录**：LFW 校准腿 v1（2026-10-17 13:15:09）出数 48.9333%（十折 48.00-51.00，恰随机水平），按 §7-A1 字面判 **FIDELITY_FAIL(v1)**；件 = `results/lfw_mirror_calib_20261017_v1.json` 拒覆盖原样留册（commit `861b4c3` 入库）。定性：**仪器病灶产物、非镜像栈真实测量值**。
2. **病灶定谳（A1 预案「回滚排查移植代码」执行完毕；官方源码 + 静态证明）**：
   - 官方铁证：CVLface@308142aa `verification.py:188-192` evaluate() 对整块 embeddings 取 `embeddings[0::2]`/`[1::2]`——只对交错布局（行 2i=第一图、行 2i+1=第二图）成立；`verification_evaluator.py:104` `collection['is_same'].numpy()[::2]`（按图存标签取步长 2）佐证交错。
   - 病灶：三件移植评测用 `np.concatenate([e1,e2],axis=0)` 造**堆叠**布局再套同一习语 → 6,000 对配对全灭 → 必然随机水平，与观测 48.9333% 定量吻合。
   - numpy 静态证明：堆叠布局下 e1n[0]=normalize(e1[0]) 而 e2n[0]=normalize(e1[1])（配对错位）；交错布局修复后 e1n==normalize(e1)、e2n==normalize(e2) 逐行 True。
   - 波及面权威全扫（Python os.walk，无视 gitignore 对 `_` 前缀件的遮蔽）：`def calculate_roc` 全仓唯一定义 = `_eval_cfp_mirror_v1.py`（校准件同源 import 零复制，**无第四副本**）；`[0::2]` 习语仅两 mirror 件。三件同源病灶：校准件（已跑出假数）、CFP 件（未跑、0 产出截获）、TinyFace 件（另处：函数返回 (3,) 被调用方误绑 `hits` 再 `.mean(axis=0)` → 三 rank 塌缩标量 → `zip` 必崩白烧 GPU——未跑、截获）。
3. **仪器修复收口（判据/协议/模型/锚/门限零动；仅修配对布局与返回值绑定 + 金丝雀增强）**。冻结 sha（三级链：D-53 跑前纠错态 → D-55 配对修复态 `fc810a32…`/`ffa328bd…`/`4eede864…` → A2 终版）：
   - `_eval_calib_mirror_v1.py` = `c2678196c2700f4204c359c2fac9531abd9b08ec3fc52e5275c9e8720bce419e`
   - `_eval_cfp_mirror_v1.py` = `2492ab776381996d6a6a70e3c0eb271acc303771765638e46e0f7c9fc01076e9`
   - `_eval_tinyface_mirror_v1.py` = `9dc1dea8c9e7874c2f5584e1615e5814c6081a2ba2118eb6e7d8be0b797da6d3`
   - `_run_canary_selftest.py`（金丝雀 runner）= `e2154d98365cf3a2e0ae52bc769a46166fcf4d1488a59a1a0ff90ee672c24d87`
   - 切分唯一实现 `pair_split_normalize()` 定义于 `_eval_cfp_mirror_v1.py`，校准件同源 import（零复制）；runner 运行时断言两模块函数为同一对象（同源自证）。对齐器 `_align_mirror_calib_v1.py` `b9237526…` 未动。
4. **金丝雀条款（用户建议采纳；操作化判据事前写死）**：
   - pairwise（校准腿 + CFP 腿共用同一函数）：① 配对正确性充分证明——same 对喂同一向量、diff 对喂反向向量、标签折聚集各半（n_pairs=4000, dim=64, seed=20261017；L2 归一后 same dist=0 / diff dist=4，官方阈值网格必有完美分离阈值）→ 必须**精确 100.000000%**。操作化披露：条款字面「全同嵌入」若实现为全 same 标签，折内择优阈值使「全判 same」恒得 100%、不构成配对正确性的充分证明（恰为本次病灶所不能测出者）；故按条款意图操作化为 same/diff 各半——错乱配对必落 ~50% 被拦截。② 判别力健全——独立随机嵌入 + 随机各半标签 → 必须 **50%±2%**。
   - closedset（TinyFace 腿）：① mate 恒同嵌入 → rank-1/5/20 必须精确 **[100,100,100]** 且 shape=(3,)（塌缩类笔误即刻暴露）；② 随机嵌入 → rank-1 必须 **< 5%**。
   - 执行位置：三件 main() 起点、拒覆盖检查之后、模型加载/嵌入之前（30 秒内失败即 exit 7 拒出数，不烧任何重活）；结果记入 payload `canary` 字段。
   - 冻结前开发自测（2026-10-17 18:40 +08:00）：两测全过——pairwise ①=100.000000% ②=49.6250%；closedset ①=[100,100,100]@shape(3,) ②=rank1=1.5625%。冻结后开跑前再跑一次出证据件（`results/_canary_mirror_v1_20261017.txt`）。
5. **一锤封口（用户授权字面，反滑坡条款）**：v2 是唯一重跑。v2 再 FAIL = 真实保真失败，照册收口，不进入第三次仪器修复循环，转 user+P4。执行细则：校准腿（LFW/AgeDB）任一再 FAIL → 即刻收口，CFP/TinyFace 两腿不再出数。
6. **门限照旧（用户授权字面）**：LFW 99.70±0.5（gate_lo 99.20）、AgeDB 96.58±0.5（gate_lo 96.08）、CFP-FP |mirror−98.00|≤0.5pp、TinyFace §4 三分支——全部不动。
7. **v2 执行与命名（事前写死）**：顺序照 §7-A1 = LFW → AgeDB → CFP-FP → TinyFace，出数即按 §4/A1 机械落册不现场解读。输出名：`results/lfw_mirror_calib_20261017_v2.json`（v1 名已被病灶件占用，拒覆盖强制）；AgeDB/CFP/TinyFace 三腿用各自 v1 名（旧仪器从未对三腿计分、v1 名未被占用，首次产出用 v1 名照实；仪器身份以 payload 内 `model_sha256` + `eval_script` sha 为准，文件名版本 = 该腿已完成计分次数）。
