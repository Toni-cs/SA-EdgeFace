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

## §3 口径冻结五字段（冻结 commit 前逐项实填 pin，空值不得开跑）

| 字段 | 冻结值（pin 时实填） |
|---|---|
| 检测器（型号 / 权重 sha256） | 待 pin：CVLFace 源码 commit hash =＿；检测权重 sha256 =＿ |
| 对齐模板（裁剪变换 / 模板点集） | 待 pin：＿ |
| 插值方式与位深 | 待 pin：＿（CVLFace eval 同款逐字） |
| 输入尺寸来源 | 待 pin：＿（检测框 → resize 路径写死） |
| 漏检策略 | 待 pin：＿（CVLFace 官方漏检处置**逐字照搬**；不得自设 fallback——样本集不得因漏检变动，probe 3,728 / gallery 157,871 计数恒等） |

- 镜像源码 pin：repo URL + commit hash + 取证渠道（ghfast 镜像）+ sha256，取证件留盘 `results/`。
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

## §6 文献论证（检索 2026-10-16）

- **原句一（主证，同源已验）**：arXiv:1811.08965v2 原文载「the CentreFace model yields 32.1% Rank-1 on TinyFace versus 65.2% on MegaFace, i.e. a 50+% performance degradation」——https://arxiv.org/html/1811.08965v2 （检索 2026-10-16）。同篇即 D-38⑤ 逐字核验件（表 3 数真实），引文保真零额外成本。
- **原句二（辅助，实验室仓库 README 非期刊文献）**：idiap 仓 README 载「TinyFace is reported under two alignment pipelines, and the choice moves absolute accuracy by a large margin」——https://github.com/idiap/Synthetic-Data-Generation-for-Low-Resolution-Face-Recognition （检索 2026-10-16；github 直连不通，取证走 ghfast 镜像）。
- 结论定位：对齐器选择系 TinyFace 的**文献已知敏感性变量**；本件协议假设 = 该已知现象在本项目的实例化，非我方事后推测。
