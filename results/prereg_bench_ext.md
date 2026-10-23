# prereg_bench_ext - 外部基准补注册（CFP-FP 起）

- 建立：2026-10-23，批次 **C2a**
- 载体裁定：本文件为**新建**（C2a ① 批），与既有四件 prereg 并列：
  `prereg_efficiency.md` / `prereg_mirror.md` / `prereg_sigma.md` / `prereg_v2_matrix.md`
- 性质：**注册（prereg）**。仅登记口径、溯源与清点；**本文件不产生、不引用任何新数值结果**。
- 行尾：纯 LF、无 BOM（沿用 `prereg_v2_matrix.md` 之干净范式）。
- 写盘三律：先算后写、临时件、读回验证；本文件为**新建件**，不涉历史回滚。

## 0. 门禁锚（机器口径，取代不可得的 LFW 对照）

C2a 原定门禁为「LFW manifest 的 `aligned_root` 实测一致性」。**实测该锚不存在**，按批复改用机器口径表。

- LFW manifest 数量 = **0**；6 件 manifest 覆盖 CFP-FP x3 / AgeDB-30 / TinyFace x2，**无 LFW**。
- LFW 盘上目录 4 个：`lfw` / `lfw_aligned` / `lfw_aligned_stale` / `lfw_mirror_112`，均无 manifest。

替代锚（`scripts/eval_generic_pairs.py`，机器生成口径）：

    DEFAULT_ALIGNED = {
        "cfp_fp":   "datasets/cfp_aligned",
        "agedb30":  "datasets/agedb30_aligned",
        "tinyface": "datasets/tinyface_112",
    }

逐项在盘核验：

| key | 默认 aligned_root | 在盘 | 图片数 |
|---|---|---|---|
| `cfp_fp` | `datasets/cfp_aligned` | **是** | **7,000** |
| `agedb30` | `datasets/agedb30_aligned` | **否** | 0（见 §4-1 陷阱） |
| `tinyface` | `datasets/tinyface_112` | 是 | 161,599 |

**门禁结论**：机器口径 `cfp_fp = datasets/cfp_aligned` 与 C2a 裁定 `aligned_root = cfp_aligned` **逐字一致**，且该目录实测 7,000 张在盘 → **本节定稿**。

## 1. CFP-FP 基准注册

### 1.1 注册口径

| 项 | 值 |
|---|---|
| 基准名 | **CFP-FP** |
| 协议 | **FP**（Frontal-Profile） |
| 类型 | `pairwise` |
| 对数 | **7,000 对** |
| fold | **10-fold**（fold 0..9） |
| 每 fold | same 350 + diff 350 = 700 对 |
| label | 1 = same（3,500）、0 = diff（3,500） |
| CSV 表头 | `fold,label,p1,p2` |
| **aligned_root** | **`datasets/cfp_aligned`**（C2a ① 裁定；经 §0 机器口径印证） |
| manifest | `datasets/eval_manifests/cfp_fp_manifest.csv` |
| 评测矩阵 | **8 模型 x 单次冻结，出多少报多少**（模型清单随 C2 主注册登记，本文件不预填） |

### 1.2 manifest 溯源（全量 sha）

| 文件 | 大小 | sha256 |
|---|---|---|
| `eval_manifests/cfp_fp_manifest.csv` | 756,017 B | `dfabbea68cfacd429d686043ca05d6f8b5ff6ddf1f33720241f29b663c0297ed` |
| `eval_manifests/cfp_fp_manifest.meta.json` | 681 B | `ce96d0363fc92fdd090af676b6bdf671f7f55dae4277213d7fd5863eb5e2d1e0` |

meta 字段照录：

    dataset        = CFP-FP
    protocol       = FP
    kind           = pairwise
    n_rows         = 7000
    aligned_root   = E:\下次比赛项目\face_recognition_research\datasets\cfp_aligned
    built          = 2026-10-15T11:49:50
    script         = scripts/eval_generic_pairs.py
    manifest_sha256= dfabbea68cfacd429d686043ca05d6f8b5ff6ddf1f33720241f29b663c0297ed

**sha 自洽实测**：CSV 实算 sha256 == meta 记录值 = **True**。

### 1.3 官方协议文件双 sha 溯源照录

| 源文件 | 大小 | 行数 | sha256 |
|---|---|---|---|
| `Protocol/Pair_list_F.txt` | 193,893 B | **5,000** | `6c6fa2476555a298a16f9134e524de5b07ca89f6aca9ca2831eccfe37d5c99eb` |
| `Protocol/Pair_list_P.txt` | 76,893 B | **2,000** | `b9c677d732e7e894e7faa8d8b682741cf153db82f513f726b4e9516210b0035f` |

protocol_root = `E:\下次比赛项目\face_recognition_research\datasets\raw\cfp\cfp-dataset\Protocol`

**双 sha 实测**：两文件实算 sha256 与 meta 记录**逐字 MATCH**。

## 2. 数据本体核验

| 项 | 实测 |
|---|---|
| 目录 | `datasets/cfp_aligned` |
| 结构 | `Data/Images/<001-500>/{frontal,profile}/NN.jpg` |
| 图片数 | **7,000** |
| 尺寸 | **112x112 全量**（7,000/7,000 JPEG） |
| subjects | 500 |
| 完整性 | **500 x (10 frontal + 4 profile) = 7,000**，由 `Pair_list_F`(5,000) + `Pair_list_P`(2,000) 独立双证 |
| 附属 | `Data/list_name.txt` 6,955 B / 500 行（500 subject 名单）|

**路径解析自洽性（三法全测，7,000/7,000）**：

| 解析方式 | 命中 |
|---|---|
| A. 以 CSV 原样相对 repo root | **7000/7000** |
| B. 剥 `datasets/<dir>/` 前缀后 join aligned_root | **7000/7000** |
| C. 归一化斜杠后相对 repo root | **7000/7000** |

## 3. 清点节（登记不采信）

以下四目录**登记不采信**，不进入 CFP-FP 基准口径。

### 3.1 `eval_manifests_fid/`（CFP-FP - fid 变体）

- `cfp_fp_manifest.csv` 812,017 B，sha256 `cab9c8a8a6d66522d23c730260fd981fa89b6a7e1802ca4b7500ce40f555a459`，7,000 行，sha 自洽 True
- `cfp_fp_manifest.meta.json` 685 B，sha256 `0251410b7d9b3be4f3d221364347bca5f4d67187946ddc06ec3571c28e0cb80c`
- aligned_root = `datasets/cfp_aligned_fid`（7,000 张 / 112x112）
- Pair_list 双 sha 与主套**相同**；built 2026-10-15T13:19:53
- **不采信理由**：非 C2a 裁定之 aligned_root（§0 机器口径为 `cfp_aligned`）

### 3.2 `eval_manifests_fid2/`（CFP-FP - fid2 变体）

- `cfp_fp_manifest.csv` 826,017 B，sha256 `09013db6517de5552f06b7da2477304e1a21e4de4417dd2cde6c03d948d4b6d4`，7,000 行，sha 自洽 True
- `cfp_fp_manifest.meta.json` 686 B，sha256 `b94e4eb9e03e51d87a9cf91a5e4c9d88a991aedc6d88fff446a1c912def4b796`
- aligned_root = `datasets/cfp_aligned_fid2`（7,000 张 / 112x112）
- Pair_list 双 sha 与主套**相同**；built 2026-10-15T14:11:05
- **不采信理由**：非 C2a 裁定之 aligned_root

### 3.3 `cfp_ibr_bin/`（扁平双份）

- 文件数 **14,000**，命名 `p%05d_%d.jpg`，全部 112x112
- 解析：subject 序号 0..6999（7,000 个），每序号恰 2 个 idx（0 与 1），`idx_cnt = {0: 7000, 1: 7000}`
- 即 **7,000 图 x _0/_1 双份 = 14,000**；unmatched names = 0
- **不采信理由**：存在重复双份、无 identity/protocol 结构，与 7,000 单份口径不等价

### 3.4 `cfp_mirror_112/`（缺 1,099 张）

- 图片数 **5,901**（非 7,000），全 112x112，结构与主套相同
- 相对 `cfp_aligned` **缺失 1,099 张**（frontal 1,060 + profile 39），仅在镜像侧缺、无多出
- 影响 **487 / 500** 个 subject；**无全缺 subject**
- 缺失频次分布：缺 1 张 x 114 人、缺 2 x 191、缺 3 x 130、缺 4 x 47、缺 5 x 5
- 对齐日志 `results/_align_cfp_mirror_v1_20261017.txt`（UTF-16LE / 1,486 B）记录：
  `total=5901`、`aligned=5827`、`fallback_resize=74`、`elapsed=587s`、`selfcheck PASS`
- **不采信理由**：**源清单本身即 5,901 而非 7,000**，缺 1,099 张破坏官方配对完整性，不能作为 FP 协议载体

## 4. 风险登记

1. **`DEFAULT_ALIGNED[agedb30]` 指向不存在目录**：`datasets/agedb30_aligned` 在盘为**否**；实盘使用 `datasets/agedb30_variantA`（经 `--aligned_root` 参数覆盖）。**任何直接用默认值的调用会失败**。
2. **三套 manifest 路径分隔符书写不一致**：主套 `eval_manifests/` 为混合斜杠（`datasets/cfp_aligned\Data/...`），fid / fid2 为纯反斜杠。
   Windows 下三种解析法均 100% 命中，但**跨平台不安全**，移植须先归一化斜杠。
3. **旧 `CFPW_sims.csv` 协议不同构，不作校验**：该件（`weights/models/buffalo_s/`，mtime 2026-09-14）为 13,914 对且 `protocol` 列值 `FF`（FF+FP 合并），与本注册之 FP-only 7,000 对**不同构**；仅可作 sanity 参照，不入门禁。
4. **CALFW / CPLFW 数据集不在盘**（全盘 calfw/cplfw 文件名命中 = 0，仅 `buffalo_s/` 下 6 个外部产物）。C1 两项下载后方可补注册。

## 5. 取证与写盘披露

- 本批全部核验均为**只读**；除新建本文件外，**零写入、零删除、零覆写**，`results/*.json` 零接触。
- **本批取证脚本缺陷如实披露**：`_c2a_gate2.py` 之 H4 段 join root **少一层**
  （误用 `E:\下次比赛项目` 而数据在 `...\face_recognition_research\datasets`），导致**误报**三套 manifest 路径全部不可解析。
  已由 `_c2a_gate3.py` 修正 join root 并**全量重测为 7000/7000**；错误结论**不入册**，仅此处披露。
- 行尾口径：纯 LF、无 BOM；`CRLF = 0`。

## 6. 待办（不预授权）

1. **C1**：CALFW + CPLFW 两项下载（官方渠道，LFW 式 6,000 对协议），落盘 `datasets\` 后回报路径；CFP pairs **无需下载**（§1.3 官方文件在盘）。
2. **C2**：CALFW / CPLFW 补注册（随 C1 到盘），8 模型清单随主注册一并登记。
3. **C3**：管线移植 + 教师先行粗校验；CFP 采用主套 `eval_manifests/` 与 `cfp_aligned`，**不用 mirror**（§3.4）。
4. 旧 `CFPW_sims.csv` 之教师 `cos_sim` 仅作 sanity 交叉参照，**非门禁**（§4-3）。

---

*本文件为注册件，仅登记口径与溯源。任何数值结果不在本文件内产生。*
