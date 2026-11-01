# SA-EdgeFace: C2-M10 消融实验

人脸嵌入模型 KD（知识蒸馏）对比训练——基于 ShuffleNetV2 0.5×（0.87M 参数）在 CASIA-WebFace 上的系统消融，预注册规格 `docs/prereg_grid_v3.6.md`。主张主语：s20_l01 + CALFW（跨域蒸馏增益）。

## 预注册与判决链

冻结 commits（KD 判决，零重写零 rebase）：

| commit | 内容 |
|--------|------|
| `8af75ea` | KD 尾部 LFW 3 格落袋（cfg0 91.45 / cfg4 92.37 / cfg4_s20_l01 92.97）+ 钉自查 H_cfpfp |
| `a0c7e29` | CALFW manifest 白名单放行（dd8aa594 == fc2df448 逐行等价） |
| `d1a29f8` | CALFW KD 三格落袋（cfg0 69.53 / cfg4 74.05 / cfg4_s20_l01 78.37）+ 四 Δ 全正叙事句 |

后续链：`a5be5c4`（line-370 补修 + cfg4@456 评测）→ `224c15f`（epoch checkpoint 补丁）→ `4e6cdc0`/`ab0db6d`/`c9d3fcf`（cfg0@456 三件套）→ `12eaa09`（分支 iii 裁决）→ `c8cfa1b`（结构整理）。

## 时间线（2026-10-20 → 11-01）

以 `git log --since="2026-10-20"` 素材为主线（36 commits），锚定事件：

- **10-20 → 10-26**：C2-M1b/M2 取证 → C2-M3 系列法证（LFW 校准重放 PASS、CFP-FP V0/V1/V2 三轮校准、CALFW 新锚铸造）→ C2-M4v3 网格预注册（17 cell）。
- **10-27**：C2-M6~M9 三钉链条铸造（第四钉 epoch==30 / 第五钉 med|w| 湮灭门 / 第六钉 AUC≥0.60 功能快检）+ v3.3~v3.6 规格演进 + 主检验 cfg3−cfg0=+0.72pp < 1.0pp → 主张终结=趋势。
- **10-28**：cfg3@456 重训完成（epoch 30/30, health-ok）+ PyTorch 2.6 torch.load weights_only 兼容修复链。
- **10-30 晨**：cfg4@456（KD 臂）完训 ~4.7h，sha256 `D4F11D61…E7F8`，final 落盘 `weights/ablation_casia/cfg4_seed456_final.pt`。
- **10-30 13:17 → 10-31 09:44**：执行链静置 19.5h（框未执行，非技术原因，`docs/CONFLICTS.md` 已记）。
- **10-31**：cfg0@456 完训（LFW 91.67 / CALFW 72.58 / CFP-FP 80.57，commits `4e6cdc0`/`ab0db6d`/`c9d3fcf`）；Δ@456 初读 −7.50pp（与 KD 判决旧 Δ +0.92pp 符号翻转）。
- **11-01 晨**：诊断闸分支 iii——重测 84.1667% 与原读逐位一致、sha 匹配、日志收敛 → 按 prereg v3.6 第三档落档（`12eaa09`）；同案四证（med|w| 差 25× / loss_arc 差 ~3pp / LFW 出带 [91.42, 93.32] / CALFW AUC 0.55）共指 cfg4_seed456 重训产物异常 → 两口径并列，归因终审 pending 11-04（不预判）。
- **11-02 / 11-03 预排**：cfg3@789、cfg3@42；**11-04 终审**。

## 评测矩阵状态表

4 臂 × 3 格（LFW / CALFW / CFP-FP），已填 7 格（含 recheck），缺格标 pending：

| 臂 | LFW | CALFW | CFP-FP |
|----|-----|-------|--------|
| cfg4@456 | 84.17%（recheck 84.17%） | 崩坏（AUC 0.55） | 78.09% |
| cfg0@456 | 91.67% | 72.58% | 80.57% |
| cfg3@789 | pending | pending | pending |
| cfg3@42 | pending | pending | pending |

## Δ@456 判读

按 `docs/prereg_grid_v3.6.md` 三档规则：

| Δ 范围 | 判定 |
|--------|------|
| ≥ +1.08pp | 主张级 |
| +0.5 ~ +1.08pp | 降描述级（主张句主语换 s20_l01 + CALFW） |
| < +0.5pp 或反向 | 域特异性 + 复现失败注脚 |

**当前 Δ@456**：LFW −7.50pp / CALFW N/A（崩坏）/ CFP-FP −2.48pp → **域特异性 + 复现失败注脚**（全反向/崩坏，与旧 Δ +0.92pp 符号翻转）。归因终审 pending 11-04。

## 仓库结构

```
face_recognition_research/
├── qad_face/          # 核心代码（models/train/data/utils）
├── scripts/           # 训练与评测脚本
│   ├── train_ablation.py      # 训练入口
│   ├── eval_generic_pairs.py  # 通用 pair-wise 评测
│   └── _eval_calib_mirror_v1.py  # LFW 校准评测
├── results/           # 评测结果 H_*.json
├── docs/              # 文档
│   ├── prereg_grid_v3.6.md    # 预注册规格（当前版）
│   ├── CONFLICTS.md           # 冲突登记与裁决
│   └── ...
├── archive/           # 归档脚本（PowerShell）
├── weights/           # 权重（gitignored，本地留存）
├── logs_multiseed/    # 训练日志（gitignored）
├── requirements.txt
└── README.md
```

## 复现指引

### 环境

```bash
pip install -r requirements.txt
```

Python 3.10+，PyTorch 2.4+（本实验在 PyTorch 2.6 下验证，`torch.load` 需 `weights_only=False`）。

### 数据

- **CASIA-WebFace**：训练集，10572 IDs / 490623 imgs。获取方式：[CASIA-WebFace 官方申请](http://vis-www.cs.umass.edu/lfw/)。
- **LFW**：Labeled Faces in the Wild，10-fold CV pair-wise 评测。
- **CALFW**：Cross-Age LFW，5 点仿射对齐至 112×112。
- **CFP-FP**：Celebrities in Frontal-Profile，fiducial 30 点 V2 校准（sha `09013db6…`）。

### 训练

```bash
python scripts/train_ablation.py \
  --data <casia-webface-path> \
  --gpu 0 --epochs 30 --batch_size 256 \
  --configs 4 --seed 456 \
  --save_dir weights/ablation_casia \
  --eval_every 3
```

`--configs` 索引：0=baseline / 1=+channel / 2=+spatial / 3=+full SAB / 4=+distill / 5=+distill+aug。

### 评测

```bash
# LFW
python scripts/_eval_calib_mirror_v1.py --dataset lfw --batch 256 \
  --ckpt weights/ablation_casia/<arm>_final.pt --cfg x05_sab_full \
  --out results/H_lfw_<arm>.json

# CALFW / CFP-FP
python scripts/eval_generic_pairs.py --dataset calfw \
  --ckpt weights/ablation_casia/<arm>_final.pt --cfg x05_sab_full \
  --aligned_root datasets/calfw_aligned \
  --manifest_dir datasets/eval_manifests \
  --out results/H_calfw_<arm>.json
```

### 权重 sha256 对账

| 文件 | sha256（前 8 位） |
|------|-------------------|
| cfg4_seed456_final.pt | `D4F11D61` |

权重文件 gitignored，本地留存。训练日志 gitignored，位于 `logs_multiseed/`。

## License

MIT（见 `LICENSE`）。
