# prereg_grid_v3 — C2-M4v3 网格预注册规格件（2026-10-26 夜）

> 本文件 = C2-M4v3 执行框全文落盘（规格补铸）。**铁律：先注册后运行；运行开始后规格零改动；重跑 ≠ 改规格。**

## 0｜cfg1/2/5 出身法证（先于一切）
- **a. 产物**：`cfg1_final.pt` size=50441311 mtime=**2026-09-29 13:15:20**；`cfg2_final.pt` size=50514225 mtime=**2026-09-29 15:16:52**；`cfg5_final.pt` size=50984223 mtime=**2026-10-01 20:38:40**。（对照 cfg0_final=09-28 19:36 / cfg3_final=09-30 11:39 / cfg4_final=10-01 10:39）
- **b. 训练落款检索**：`logs_cfg1-5_v3.txt`（cfg1@612、cfg2@1225）、`logs_cfg5_v2.txt`（cfg5@613）；**两日志头部均无 seed 记录**（无 `[ARM] … seed=` 行，旧版脚本产生）。config 描述与今日一致（cfg1: w_feat=0/aug=False；cfg5: w_feat=0.1/aug=True/distill_start=20）。
- **c. seed 命名约定史 / config drift**：`git log -p scripts\train_ablation.py` 显示 `save_name = f"cfg{orig_i}_seed{args.seed}{suffix}"`（`4a94a96` 起带 suffix）；`cfgN_final.pt`（无种子名）与 `cfgN_seed{S}_final.pt`（带种子）为两批产物。**cfg1/2/5_final 的 mtime（09-29~10-01）早于仓库首次提交 `e463d10`（2026-10-04）→ git 追溯不到其代码版本 → config drift 不可排除**。
- **d. 裁定**：**cfg1/cfg2/cfg5 = RED**（seed 未落定 ∧ 与今日 config 一致性不可验证；任一不落即 RED）→ **今晚不入核心层**。训练时长估计（明晚重训候选）：cfg1 ≈1.6h、cfg2 ≈1.6h、cfg5 ≈2.2h，合计 ≈5.4h。

## 1｜网格三层（落定规模 = 17 cell）
- **核心层（arm × 3 协议）**：
  - 无条件（9 cell）：`cfg0`/`cfg3`/`cfg4` @ **seed456** × {LFW, CALFW, CFP-FP}
  - 法证 GREEN 才入（0–9 cell）：cfg1/2/5 @ 法证落定 seed → **本批 RED，0 cell**
- **方差层（仅 CFP-FP，8 cell）**：`cfg0` × seed{42,123,789}=3；`cfg3` × seed{42,123,789}=3；`cfg4` × seed{123,789}=2（seed42 仅存变体 s20_l01，非臂内，排除；seed2024 同理不入）
- **合计 = 17 cell**（核心 9 + 方差 8）
- **CFP-FF**：不入本批，转 C2-M4b 白昼批（manifest 铸造 + 教师 FF 锚 + 补 cell）

## 协议死条款（继承 v2）
- `LFW` → `datasets\lfw_mirror_112`，教师锚 **97.08**（逐位认证）
- `CALFW` → `datasets\calfw_aligned` ＋ `calfw_manifest.csv`（sha **65879bca**），教师锚 **0.967338**
- `CFP-FP` → `cfp_aligned_fid2` ＋ `eval_manifests_fid2\cfp_fp_manifest.csv`（sha **09013db6**），教师锚 **95.5429**
- **禁触**：`cfp_aligned\`、`ca-aligned`、`eval_manifests` 根目录
- **canary**：①=100% ②=49.625% 每 cell 必带
- **sidecar 三钉**：`model_sha`（臂权重逐 cell 算）/ `manifest_sha` / `crop_chain`（CFP=fid2 / CALFW=calfw_v1）
- **cell_id** = `G{nn}_{arm}_{protocol}`；指标 = `acc_cv10±std` / `AUC` / `EER`

## 判读规则（注册后零改动）
- **卫生 FAIL**：canary 败 / 缺钉 / AUC<0.90 → 该 cell 无效重跑（重跑 ≠ 改规格）
- **主检验**：核心层最优臂 `CFP-FP acc − cfg0 CFP-FP acc ≥ +1.0pp` → 论文主张成立；否则降级为趋势描述（数据照报不删）。**法证臂 RED → 检验限干净臂集合 {cfg0,cfg3,cfg4}，回报注明覆盖范围**；法证臂 seed≠456 时带 seed 错配注记参与。
- **副检验（描述性，不设阈值）**：方差层种子间 std vs 主检验 gap 的信噪对比报出；**孤例复核**：同臂相邻协议差 >15pp → 重跑取重跑值并注记
- **教师锚差全表报出，不设 FAIL 线**（落后幅度是论文内容，非事故）

## 2｜发射（挂夜）
- `Start-Process` 脱离会话；单日志；顺序 = ①干净臂核心 9 → ②方差 8 → ③法证臂（GREEN 才跑）
- 单 cell 预期 10–20 分钟；**cell 失败 = 停、报回、不跳过**

## 3｜晨报
- 行 = `cell_id｜arm｜protocol｜acc±std｜AUC｜EER｜教师 gap｜状态`，FAIL 置顶
- 尾行 = 完成 x/17、总时长、异常清单
- 附三表：CFP-FP 臂排名（含种子散点）/ 完成矩阵 / 法证三臂出身摘要

## 4｜簿记
- commit `C2-M4v3：法证+预注册+发射`（照实措辞，未发生项不入 message）

---
**注册信息**：规格件 = 本文件；法证裁定 = cfg1/cfg2/cfg5 RED；网格规模落定 = 17 cell（核心 9 + 方差 8）。