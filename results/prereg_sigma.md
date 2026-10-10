# prereg_sigma.md — 冻结于 22b8cdc（Phase 3 首次运行之前）
<!-- <commit> 填冻结时父提交 22b8cdc：freeze commit 自身哈希无法自引用，见 CONFLICTS §3 登记行。 -->
## 1. H5 预演分布（原样，2026-10-05，logs_multiseed/**，13 条 RESULT: AUC）
   通过 0/13；max 0.9418；min 0.5001；median 0.9373；
   30ep 档（3 条）max 0.8103；5ep 档（10 条）max 0.9418。
   结论：0.95 门在既有数据上通过率为 0；阈值保持 0.95，不降。
   <!-- 复算留证：results/_code_evidence/H5_prerun_13logs.txt，2026-10-10 按 PLAN_v6:744-757 原命令
        过滤 g2prime|d4a1|guard 3 条新日志后复算，五值逐位 = TOTAL 13 / PASS 0 / MAX 0.9418 /
        MEDIAN 0.9373 / MIN 0.5001（PLAN_v6:761-765）。原档 2026-10-05 执行，盘上证据件曾缺失，已补。 -->
## 2. σ̂ 三分支（见 PLAN_v4 §0.3，阈值与功效句原样照抄）
   | 实测 σ̂ (pp) | seed 数 | 对策 | 时程 | 功效句（原样） |
   |---|---|---|---|---|
   | **σ̂ ≤ 0.40** | n=8 | 0 | 10–12 周 | `n=8 gives power 0.458 at δ=0.667pp under Holm α=0.00333; the smallest effect detectable with 80% power is ≈1.1pp` |
   | **0.40 < σ̂ ≤ 0.50** | n=16 | +8 | 12–14 周 | 上句 + 扩样原因与扩样时刻（预注册分支） |
   | **σ̂ > 0.50** | **触发降级路径 D11**：在「a) n=16 并以 CI 为主、主张降为 pilot」与「b) 放弃主效应显著性主张、论文降为创新点 2+3」之间书面二选一 | +0/+8 | 10–12 或 12–14 周 | 必报 `underpowered at this effect size` |
   | **σ̂ > 0.60** | **强制走 b)**，停止扩样 | 0 | 10–12 周 | 论文降为**协议 + 尾部分解**两条创新点 → **3 区** |
## 3. 有效 n 降级规则（v3）：任一臂被剔 → 核算 n → n<8 → 立即 D11；
     H5 造成的 n<8 → 预注册为「扩样不可行」→ D11-(b)。
## 4. pilot 点：cfg3_seed42_final / cfg3_seed123_ep30 / cfg4_final（跨 cell、seed 归属不明，已披露）
## 5. amendment 规则：此后只允许追加带时间戳的 amendment（只增行、不改上面任何一行）。

## 6. Evaluation Protocol 子章（D-22 FAR 分辨率地板定义；件3a，随本冻结一次成型 2026-10-10）
TAR@FAR below the 3.33e-4 resolution floor (set by 3000 negative pairs)
is reported as the first measurable operating point; we do not claim
measurements below it. This definition applies to every table, figure, and
text statement of operating-point performance in this paper.

## 7. amendment（2026-10-11，D-17 H5 配对口径修复重跑后追加，只增行）
- 旧 §1 的 0/13 系坏配对口径（train_ablation 旧滑窗配对，T0 commit a6d5e82 已修复为 RandomState(0) 分层正负平衡）下的历史分布，不作推断依据。
- 新分布（eval-only 重跑 2026-10-11，scripts/_h5_rerun.py，评测时 HEAD=bc24e95；代码树零差异，本 §7 追加后由本 commit 新哈希取代）：n=12/13（cfg0_seed123_test 失联：final.pt 被同名 5ep screening 覆写无 bak，n_recoverable=12/13），max 0.9361、min 0.5002、median 0.93055 (=(0.9299+0.9312)/2，4 位临界值故保留 5 位)，**≥0.95 通过 0/12**；30ep 档（3 件）max 0.8119、5ep 档（9 件）max 0.9361。
- cfg3_seed123_30ep 以同 run ep30 代理已失联 final（student 张量 torch.equal 全等核验 2026-10-11，完好对 cfg3/0_seed42 ep30 vs final）。
- 结论不变：0.95 门在既有数据上通过率仍为 0；阈值保持 0.95 不降。旧 0/13 降为「旧配对下的历史分布」。
