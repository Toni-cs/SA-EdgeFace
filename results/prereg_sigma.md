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

## 8. amendment（2026-10-14，用户全权授权代录——会话原文「等待我的四件签字均授权你进行」；本节只增行，§1–§7 一字未动）

1. **D11(b) 激活（P4 终审庭 2026-10-14 给定句，用户授权落笔）**：
   σ̂ ≈ 1.15pp（pilot 三点〔§4 三件〕，2026-10-11 三庭终审实测，下限口径）→ 按 §2 冻结表字面判定：**D11(b)**。v6 12-cell 主效应族就此关闭。
   附带效力：§2 字面执行、不修口径（终审已裁：修口径=HARKing）；任何文书（论文、答辩、QA 表）不得将本判决复述为「待重新评估」「口径存疑」「历史遗留问题」。
2. **D13 披露（PLAN:109）**：B 臂（cfg4_seed42）attempt1 双门失败——G3-f1 `acc_cv=0.8627 < 0.92`（`logs_phase3/B_cfg4_seed42.txt.eval1.txt`）；H5 `AUC=0.9119 < 0.95`（`B_cfg4_seed42.txt:648`）。训练集自评 AUC 分布以 §1/§7 为准（新口径 0/12 ≥0.95、max 0.9361）。n 核算：B 不入 12-cell、不入 σ̂ 池（PLAN:907/:1074）→ 有效 n 不变，不因 B 触发 D11；G3 臂族缺 B + E BLOCKED 共 2 缺 ≤「缺臂 >2 才缩减」（PLAN:1182），登记不缩减。
3. **B 处置 = B′（法证庭推荐 2026-10-14，用户授权选择）**：接受 attempt1 结果 86.27% 为该臂终值，`_phase3_progress.json` 置 `status=excluded`，**放弃 PLAN:1094 的最后一次重跑权利**。授权依据（法证庭 2026-10-14）：attempt2 ep1–ep16 与 attempt1 张量逐位相同、跨 seed 塌陷 0/6、锚点系 10-01 未提交工作树产物（打印行与保存名互斥，任何 commit 均不含该配方）。操作限于 status/note 字段与残余 ep 文件隔离至 `weights/_hist/`；日志与既有 `_hist` bak 字节不动。
   锚点跨 era 补披露：`cfg4_final.pt`（94.30%）的 seed 归属 §4 已披露，**其训练时 `--grad_clip` 实际值当时未披露**，本行补——不可考；「复现 94.30」不可执行，94.30 只作跨 era 观察（D-12(4)/D-13(1)）。
4. **lr 0.1 → 0.05（Phase 3 后续全部臂）**：如实披露——此为看到 B attempt1（86.27%）与 G2′/D4（86.90% / 95.52%）**之后**的参数变更。中立依据（era×lr 矩阵）：≤10-01 工作树 lr0.1 = 6/6 存活；10-01 之后代码态 lr0.1 = 0/6（10-03 ×3、9caa5a8 guard、a6d5e82 g2prime、dc43304 B）；同代码态 lr0.05 = 1/1（d4a1 95.52%）；张量法证 = ep4 起 loss 冻结、med|w| 单调腰斩、weight decay 磨穿参数。性质 = **稳定性修复，非扫参择优**（PLAN:913：Methods 禁写成扫描找最优）。生效方式：`_phase3_run.py` train_cmd 显式传 `--lr 0.05`（现 driver 不传、`train_ablation.py:198` 默认 0.1）——该代码改动经本行授权，随 v2.0 manifest 重建一并落码并过 S1–S17 自测。
5. **pilot 登记（法证庭提议，用户 2026-10-14 放行）**：cfg4 × `--lr 0.05` × seed42 × 30ep × batch 256 × grad_clip 默认 10.0，单次 ≈1.75 GPU-h；显式传 `--distill_start 20 --w_feat 0.1`（值与 cfg4 配方相同，仅触发 dc43304 因子后缀）→ 落名 `cfg4_seed42_s20_l01_*`，与 B 产物物理隔离。**标 pilot**：不入 manifest、不计入任何注册臂计数、不冒充 B；只作「lr0.05 能否救活 cfg4 配方」的单点证据。命令与日志路径见 CONFLICTS D-26。
