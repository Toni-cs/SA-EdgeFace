# prereg_efficiency.md — D-60 效率测量预登记（非判门工程测量）

> 关系条款：本件 = D-60 效率测量豁免的**跑前预登记**（CONFLICTS D-60，2026-10-18）。
> 非判门数据，只进效率表替换 legacy 数（闭 C4），**不触任何判门数字**。
> 不 amendment 任何既有冻结件（prereg_mirror / prereg_v2_matrix / prereg_sigma 各自 append-only，互不触碰）。
> 授权 = 用户口令 2026-10-18（任务 B 部分，唯一放行新跑）；先冻结后开跑。

## 跑前一行测量预登记（2026-10-18，跑前冻结）

D-60 效率测量预登记（2026-10-18，跑前冻结）：脚本 scripts/stats_efficiency.py（T-R4 合规，单测在案 unit_tests_matrix_v2.py:166-183）；输入 112、batch=1；warmup=50 不计入、测量 200 报 p50/p95（中位口径）；计时 GPU=torch.cuda.Event 成对 / CPU=perf_counter（脚本内置）；推理路径 = get_embedding（与评测同径，含 L2 归一化）；硬件型号脚本自动记入 JSON；口径 = 仅推理时延；输出 results/_t4_efficiency_20261018.json（--out 已存在 exit 5 拒覆盖）；用途 = 效率表替换闭 C4，非判门数据。
