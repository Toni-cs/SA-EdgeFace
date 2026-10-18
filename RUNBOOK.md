# RUNBOOK（通用跑件出件纪律）

> 本件为通用出件/失败防护纪律清单（2026-10-17 晚裁决指定、2026-10-18 晨落位；此前无通用 runbook 文件，Phase1-2 训练跑件另有 `paper_plan/RUNBOOK_Phase1-2.md`，各管各域）。

## 出件与失败防护（现行两条）

1. **分片落盘（D-43，2026-10-16）**：长跑件必须 checkpoint/分片落盘 + done_list，中断只损在途分片；不得全内存攒总出件。
2. **出件顺序硬化（D-56/D-57，2026-10-17 GBK 崩溃事故）**：JSON 落盘先于一切打印；print 一律 ASCII 安全（U+2212 → "-"）或 try/except 包裹；或预设 `PYTHONIOENCODING=utf-8`。计分完成、产物未写、崩在装饰性打印 = 失败时序颠倒。D-43 教「结果要分片落盘」，这次教「结果先于日志」——两条合起来封死「跑完了却没拿到结果」同类事故。

## 环境/网络先查（D-58，2026-10-18）

- **github 再断先查 hosts 是否被写回**：`Select-String -Path C:\Windows\System32\drivers\etc\hosts -Pattern 'github'`；指纹备份 = `hosts.bak_20261018`（留作比对，勿删）。10-17 夜 push 4 连败归因从「网断复发」更正为「hosts 被写」（21:31:51 写入 25 行 github→127.0.0.1，已删即通）——环境事实不明致误判，本项目第二次；断网类事故先核实环境再下结论。
- 复发时嫌疑度排序：迅游类加速器（惯用改 hosts）> 远控类常驻（向日葵/GameViewer，不用的卸载即收攻击面）；不追查写入者（收益不抵时间）。

## 环境注意（Windows PS 控制台）

- PS 控制台默认 GBK：Python 输出含非 ASCII 字符（−、±、中文）时必须先 `$env:PYTHONUTF8="1"`（或 `PYTHONIOENCODING=utf-8`），否则解码崩溃可发生在任何 print。
