# CASIA-WebFace 全量训练 + LFW 评测指南

## 环境

- GPU: RTX 5060 Laptop (8GB, sm_120)
- PyTorch: 2.11.0+cu128 (已安装, GPU 可用)
- 数据: CASIA-WebFace 10572 身份, 490623 张图 (112x112 已对齐)
- 教师模型: weights/models/buffalo_s/w600k_mbf.onnx

## Step 1: 全量训练 (6 配置消融)

在 PowerShell 中运行:

```powershell
cd E:\下次比赛项目\face_recognition_research
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --save_dir weights/ablation_casia --teacher_cache weights/ablation_casia/teacher_feats.npy
```

预计时间: ~24 小时 (6 配置 * 30 epoch * ~8min/epoch)

### 分配置运行 (如果时间不够分段跑)

```powershell
# 先跑 baseline + C1 (无蒸馏, 无需教师特征)
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --save_dir weights/ablation_casia --configs 0 1 2 3

# 再跑蒸馏配置 (需要教师特征, 首次自动预计算 ~5 分钟)
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --save_dir weights/ablation_casia --configs 4 5 --teacher_cache weights/ablation_casia/teacher_feats.npy
```

## Step 2: LFW 评测

训练完成后:

```powershell
python scripts/eval_model.py --batch --save_dir weights/ablation_casia --lfw --lfw_pairs datasets/pairs.txt --gpu 0 --output results/casia_lfw_results.txt
```

预计时间: ~10 分钟 (6 模型 * ~100s)

## Step 3: 填论文数值

将 `results/casia_lfw_results.txt` 中的 LFW Acc/AUC/EER 填入 `paper/main.tex` 的实验表格。

## 注意事项

1. 如果 GPU 显存不足 (OOM), 降低 batch_size 到 128 或 64
2. 教师特征缓存 (~950MB) 只需计算一次, 后续蒸馏配置直接加载
3. 训练日志中每个 epoch 约 8 分钟, 30 epoch 约 4 小时/配置
4. 如果中断, 需从头跑该配置 (无断点续训, 可后续添加)
5. C3 增强配置 (cfg5) 会在线增强, 速度约慢 20%

## 当前已有结果

| 实验 | 状态 | 结果文件 |
|------|------|----------|
| 6 配置消融 (52 id) | 完成 | results/ablation_results.txt |
| CASIA 50id 6 配置 | 完成 | weights/casia50/results.txt |
| 蒸馏超参搜索 | 完成 | results/distill_sweep.txt |
| 效率评测 | 完成 | results/efficiency.txt |
| LFW 评测 (50 id) | 完成 (近随机, 符合预期) | results/lfw_eval_results.txt |
| **CASIA 全量训练** | **待用户本地运行** | - |
| **LFW 评测 (CASIA)** | **待全量训练后** | - |
| QMUL-SurvFace | 需申请数据集 | - |
