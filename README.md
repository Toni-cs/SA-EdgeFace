# SA-EdgeFace: 面向监控场景的轻量人脸识别

> Surveillance-Attention EdgeFace — 基于 ShuffleNetV2 的超轻量人脸识别框架，融合注意力机制、知识蒸馏与监控场景数据增强。

## 项目简介

SA-EdgeFace 是一个面向**监控场景边缘部署**的轻量人脸识别系统。骨干网采用 ShuffleNetV2 x0.5（仅 0.87M 参数），通过以下三个模块的系统消融实验，在 LFW 上达到 94.68% 准确率：

- **C1 — 监控注意力模块 (SAB)**：ECA 通道注意力 + 空间注意力 + 多尺度局部细节增强
- **C2 — 知识蒸馏**：InsightFace buffalo_s 教师模型离线特征蒸馏，分阶段启动策略
- **C3 — 监控退化增强**：低分辨率、模糊、遮挡、JPEG 压缩等监控场景数据增强

## 模型架构

### 骨干网络
- **ShuffleNetV2 x0.5**：0.866M 参数，0.0228 GFLOPs，CPU 推理 8.3ms
- SAB 模块插入 stage3 和 stage4 末尾
- 嵌入维度：512

### SAB 注意力模块
三个可独立开关的分支：

| 分支 | 作用 | 参数量 |
|------|------|--------|
| Channel (ECA) | 通道注意力，1D 卷积 | ~k |
| Spatial | 空间注意力，DW 5x5 + PW 1x1 | 极少 |
| Local Detail | 多尺度局部细节增强，DW 3x3 + 5x5 | 较少 |

输出公式：`SAB(x) = x ⊙ c ⊙ s + α · d`（残差连接，α=0.01 固定）

### 知识蒸馏
- 教师：InsightFace buffalo_s（MobileFaceNet 骨干，预训练）
- 策略：前 20 epoch 纯 ArcFace 训练，epoch 20 后 lr 衰减时启动蒸馏
- 损失：`L = w_arc · L_arc + w_feat · L_feat`（w_feat=0.1，余弦特征蒸馏）
- 教师特征预计算缓存，训练时无需在线推理教师模型

## 实验结果

### 6 配置消融实验（LFW，对齐后 6000 对协议）

| 配置 | 模块 | Acc (%) | AUC | EER (%) | TAR@FAR=1e-3 (%) |
|------|------|---------|-----|---------|-------------------|
| cfg0 | baseline | 94.32 | 0.9860 | 5.85 | 42.87 |
| cfg1 | +ECA channel | 94.03 | 0.9853 | 6.22 | 48.37 |
| cfg2 | +spatial | 94.35 | 0.9880 | 5.78 | **58.00** |
| cfg3 | +full SAB | 93.92 | 0.9851 | 6.12 | 44.77 |
| **cfg4** | **+distill** | **94.68** | **0.9882** | **5.47** | 39.70 |
| cfg5 | +augmentation | 93.73 | 0.9829 | 6.43 | 37.93 |

### 关键发现

1. **知识蒸馏 (C2) 提升最显著**：cfg3→cfg4 提升 +0.76% Acc，是所有模块中最有效的
2. **空间注意力 (C1 spatial) 提升 TAR 明显**：TAR@FAR=1e-3 从 42.87% 提升到 58.00%（+15.13%），对低 FAR 场景价值大
3. **Full SAB 反而有害**：三分支叠加比单独 spatial 降 0.43%，说明超小模型上注意力模块存在冗余
4. **数据增强 (C3) 有害**：light 强度增强仍导致 -0.95%，小模型容量不足以吸收增强噪声
5. **ECA 通道注意力无效**：单独加 ECA 反而降 0.29%

### 模型效率对比（CPU，112x112 输入，100 次平均）

| 模型 | Params (M) | FLOPs (G) | Latency (ms) |
|------|-----------|-----------|--------------|
| EdgeFace 1.0x | 1.778 | 0.0799 | 9.77 |
| EdgeFace 1.0x + SAB | 2.090 | 0.0943 | 10.75 |
| **ShuffleNetV2 x0.5 (ours)** | **0.866** | **0.0228** | **8.34** |
| x0.5 + SAB full | 0.930 | 0.0259 | 11.10 |
| x0.5 + spatial only | 0.874 | 0.0232 | 9.07 |

## 环境配置

### 硬件要求
- GPU：NVIDIA RTX 系列（推荐 ≥6GB 显存）
- 训练：RTX 5060，batch_size=256，约 4 小时/30 epoch
- 推理：CPU 即可，单张 8-11ms

### 软件依赖
```
Python >= 3.10
PyTorch >= 2.0 (CUDA 12.x)
torchvision
opencv-python
numpy
insightface (用于人脸对齐)
onnxruntime-gpu
scikit-learn
matplotlib
```

安装：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy insightface onnxruntime-gpu scikit-learn matplotlib
```

## 数据集

### 训练集：CASIA-WebFace
- 规模：~490,623 张图像，10,572 个身份
- 来源：中科院自动化所公开数据集
- 预处理：InsightFace buffalo_s 检测 + ArcFace 5 点模板对齐到 112x112
- 目录结构：`datasets/raw/casia-webface/{identity_id}/{image_id}.jpg`

### 测试集：LFW (Labeled Faces in the Wild)
- 规模：13,233 张图像，5,749 个身份
- 协议：标准 10-fold 6000 对（3000 正例 + 3000 负例）
- 预处理：同训练集对齐标准，输出到 `datasets/lfw_aligned/`
- pairs 文件：`datasets/pairs.txt`（首行 "10 300"）

## 使用方法

### 1. 数据准备
```bash
# 解压 CASIA-WebFace
# 对齐 LFW
python scripts/align_lfw.py
```

### 2. 训练消融实验（6 配置）
```bash
python scripts/train_ablation.py \
  --data datasets/raw/casia-webface \
  --gpu 0 \
  --epochs 30 \
  --batch_size 256 \
  --save_dir weights/ablation_casia \
  --teacher_cache weights/ablation_casia/teacher_feats.npy
```

单独训练某个配置：
```bash
python scripts/train_ablation.py --configs 4 --data datasets/raw/casia-webface --gpu 0
```

### 3. 训练 EdgeFace 1.0x baseline
```bash
python scripts/train_edgeface_1x.py --data datasets/raw/casia-webface --gpu 0
```

### 4. LFW 评测
```bash
# 批量评测所有配置
python scripts/eval_model.py \
  --batch \
  --save_dir weights/ablation_casia \
  --lfw \
  --lfw_pairs datasets/pairs.txt \
  --lfw_dir datasets/lfw_aligned \
  --gpu 0 \
  --output results/casia_lfw_results.txt

# 单模型评测
python scripts/eval_model.py \
  --checkpoint weights/ablation_casia/cfg4_final.pt \
  --config baseline \
  --lfw \
  --lfw_dir datasets/lfw_aligned \
  --gpu 0
```

### 5. 效率测试
```bash
python scripts/eval_efficiency.py
```

## 项目结构

```
face_recognition_research/
├── datasets/                  # 数据集（不包含在压缩包中）
│   ├── raw/casia-webface/     # CASIA-WebFace 训练集
│   ├── lfw/                   # LFW 原始数据
│   ├── lfw_aligned/           # 对齐后的 LFW
│   └── pairs.txt              # LFW 6000 对列表
├── qad_face/                  # 核心代码
│   ├── models/
│   │   ├── edgeface.py        # EdgeFace/ShuffleNetV2 模型定义
│   │   ├── attention.py       # SAB 注意力模块（ECA/Spatial/Detail）
│   │   └── shufflenetv2.py    # ShuffleNetV2 骨干网
│   ├── train/
│   │   ├── losses.py          # ArcFace Loss + 特征蒸馏 Loss
│   │   └── distill.py         # 蒸馏训练器（支持分阶段蒸馏）
│   └── utils.py               # 中文路径读写工具
├── scripts/                   # 训练/评测脚本
│   ├── train_ablation.py      # 6 配置消融训练
│   ├── train_edgeface_1x.py   # EdgeFace 1.0x baseline 训练
│   ├── eval_model.py          # LFW 评测
│   ├── align_lfw.py           # LFW 人脸对齐
│   └── eval_efficiency.py     # 效率测试
├── weights/                   # 模型权重（不包含在压缩包中）
├── results/                   # 实验结果
│   ├── casia_lfw_results.txt  # 6 配置 LFW 评测结果
│   └── efficiency.txt         # 效率测试结果
├── paper/                     # 论文（LaTeX）
│   ├── main.tex               # 主文件
│   ├── methods.tex            # 方法部分
│   └── figs/                  # 论文图表
└── README.md                  # 本文件
```

## 训练超参数

| 参数 | 值 |
|------|-----|
| 优化器 | SGD, momentum=0.9, weight_decay=5e-4 |
| 学习率 | 0.1，warmup 2 epoch，step 衰减（20ep→0.01, 25ep→0.001） |
| Batch size | 256 |
| Epochs | 30 |
| ArcFace | s=30, m=0.3 |
| 梯度裁剪 | 10.0 |
| 蒸馏启动 | epoch 20 |
| 蒸馏权重 | w_feat=0.1 |
| 数据增强 | light 强度，aug_prob=0.3（仅 cfg5） |
| 随机种子 | 固定（可复现） |

## 关键技术细节

### 训练稳定性修复（踩坑记录）

1. **SpatialAttention 加 BatchNorm**：防止 pwconv 输出漂移导致 sigmoid≈0、输出归零、梯度消失
2. **ArcFace Loss 强制 fp32**：AMP fp16 下 exp(64) 溢出，loss 计算用 fp32，模型 forward 仍 fp16
3. **梯度裁剪默认 10.0**：过小的裁剪（如 5.0）会把有效 lr 缩减到 1.7%，模型几乎不更新
4. **SGD 而非 AdamW**：ArcFace 人脸识别标配 SGD + 高 lr，AdamW 收敛差
5. **s=30 m=0.3**：s=64 m=0.5 对 0.87M 小模型太激进，loss 卡住不下降
6. **残差连接**：注意力输出改为 `x + x*sigmoid(y)`，输出范围 (x, 2x)，永不归零
7. **分阶段蒸馏**：前 20 epoch 纯 ArcFace，lr 衰减后再加蒸馏，避免早期梯度冲突

## 论文信息

- **标题**：Surveillance-Attention EdgeFace: Lightweight Face Recognition via Detail-Aware Attention and Knowledge Distillation
- **目标期刊**：IEEE Access / Sensors (SCI Q3)
- **格式**：IEEEtran，双栏，10pt
- **编译**：`pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

## 许可证

本项目仅用于学术研究。

## 联系方式

如有问题，请提交 Issue 或联系项目作者。
