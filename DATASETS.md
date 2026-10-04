# 数据集说明

本项目使用两个公开标准数据集：CASIA-WebFace（训练）和 LFW（测试）。

## 1. CASIA-WebFace（训练集）

### 基本信息
- **规模**：约 490,623 张图像，10,572 个身份
- **来源**：中国科学院自动化研究所（CASIA）公开数据集
- **官网**：http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- **大小**：约 2.5 GB（解压后）

### 下载方式
1. 访问官网注册账号
2. 下载 CASIA-WebFace 压缩包（通常为 .zip 或 .7z 格式）
3. 解压到 `datasets/raw/casia-webface/` 目录

### 目录结构
```
datasets/raw/casia-webface/
├── 000000/           # 身份ID（6位数字，000000 ~ 010571）
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── 000001/
│   └── ...
└── ...
```
共 10,572 个子文件夹，每个子文件夹对应一个身份，包含该身份的多张人脸图像。

### 预处理
- 使用 InsightFace buffalo_s 检测器检测人脸和 5 个关键点
- 使用 ArcFace 5 点模板进行相似变换对齐到 112×112
- 对齐脚本：`python scripts/align_lfw.py`（LFW 对齐，CASIA 训练数据已对齐）

## 2. LFW（Labeled Faces in the Wild，测试集）

### 基本信息
- **规模**：13,233 张图像，5,749 个身份
- **来源**：马萨诸塞大学阿默斯特分校公开数据集
- **官网**：http://vis-www.cs.umass.edu/lfw/
- **大小**：约 110 MB（原始），约 60 MB（对齐后）

### 下载方式
1. 访问官网下载
   - 原始图像：http://vis-www.cs.umass.edu/lfw/lfw.tgz
   - Pairs 列表：http://vis-www.cs.umass.edu/lfw/pairs.txt
2. 解压原始图像到 `datasets/lfw/`
3. pairs.txt 放到 `datasets/pairs.txt`

### 目录结构
```
datasets/
├── lfw/                    # 原始 LFW 图像（250×250）
│   ├── Aaron_Eckhart/
│   │   └── Aaron_Eckhart_0001.jpg
│   └── ...
├── lfw_aligned/            # 对齐后的 LFW 图像（112×112）
│   ├── Aaron_Eckhart/
│   │   └── Aaron_Eckhart_0001.jpg
│   └── ...
└── pairs.txt               # 6000 对验证列表（首行 "10 300"）
```

### 评测协议
- 标准 10-fold 交叉验证
- 共 6000 对：3000 对正例（同一人）+ 3000 对负例（不同人）
- pairs.txt 格式：首行 "10 300" 表示 10 折，每折 300 正例 300 负例

### 对齐方法
```bash
python scripts/align_lfw.py
```
- 使用 InsightFace buffalo_s 检测器
- ArcFace 5 点模板（双眼、鼻尖、双嘴角）
- 相似变换对齐到 112×112
- 未检测到人脸的图像（约 24 张）使用原图 resize 到 112×112 代替
- 输出到 `datasets/lfw_aligned/`

## 3. 迷你样本子集（sample_datasets/）

为方便代码验证，压缩包中包含一个迷你样本子集：

```
sample_datasets/
├── casia-webface/      # CASIA 样本：5 个身份 × 5 张图 = 25 张
│   ├── 000000/
│   ├── 000001/
│   └── ...
└── lfw_aligned/        # LFW 对齐样本：20 张
    └── *.jpg
```

**注意**：样本子集仅用于验证代码能正常运行，不能用于训练或正式评测。完整数据集需按上述方式自行下载。

## 4. 教师模型特征缓存

知识蒸馏使用 InsightFace buffalo_s 作为教师模型，教师特征预计算并缓存：

- **路径**：`weights/ablation_casia/teacher_feats.npy`
- **大小**：约 0.94 GB
- **格式**：numpy 数组，shape = (490623, 512)，float32
- **生成方式**：运行 `python scripts/prepare_teacher.py`（需先下载 CASIA 并对齐）

此文件较大，不包含在压缩包中，需自行生成。

## 5. 数据集使用注意事项

1. CASIA-WebFace 仅供学术研究使用，商用需联系官方授权
2. LFW 是公开数据集，可自由用于学术研究
3. 训练和测试必须使用相同的对齐标准（本项目均使用 buffalo_s + ArcFace 5点模板）
4. 评测时必须使用官方 pairs.txt，不能自行划分测试集
