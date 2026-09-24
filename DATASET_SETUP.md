# 数据集与模型准备指南

由于网络环境限制，部分数据集和模型需要手动下载。

## 一、评测数据集

### 1. LFW (Labeled Faces in the Wild)
- 官网: http://vis-www.cs.umass.edu/lfw/
- 需要下载:
  - `lfw.tgz` (约 170MB): 人脸图片
  - `pairs.txt`: 评测配对文件
- 国内镜像备选:
  - 百度网盘搜索 "LFW 数据集"
  - Gitee 搜索 "lfw dataset"
- 放置路径:
  ```
  datasets/
  ├── lfw/           # 解压后的图片目录 (lfw/name/name_0001.jpg)
  └── pairs.txt      # 评测配对
  ```

### 2. CFP-FP (后续)
- 官网: http://www.cfpwd.org/
- 放置路径: `datasets/cfp_fp/`

### 3. AgeDB-30 (后续)
- 官网: https://ibug.doc.ic.ac.uk/resources/agedb/
- 放置路径: `datasets/agedb_30/`

### 4. IJB-B / IJB-C (后续，核心评测)
- 官网: https://www.nist.gov/programs-projects/face-challenges
- 放置路径: `datasets/ijbb/`, `datasets/ijbc/`

### 5. QMUL-SurvFace (监控场景专项，核心)
- 官网: https://qmul-survface.github.io/
- 放置路径: `datasets/qmul_survface/`

## 二、模型权重

### InsightFace 模型 (自动下载)
- buffalo_s: 轻量模型（检测+识别），约 30MB
- buffalo_m: 中等模型
- buffalo_l: 大模型（ResNet50 识别）
- 自动下载到 `weights/` 目录

### EdgeFace (需手动获取)
- GitHub: https://github.com/IdiapResearch/EdgeFace
- 预训练权重在 GitHub Release 或 HuggingFace
- 放置路径: `weights/edgeface.onnx` 或 `weights/edgeface.pth`

### GhostFaceNet (需手动获取)
- GitHub: https://github.com/HamadYA/GhostFaceNets
- 放置路径: `weights/ghostfacenet.onnx`

## 三、验证环境

```bash
# 激活虚拟环境
E:\下次比赛项目\.venv\Scripts\activate

# 运行冒烟测试（需要至少2张有人脸的图片）
python scripts/smoke_test.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg

# 运行 LFW 评测（数据集准备好后）
python scripts/eval_lfw.py --model insightface_buffalo_s
```
