# EfficientNet 物种分类项目

项目位置：`E:\ELEC4240\SpeciesRecognition`。本阶段实现 **100 类、只有图片输入的监督分类基线**，以便后续比较文字辅助训练的提升。训练仍需要类别标签，预测只输入图片。文字训练分支尚未实现。

## 环境和模型

- 本机：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB 显存。
- Python 3.12；PyTorch 2.11.0+cu128；Torchvision 0.26.0+cu128。
- 项目解释器：`.venv\Scripts\python.exe`。
- 该 venv 通过 `--system-site-packages` 复用 `E:\conda-envs\comp4471` 的已安装依赖；未修改该课程环境。此环境依赖原环境存在，不是可独立搬走的完整安装。
- EfficientNet-B0 使用 ImageNet-1K 预训练权重，分类层替换为 100 类；权重缓存在本项目 `cache\torch`。
- 纯图片训练没有 CLIP、文字描述或物种分类层级输入。物种名称只作为类别映射和显示信息。

## 双击操作

1. `check_environment.cmd`：检查 CUDA，使用预训练模型执行 100 类前向和反向传播。
2. `prepare_metadata.cmd`：下载官方标注，固定 100 个物种并生成数据划分。已有划分保留，不会随机重选。
3. `download_data.cmd`：下载官方图片压缩包，验证 MD5，只解压选定物种的 6,000 张图片。网络中断后可再次运行，保留 `.part` 文件以续传。
4. `train_baseline.cmd`：运行纯图片基线，最多 20 轮，batch size 16，混合精度，验证集连续 5 轮没有改善时提前停止。
5. `evaluate_baseline.cmd`：使用最近一次真实数据训练的最佳模型评估保留测试集。
6. 把图片拖到 `predict_image.cmd`：输出前 5 个候选物种及 softmax 分数。

评估和预测入口会排除 synthetic smoke 模型。如果还没有真实数据模型，会明确提示先训练。测试集用于最终报告；调学习率、轮数和增强策略时只看内部验证集。

## 数据和实验划分

数据源为 [iNaturalist 2021 官方发布](https://github.com/visipedia/inat_comp/tree/master/2021)。选择植物、昆虫、鸟类、哺乳动物、真菌各 20 个物种，每组随机选 10 个属，每属选 2 个物种。固定种子 42；属内配对用于保留细粒度分类难度，但不保证每一对都视觉相似。

| 划分 | 每个物种 | 合计 | 来源 |
|---|---:|---:|---|
| 训练 | 40 张 | 4,000 张 | 官方 train_mini |
| 内部验证 | 10 张 | 1,000 张 | 官方 train_mini 中预先留出 |
| 保留测试 | 10 张 | 1,000 张 | 官方 val，作为本课程项目测试集 |

官方隐藏标签的 public_test 不使用。训练代码不读取测试集图像；选择最佳模型依据内部验证集 macro-F1。划分记录在 `data\manifest.json`，物种清单在 `data\species.csv`，保留原始类别 ID、名称、分类层级及可用图片署名和许可信息。

下载说明：官方 mini 图片包约 42 GB，val 图片包约 8.4 GB，标注共约 54 MB。尽管只使用 100 类，官方提供的是整体 gzip 压缩包，仍需下载整个图片包。只解压本项目所需图片；压缩包保留在 `data\archives` 以便恢复与校验。请预留至少 55 GB 空间。下载和解压时间取决于网络与磁盘性能。

遵守官方数据使用条款：用于非商业研究和教学，不重新分发图片。此处是课程实验划分，不代表参加官方竞赛。

## PowerShell 命令

```powershell
cd E:\ELEC4240\SpeciesRecognition

# 准备标注和类别划分
.\.venv\Scripts\python.exe data_tools.py prepare

# 下载图片并解压选定类别（支持下载续传）
.\.venv\Scripts\python.exe data_tools.py download

# 标准基线；运行名不能重复，避免覆盖已有结果
.\.venv\Scripts\python.exe run.py train --name baseline_seed42 --seed 42 --epochs 20 --device cuda

# 低样本实验：固定验证集、测试集，仅减少训练图像
.\.venv\Scripts\python.exe run.py train --name baseline_10shots_seed42 --samples-per-class 10 --seed 42 --device cuda

# 在决定好设置后，评估保留测试集
.\.venv\Scripts\python.exe run.py evaluate --checkpoint runs\baseline_seed42\best.pt --split test --device cuda

# 单图预测；替换为真实图片路径
.\.venv\Scripts\python.exe run.py predict --checkpoint runs\baseline_seed42\best.pt --image "E:\path\photo.jpg" --device cuda

# 自动检查数据划分与指标实现
.\.venv\Scripts\python.exe -m unittest -v test_project

# 验证完整训练、保存、重载、评估、预测流程；使用合成随机图片
.\.venv\Scripts\python.exe run.py smoke --device cuda --workers 2
```

训练默认对全模型微调：主干学习率 1e-4，分类层 1e-3，AdamW，weight decay 1e-4，余弦学习率调度，交叉熵损失。图片训练增强为随机裁剪和水平翻转；验证和预测采用预训练权重对应的确定性缩放、中心裁剪和归一化。默认输入 224 × 224。

使用 `--samples-per-class 5/10/20/40`（每次选一个数字）和 `--seed 42/43/44` 进行低样本与多随机种子比较，保持数据划分不变。同一种子下的样本子集是嵌套的。未来文字辅助模型应使用相同初始化、图像增强、训练样本和训练预算。

如果显存不足，将 `--batch-size` 改为 8；Windows 数据加载遇到问题可改为 `--workers 0`。同一训练运行不提供中断续训，需另起运行名；图片下载支持续传。自动选择最新真实模型的快捷入口适用于初次操作，多组实验时请用明确的 `--checkpoint` 路径。

## 输出结果

每次训练创建独立 `runs\<运行名>`，包含：

- `best.pt`：验证集 macro-F1 最佳权重、类别映射和配置；未使用测试集选模型。
- `config.json`、`classes.json`、`training_rows.json`：环境、设置、类别及实际训练样本。
- `history.csv`、`learning_curves.png`：训练与验证损失、准确率和 F1。
- `summary.json`：本次训练完成状态。
- 评估后新增 `evaluation_test_<时间>`：Top-1/Top-5 accuracy、macro-F1、逐类指标、混淆矩阵、逐图预测和预测示例图。

混淆矩阵行是真实类别、列是预测类别，顺序对应 `classes.json`。`confusion_matrix.csv` 是原始计数，PNG 按行归一化。图片输入只在这 100 个候选物种中分类，不能自动识别所有物种或可靠拒绝未知物种；softmax 分数也不等同于经过校准的正确概率。

`smoke_*` 使用随机合成图片检验程序链路，其指标不能写入项目报告作为真实识别准确率。`deployment_check.json` 与 `smoke_test_result.json` 记录环境和链路检查结果。

## 在另一台电脑重建环境

先安装 Python 3.12 和支持 CUDA 12.8 的 NVIDIA 驱动；在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3" "Pillow>=10,<13" "matplotlib>=3.8,<4"
.\.venv\Scripts\python.exe run.py doctor
```

跨机器移动数据时，manifest 中的绝对 `image_root` 需要调整。评估检查 manifest 哈希以防混用数据，因此应在新位置生成划分后重新训练，或在明确核对路径以外内容相同后另行增加迁移支持。当前部署以本机 E 盘路径为准。

## 参考

- [EfficientNet 原论文](https://proceedings.mlr.press/v97/tan19a.html)
- [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- [iNaturalist 2021 数据、标注及条款](https://github.com/visipedia/inat_comp/tree/master/2021)
- [PyTorch 安装说明](https://pytorch.org/get-started/locally/)
