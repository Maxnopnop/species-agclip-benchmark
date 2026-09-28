# Species recognition: five backbones with attribute-guided adaptation

**CUB 人工属性诊断：** [中文报告](reports/cub_attributes_v1/summary_zh.md)、[运行与协议说明](docs/cub_attributes_v1.md)。采用官方 CUB 逐图属性／可见部位标注，固定 20 个物种和 24 条属性，对比自动属性、人工标注、打乱监督及额外区域监督，并用冻结特征线性探针单独检验属性可学习性。分类最优与属性最优检查点分别锁定；真实测试属性和部位不作为输入。入口 `cub_models.cmd`，单图入口 `predict_cub.py`。完整状态、成绩及第 0 步选择说明见该版本报告。

**本机新起步配置（grounded_v1）：** [部署与运行说明](docs/grounded_starter.md)、[中文实测报告](reports/grounded_v1/summary_zh.md)、[对比表](reports/grounded_v1/comparison.csv)。CLIP ViT-B/32＋OWL-ViT 实际区域＋独立属性视觉分支＋CAF，两个视觉分支都训练末端两层。训练图片由 200 增至 300；保留原始 CLIP、普通微调、区域无文字和打乱属性对照。`grounded_models.cmd` 运行完整流程，`predict_grounded.cmd` 可接受拖入的图片。用户只输入图片，内部仍使用固定文字库和检测器；它不是 CoCa 论文的严格复现。训练与最终评估状态、局限及准确率以该版本报告为准。

**新增类别隔离零样本实验：** [中文结果](reports/zsl_v1/Summary_CN.md)、[ZSL/GZSL 完整报告](reports/zsl_v1/README.md)、[运行说明](docs/zsl_v1.md)。原五种模型重新初始化任务适配模块，使用互不重叠的 10 个训练物种、4 个开发物种和 6 个最终评估物种。类别名称与文字属性可提前提供，最终未见物种图片不参与训练或参数选择。此轮复用已有图片，属于探索性 AG 改编，不是预训练无重叠保证或原论文严格复现。

**最新图片级属性实验：** [中文结果](reports/visible_v1/Summary_CN.md)、[完整指标](reports/visible_v1/README.md)、[复现说明](docs/visible_v1.md)。CLIP ViT-B/16 与 SigLIP 2 使用可见属性监督和区域对齐，另用 FG-CLIP 做小规模复核。分别评价属性识别、定位和物种分类；不把属性指标的改善等同于分类提升。该轮属于验证集探索，保留全图基线、匹配容量对照和错误监督对照。

课程实验项目，主体位于 `E:\ELEC4240\SpeciesRecognition`。以少量有标签图片将四种 ImageNet 视觉模型对齐到 CLIP 文字空间，再比较五个模型的属性融合效果。

**已完成 20 物种、1,000 张真实图片的扩展实验，共 225 组比较。** [完整结果](reports/expanded20/README.md)、[对比图](reports/expanded20/comparison.png)、[实验定义和运行方法](docs/expanded20.md)。5-shot 基线准确率约 60%–75%；AG 改编带来小幅且不一致的变化，15 个“模型 × 样本数”设置中，没有正向提升通过多重比较校正。不能据此声称 AG 显著有效。正式 100 类实验仍待完整数据和来源核查属性库准备完毕。

**后续诊断已完成：** [96 组验证集诊断](reports/diagnostics_v1/README.md)及[运行说明](docs/diagnostics_v1.md)。EfficientNet 的已有 AG 模型移除或打乱融合文字后，验证集预测标签未变；CLIP 变化也较小。匹配架构重新训练后，正常、打乱和移除属性仍基本持平。微调视觉主干末端能带来部分验证集提升，但未显示明确的属性语义增益。该轮只使用训练与验证集，不是新测试集成绩。

## 实验定义

新增 [20 物种扩展实验](docs/expanded20.md)：1,000 张真实图片、58 条来源支持的属性、三种 AG 方案及局部图像无文字对照，共 225 组预设比较。入口为 `expanded_models.cmd`；原 4 类 pilot 保留。扩展实验修正了非零门控、有效训练更新步数与验证指标持平时的检查点选择问题，并将 AG 版本选择锁定在测试之前。

扩展实验比较 `baseline`、`region_only`、`ag_mean`、`ag_attention`、`ag_aux`，使用五个固定重叠裁剪区域；并未使用 OWL-ViT 定位。每个模型分别运行 5/10/20-shot、三个种子。新增五项协议与统计测试通过；单图推理与缓存评估的一致性抽查见 [验证记录](reports/expanded20/verification.json)。用户只输入图片，固定属性向量保存在检查点内部。

## 早期四物种 pilot（保留记录）

下列三变体、OWL-ViT 和 `benchmark.py` 说明适用于旧 pilot，不能与上面的扩展实验混为同一个协议。[旧结果](reports/pilot/README.md)共 15 组比较，AG 的 Top-1 与基线持平。

| Backbone | Initial weights | Stage 1 | Stage 2 comparisons |
|---|---|---|---|
| EfficientNet-B0 | ImageNet-1K V1 | Train visual-to-text projection | Baseline / average / AG-CLIP adaptation |
| ResNet-18 | ImageNet-1K V1 | Train visual-to-text projection | Same |
| ConvNeXt-Tiny | ImageNet-1K V1 | Train visual-to-text projection | Same |
| ViT-B/16 | ImageNet-1K V1 | Train visual-to-text projection | Same |
| CLIP ViT-B/32 | OpenAI CLIP | Train residual visual adapter | Same |

- 共享冻结的 OpenAI CLIP ViT-B/32 文字编码器。第一阶段以图像和正确类别名称提示构造分类交叉熵，学习图文匹配分数。它是**有监督小样本文字空间适配**，不是从头训练通用 CLIP，也不能由此声称掌握任意文本检索或未见物种零样本能力。
- 初始实验冻结五个视觉主干，缓存全图与局部区域特征；训练投影/残差适配层及属性融合层。未进行全主干微调或随机图像增强，便于在 8 GB 显存上完成公平的第一轮验证。
- AG-CLIP 改编：固定视觉属性描述 → OWL-ViT 区域定位 → 局部视觉特征与属性文字编码 → Cross-Attention Fusion → 类别文字相似度。属于论文思路的**自定义实现**，不是官方代码或原论文的严格复现；未实现论文完整的 CoCa 与 LLM 属性挖掘流程。
- 三个变体均从同一个第一阶段检查点开始，并获得相同的额外训练轮数：`baseline` 继续训练对齐层；`average` 学习属性编码并平均融合；`agclip` 学习属性编码与交叉注意力融合。平均融合也有可训练的属性编码器。
- 每张图均使用同一份属性提示集合，不按测试图片真实类别选择提示。验证集 macro-F1 选择检查点；测试集只报告结果。无区域检出时退回全图特征。
- **用户预测时只需提供图片，但 AG 分支内部仍使用固定属性文字库与定位器。**这不同于“文字仅训练使用、部署完全不需要属性模块”的早期方案。

## 两种数据规模

**真实图片 pilot**：从官方 iNaturalist 2021 压缩包已下载部分中读取完整 JPEG，4 个物种共 60 张，每类 5 训练 / 5 验证 / 5 留出测试，固定种子。包括昆虫、哺乳动物和两种植物。它不代表完整物种分布，也不是正式的 100 物种结果。单张图片经过解码与 SHA256 记录；部分压缩包尚不能通过完整档案 MD5 校验。四类实验中 Top-5 恒为 100%，不用于评价。

**正式方案**：100 物种，植物、昆虫、鸟类、哺乳动物、真菌各 20 类；每组 10 个属，每属 2 类。每类 40 张候选训练 / 10 验证 / 10 测试，共 6,000 张。训练与内部验证来自官方 train_mini，测试来自官方 val。`configs/benchmark.json` 定义 5/10/20-shot × 3 seeds × 5 backbones × 3 variants，共 135 组最终比较。

官方图片包需要整体下载（train_mini 约 42 GB + val 约 8.4 GB）。目前下载与 pilot 可同时进行，只解压所需图片。主数据下载未完成时不能运行正式矩阵。

`configs/pilot_attributes.json` 包含四个 pilot 物种的来源链接和经来源核对的 15 条视觉属性。它不是生物学专家审核，也不保证属性在每张图片中可见。`configs/attributes.json` 是早期通用调试词表，不应冒充已核查的 100 物种属性集。正式矩阵入口会要求另行准备覆盖全部 100 类的来源核对属性 JSON；不能直接使用 pilot 词表。

## 在本机运行

双击 `check_progress.cmd` 查看进度；双击 `pilot_models.cmd` 运行或重启真实 pilot。首次需要下载 CLIP、OWL-ViT 和视觉主干权重，所有缓存位于 E 盘本项目 `cache`。已完成实验会保留；未完成模型会从已保存的第一阶段检查点重新运行比较阶段。不要同时启动两个相同工作流。

```powershell
cd E:\ELEC4240\SpeciesRecognition
# 官方完整数据：已经有后台下载时不要重复启动
.\.venv\Scripts\python.exe data_tools.py download
# 已下载前缀至少包含 4 个完整类别后，准备独立 pilot
.\.venv\Scripts\python.exe make_pilot.py
# 顺序处理五个模型（默认每类 5 张，第一阶段 10 轮、比较阶段各 10 轮）
.\.venv\Scripts\python.exe benchmark.py all
# 仅重跑/补齐训练，无需重新定位和提取图像特征
.\.venv\Scripts\python.exe benchmark.py train
# 导出表格与图表，不复制数据图片或模型权重
.\.venv\Scripts\python.exe report_results.py
# 对一张新图片运行已训练的 AG 模型（替换图片路径）
.\.venv\Scripts\python.exe predict_multimodal.py --checkpoint runs\pilot_multimodal\efficientnet_b0_shots5_seed42\agclip\best.pt --attributes configs\pilot_attributes.json --image E:\path\photo.jpg
# 正式 100 类数据与已核查属性均准备完毕后
.\.venv\Scripts\python.exe run_matrix.py --attributes configs\main_attributes.json
# 自动验证数据泄漏保护、断点下载、五个真实架构的前向传播、梯度及检查点重载
.\.venv\Scripts\python.exe -m unittest -v test_project test_multimodal test_experiment
```

`benchmark.py --help` 可指定模型、样本数、随机种子、轮数及独立输出目录。改变实验设置时指定新的 `--output`；改变数据或属性时还要指定新的 `--cache`。缓存检查元数据与属性文件哈希，禁止静默混用。

## 结果文件

- `runs/pilot_multimodal/status.json`：当前阶段或失败原因。
- `runs/pilot_multimodal/comparison.csv`：五模型 × 三变体汇总。
- 每个模型目录中的 `alignment/best.pt`：第一阶段适配层；各变体包含 `best.pt`、`history.csv`、`metrics.json`、`predictions.json`、`confusion_matrix.csv`。
- `reports/pilot/`：可上传的汇总报告与图表；实际生成后才存在。
- 原来的纯图片 EfficientNet 分类实验保留在 `run.py`，说明见 [legacy guide](docs/legacy_efficientnet.md)。其全模型微调训练预算与当前冻结主干方案不同，不能直接作为控制条件归因提升。

对比 Top-1、macro-F1，并在正式多种子结果中报告均值及标准差。先检查是否改善，不预设 AG 必然有效。记录的比较训练耗时不含检测器、特征提取与下载，不能当作端到端推理耗时。预训练数据是否与 iNaturalist 图片重叠无法由此实验排除。

## 环境与重建

本机 Python 3.12、PyTorch 2.11.0+cu128、Torchvision 0.26.0+cu128，RTX 5060 Laptop 8 GB。项目 venv 复用 `E:\conda-envs\comp4471` 已有底层依赖；新装 OpenCLIP、Transformers 等在项目 venv 内，不修改原课程环境。

在另一台 Windows CUDA 机器上建立独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe data_tools.py prepare
```

需将本机的 `data/manifest.json`、`data/pilot/manifest.json` 中 `image_root` 改为新位置；不要搬走旧 venv。CPU 仅适合程序检查或较慢的试验，可给 benchmark 加 `--device cpu`。

## 数据、权重与引用

数据仅用于课程教学与非商业研究，遵循官方及单张图片许可，保留元数据署名。Git 排除 `data/`、`cache/`、`runs/`、虚拟环境、密钥和权重；不重新分发图片。代码使用 PyTorch、Torchvision、OpenCLIP 与 Transformers 的公开 API。

- [iNaturalist 2021 official release](https://github.com/visipedia/inat_comp/tree/master/2021)
- [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html) and [OpenCLIP implementation](https://github.com/mlfoundations/open_clip)
- [Google OWL-ViT model](https://huggingface.co/google/owlvit-base-patch32)
- [AG-CLIP: Attribute-Guided CLIP for Zero-Shot Fine-Grained Recognition](https://doi.org/10.1109/OJCS.2026.3654171)
- Pilot attribute sources are recorded individually in `configs/pilot_attributes.json`.
