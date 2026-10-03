# CoCa ViT-L/14：AG-CLIP小规模近似复现

同主干、同数据、同预算下，CAF属性方案相对经过同预算微调的CoCa基线，未见类别ZSL均值变化为-0.75个百分点。这是两个种子的探索性结果，不是原论文性能复现或显著性结论。

## 已实现的流程

实际部署CoCa ViT-L/14，使用公开LAION-2B权重，2.55GB下载已核对发布方LFS SHA-256。使用原生768维图文空间。前23层视觉编码器和前11层文字编码器冻结并缓存输出，训练最后一层及池化/归一化/投影、属性MLP和CAF。图像和区域共享视觉编码器。

缓存末层输入的实现已与原生前向核验；小测试图像/文字向量最大误差均为0。两个更新后视觉末层、文字末层、属性MLP和CAF均有非零梯度。属性与CAF输出初始化为0，使各分支起步均保留原生全图输出；因此CAF内部第一步梯度为0是预期行为，第二步已验证非零。

## 数据与选择规则

25个此前项目未用CUB物种、700张图片：10个已见类各20张训练+10张验证+10张最终已见测试；5个开发未见类各20张；10个最终未见类各20张。三类物种互不重叠。训练图片来自官方训练部分，其余来自官方测试部分。按元数据及固定随机顺序选样，不按预测结果选择。

每批10张，已见类各一张，使用类别名文字配对的双向图文对比损失；每个变体80次更新，两个种子。视觉/文字末层学习率1e-5、新增模块1e-4。按开发GZSL H、再按开发ZSL、再优先较早步选模型；允许第0步，避免强行使用变差的微调结果。全部检查点封存后才处理最终测试图片。

Step0 selected: 4/10. Histories,selected steps and parameter-change norms: training_diagnostics.json.

## 最终结果

| Method | Unseen-only ZSL (%) | Seen S (%) | Unseen U (%) | GZSL H (%) | Selected steps |
|---|---:|---:|---:|---:|---|
| Native CoCa,no adaptation | 97.00 | 87.00 | 95.50 | 91.05 | 0 |
| baseline | 97.25 ± 0.35 | 91.00 | 94.50 | 92.72 | [40, 80] |
| attributes | 97.50 ± 0.71 | 91.00 | 95.00 | 92.96 | [40, 60] |
| caf | 96.50 ± 0.71 | 89.50 | 93.75 | 91.53 | [0, 60] |
| confidence | 95.75 ± 1.77 | 92.00 | 90.25 | 90.83 | [20, 0] |
| regions | 97.00 ± 0.00 | 87.00 | 95.50 | 91.05 | [0, 0] |

ZSL只在10个最终未见类中选择；GZSL在10个已见+10个最终未见类中选择。S/U按类别平均，H逐次运行计算后再取均值。ZSL的±是两个种子的样本标准差，不是置信区间。原生基线仅评估一次。attributes=属性分支无CAF；caf=属性+CAF；confidence=CAF再加检测置信度权重；regions=CAF结构相同但属性文字向量置零。

## 与原论文的区别

| Item | Paper | This pilot |
|---|---|---|
| Backbone | CoCa ViT-L/14 | Same architecture;public LAION-2B checkpoint,exact paper checkpoint unspecified |
| CUB protocol | 150 seen / 50 unseen | 10 seen / 5 development unseen / 10 final unseen |
| Encoder updates | Image/text encoder fine-tuning | Last visual/text block + pooling/projection only |
| Training | 50 epochs,batch64 | 80 updates,batch10,two seeds |
| Attributes | GPT-4o mining/filtering | Fixed24 shared bird prompts;no test per-image attribute labels |
| Grounding | OWL-ViT top-K | OWL-ViT top2,threshold0.05 |
| Fusion | Attribute aggregation + CAF | Explicit shared encoder,average-before-MLP,custom256dim CAF and zero residual initialization |

Paper Table4 reports CoCa65.4% -> AG without CAF69.0% -> AG with CAF73.3% on CUB50 unseen classes (+7.9pp total). Plant disease70.2% ->78.8% ->84.6% (+14.4pp). [Author paper](https://www.researchgate.net/publication/399822501_AG-CLIP_Attribute-Guided_CLIP_for_Zero-Shot_Fine-Grained_Recognition).

论文聚合/CAF的相加与拼接、共享或独立视觉编码器存在描述歧义，本实现作了明确选择。不能把本轮更小候选类别集合的绝对准确率和论文73.3%直接比较，也不能把置信度加权当作论文已验证的增益。

## 局限与复现

新类别仅相对本项目历史实验未用；CoCa/OWL-ViT预训练是否见过这些CUB图片未排除。属性定位未经人工核验。少量样本、两个种子和很短预算仅能检验可运行性及初步方向；没有提升不证明完整论文无效，有提升也不证明完整复现。测试集已查看，后续调参不可继续称其独立盲测。

```powershell
.\.venv\Scripts\python.exe coca_download.py
.\.venv\Scripts\python.exe -m unittest test_coca_pilot -v
.\.venv\Scripts\python.exe coca_experiment.py all
.\.venv\Scripts\python.exe report_coca_pilot.py
```

Preparation depends on local official CUB images and historical manifests/fingerprints. Weights,photos and intermediate prefixes remain local;GitHub contains code/configurations/split identities and reports only.
