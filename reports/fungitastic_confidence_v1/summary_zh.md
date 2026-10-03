# FungiTastic：按年份隔离的新数据验证

500张照片、500条独立观察记录、10个物种。2022年200张训练、100张验证；2023年200张测试。每类20/10/20张，所有随机种子共享划分。仅按数量和固定随机顺序选择物种，不按结果选样。

以下为三个种子的准确率均值±标准差。两主干冻结，原始Head/train不变，每阶段200步，七分支配对。CLIP纯视觉仍含原有图文预训练；本轮图文适配不等于从零预训练CLIP，AG也是轻量改编。

| Method | EfficientNet-B0 (%) | CLIP ViT-B/32 (%) |
|---|---:|---:|
| visual | 64.67 ± 2.93 | 62.50 ± 2.29 |
| random_codes | 73.17 ± 1.04 | 70.00 ± 1.00 |
| text | 72.17 ± 0.76 | 74.17 ± 1.26 |
| region_only | 71.17 ± 1.04 | 74.83 ± 0.76 |
| ag_uniform | 71.67 ± 1.44 | 74.50 ± 1.73 |
| ag_confidence | 71.67 ± 1.44 | 74.83 ± 0.58 |
| ag_shuffled | 71.50 ± 1.32 | 75.00 ± 0.50 |

本轮纯视觉准确率仅约63%–65%，仍未得到置信度加权的显著正向证据。EfficientNet加权与等权均值相同；CLIP加权均值仅高0.33个百分点，低于打乱权重，且与区域无文字对照相同。因此不能把文字适配相对线性头的提升归因于AG，也不能归因于置信度排序。随机类别向量也带来大幅提升，说明分类头/优化差异是重要混杂因素。

测试图片177/200具有两个区域，因此本轮收益不足不能简单归因于缺少可加权区域。固定加权模型中切换为等权或反序权重，EfficientNet三个种子的预测均未改变，CLIP最多改变3/200张；这提示当前模型对权重排序的决策依赖有限，但不证明属性普遍无效。

种子均值和概率集成是不同汇总：例如CLIP加权相对等权的种子均值为+0.33个百分点，但概率集成为−1.50个百分点。不能只选取较有利的汇总来声称提升。

## 置信度加权的配对证据

| Backbone | Reference | Ensemble delta (pp) | Fixes / breaks | 95% CI (pp) | Holm p |
|---|---|---:|---:|---|---:|
| efficientnet_b0 | ag_uniform | +0.50 | 1/0 | [0.0, 1.5] | 1.0000 |
| efficientnet_b0 | text | +0.50 | 2/1 | [-1.0, 2.0] | 1.0000 |
| efficientnet_b0 | ag_shuffled | +0.00 | 0/0 | [0.0, 0.0] | 1.0000 |
| clip_vit_b32 | ag_uniform | -1.50 | 4/7 | [-4.500000000000001, 1.5] | 1.0000 |
| clip_vit_b32 | text | +1.00 | 6/4 | [-2.0, 4.0] | 1.0000 |
| clip_vit_b32 | ag_shuffled | -2.00 | 0/4 | [-4.0, -0.5] | 0.7500 |

六项预设检验中，校正后显著正向对比数量：0.

检验使用三种子概率集成的200张独立观察图片，而不是把三种子当作600张独立照片。使用精确McNemar检验和5000次按类别分层自助区间。区间未作多重校正，且条件于固定模型；[0,0]不证明总体效应为零。

## 检测与权重利用

| Split | Zero boxes | One box | Two boxes | Mean larger weight |
|---|---:|---:|---:|---:|
| train | 2 | 11 | 187 | 0.679640531539917 |
| val | 3 | 7 | 90 | 0.6819465160369873 |
| test | 8 | 15 | 177 | 0.6737488508224487 |

固定30条蘑菇可见属性提示，所有图片共用；不读取图片自带caption、坐标或测试属性。OWL-ViT最多取两个区域，阈值0.05。置信度不是经校准的属性正确概率。固定加权检查点替换为等权/打乱权重的诊断在verification.json；未用于调参。

## 数据保护与结论边界

Historical images screened: 8742. Minimum pHash/dHash distances: 12/10;threshold4. Zero exact/pixel/perceptual-threshold matches after filtering. Observation IDs are disjoint across splits. Photographer identities are unavailable for grouping.

先锁定清单、代码和配置，再提取训练/验证特征。42个检查点按验证macro-F1/CE选定并封存后才运行测试检测与特征提取；42个模型重放已核验。未按测试结果修改模型。

2022/2023是元数据中的观察日期，晚于2021年发布的OpenAI CLIP及ImageNet1K数据；这加强了针对旧主干的时间隔离，但不是独立验证过的拍摄日期证明。OWL-ViT预训练重叠未审计，不能宣称整个系统从未见过图片；物种概念也可能早已见过。

这是自定义闭集时间划分，不是未见类别零样本实验，也不是官方完整基准。500张小样本、短训练预算及未人工核验的属性定位限制推广；新数据不能保证不过拟合。若后续继续改方法，本测试集也已被看过。跨鸟类/真菌数据集的绝对准确率不能直接比较为算法收益。

Sources: [official dataset](https://github.com/BohemianVRA/FungiTastic), [author Kaggle mirror](https://www.kaggle.com/datasets/picekl/fungitastic), [CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md).

```powershell
.\.venv\Scripts\python.exe -m unittest test_fungitastic test_confidence test_statistics -v
.\.venv\Scripts\python.exe fungitastic_experiment.py all
.\.venv\Scripts\python.exe report_fungitastic.py
```

Photos,feature caches and weights remain local on E:. Only code,configurations and aggregated reports are published. Preparation currently uses historical local fingerprints for project-overlap screening;see data_audit.json.
