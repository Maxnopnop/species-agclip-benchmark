"""Create the requested feasibility summary after the current matrix finishes."""
import json
from datetime import datetime
from data_tools import ROOT,write_json


def main():
    repeated=json.loads((ROOT/'reports/cub_resplit_v1/summary.json').read_text());assert len(repeated['rows'])==6
    c=json.loads((ROOT/'configs/cub_resplit_v1.json').read_text());cells=0
    for backbone in c['backbones']:
        for split in c['class_partition_seeds']:
            folder=ROOT/f'runs/cub_resplit_{backbone}_{split}';lock=json.loads((folder/'selection_locked.json').read_text());assert len(lock['selected'])==5
            records=list(folder.glob('*/result.json'));assert len(records)==10;cells+=len(records)
            for row in lock['selected']:assert (ROOT/row['checkpoint']).with_name('final_predictions.pt').is_file()
    rows=[]
    for version,name in [('cub100_v1','CLIP ViT-B/16'),('cub_siglip_v2','SigLIP 2 Base/16')]:
        s=json.loads((ROOT/f'reports/{version}/summary.json').read_text());by={r['variant']:r for r in s['rows']};rows.append(f'| {name} | {s["native"]["H"]:.2f}% | {by["tokens"]["H_mean"]:.2f}% | {by["region_only"]["H_mean"]:.2f}% | {by["tokens"]["H_mean"]-by["region_only"]["H_mean"]:+.2f} pp |')
    repeat_lines=[f'- **{r["backbone"]}**：三个新划分中，token 相对无文字对照的平均差值为 **{r["mean_token_vs_region_H"]:+.3f} 个百分点**；相对常量属性对照为 {r["mean_token_vs_constant_H"]:+.3f} 个百分点；{r["token_step_zero_count"]}/3 个划分选中第 0 步，即没有验证到适配收益。' for r in repeated['descriptive_means']]
    text='''# 当前模型的可行性与暂停结论

**当前这轮更换类别划分的 60 组训练和最终评估已完成，研究按用户要求暂停。没有启动下一轮实验。**

## 总体判断

**作为本机可运行的物种识别系统：可行。作为“AG 能显著提高准确率”的研究方案：目前证据不足。** 系统已能使用公开数据完成训练、类别隔离评价和只输入图片的预测；但辅助属性识别的改善没有稳定转化为强于匹配对照的物种分类提升。

目前测试中，修正预处理后的 SigLIP 2 是更强的原始识别模型。它优于 CLIP B/16 的分数，是主干／预训练／原生预处理的综合效果，不是 AG 的效果。

## 同一 100 物种协议的结果

本轮数据为 CUB 鸟类，100 个鸟类物种；并非已经验证了跨动物、植物的通用物种识别。每次使用 50 个已见训练类、25 个开发未见类、25 个最终未见类。输入预测图片时不提供该图片的文字标签、真实属性或部位坐标；模型内部保留固定类别／属性文字向量。

H 衡量同时识别已见与未见物种的平衡表现，是两组宏平均准确率的调和平均，不能直接当作所有图片的普通 accuracy。下面的融合分数是同一类别划分下三个融合训练种子的均值。

| 主干 | 原始 H | 独立属性 token H | 无文字对照 H | AG 与无文字的差值 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

这两个主干的 AG 与无文字差值的配对条件区间均覆盖 0。常量属性、错误关联或平均融合的结果也相近。因此，不能将相对于原始模型约 1 个百分点的提升全部归因于属性语义。

## 刚完成的更换类别划分复测

重新划分哪 50 个物种参与训练、哪 25 个用于开发、哪 25 个留作最终评价。每次重新确定训练集支持的属性、训练探针和融合模块，两个主干共 60 组训练。每个划分固定一个融合种子，区别于上表的三个融合种子；六个组合全部锁定选择后再统一最终评分。

'''+ '\n'.join(repeat_lines)+'''

这些划分复用了同一批 100 个物种与部分图片，均值仅作稳健性描述，不当作三个完全独立数据集的显著性证据。完整逐划分成绩见 [类别划分复测](../cub_resplit_v1/summary_zh.md)。

## 为什么部署可行，但 AG 收益尚未证实

1. **工程链路已经实现。** 本机完成数据准备、OWL-ViT 区域检测、属性探针训练、融合对照和单图推理。原 100 类 CLIP、区域微调模型与修正版 SigLIP 都通过了从真实照片重新计算与缓存结果一致的检查。没有要求用户另拍照片或采购训练服务。
2. **当前规模适合本机。** 冻结大部分预训练参数、缓存特征并训练较小模块，已在本机 GPU 上实际完成。这个证据覆盖当前 100 类、每个训练类 10 张图片的设置，不等于完整大模型端到端复现的资源需求也相同。
3. **属性确实能学到，但迁移有限。** 比较相同属性集合，未见物种的属性 mAP 明显低于已见物种；多次提升属性指标后，分类仍与无文字／常量属性对照接近。属性损失在运行，不代表它提供了额外的分类信息。
4. **明显的实现与简单优化方向已做对照。** 独立 token、属性瓶颈、覆盖扩展、稀疏选择、对比损失、标签确定度、区域末端微调、公开描述、对应表、正则化、模型更换和类别划分均有记录。真实部位点裁剪也未在当前融合机制中激活独特收益。这不能排除所有其他 AG 方法。
5. **发现的 SigLIP 缺陷已修复。** 快速分词器没有自动小写物种名，导致错误的低基线。修正后重新训练全部对照，缺陷版本排除科学比较。不能用那个错误低基线证明 AG 大幅改善。

## 对课程项目的建议

如果目标是**搭建一个可以演示和复现的图像物种识别系统，并系统评估属性监督**，目前方向是可行的，代码、数据协议、失败对照与分析都已具备。若 proposal 的核心承诺是 **AG 将显著优于原模型**，目前不宜这样写。

更符合现有证据的研究问题是：*Can attribute-guided supervision improve generalization to unseen species beyond a strong vision-language baseline, and under what conditions does it fail?* 结果可以报告属性可学习性、跨物种迁移差距及分类增益的边界，而不是预先保证正提升。是否满足最终评分标准仍应以课程要求为准。

当前实现应标为 **AG-CLIP-inspired adaptation**：冻结或局部微调视觉编码器、自动区域与属性监督／融合，并非原论文完整 CoCa、区域属性编码器与大规模训练复现。未来若继续，需新的可检验机制、可靠的判别属性或独立数据验证；目前不能保证再扩大样本或更换模型就会产生显著 AG 收益。

“未见物种”指该次项目训练未使用这些类别的图片，不代表基础模型预训练绝对没有见过相关物种或数据。

详细诊断与原始报告见 [综合诊断](diagnostic_update_zh.md)、[CLIP 100 类](../cub100_v1/summary_zh.md)、[SigLIP 修正版](../cub_siglip_v2/summary_zh.md)。
'''
    out=ROOT/'reports/cub_followup_v1';(out/'feasibility_zh.md').write_text(text,encoding='utf-8')
    write_json(out/'research_status.json',dict(status='paused',reason='User explicitly requested pause after the current class-partition test and a feasibility-focused summary.',current_test='cub_resplit_v1',current_test_training_cells=cells,current_test_final_evaluation_complete=True,valid_fusion_cells_this_followup=246+cells,privileged_development_fusion_cells=30,excluded_defective_siglip_cells=30,probe_regularization_cells=12,recorded_local_time=datetime.now().isoformat(),next_experiments_started=False))
    print('Current test complete; feasibility report saved; research paused.')


if __name__=='__main__':main()
