"""Report supervision quality separately from species classification utility."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json
from multimodal.cub_data import OUT,REPORT,prepare
from multimodal.cub_experiment import provenance
from report_grounded import bootstrap_h_difference

NAMES={'native':'Original CLIP','finetune':'Fine-tuning','region_only':'Regions, no attributes','automatic':'Automatic supervision','gold':'Human attributes','shuffled':'Shuffled attributes','gold_region':'Human + region supervision'}

def main():
    p,m=prepare();result=json.loads((REPORT/'final_results.json').read_text(encoding='utf-8'));assert result['provenance']==provenance();rows=result['classification_selected'];attr_rows=result['attribute_selected'];lookup={r['variant']:r for r in rows}
    checks=json.loads((REPORT/'verification.json').read_text(encoding='utf-8'));deployment=json.loads((REPORT/'deployment_verification.json').read_text(encoding='utf-8'));assert checks['status']==deployment['status']=='passed' and len(deployment['checks'])==8
    assert checks['provenance']==deployment['provenance']==provenance()
    coverage=json.loads((REPORT/'coverage.json').read_text(encoding='utf-8'));probe=json.loads((REPORT/'attribute_probe_final.json').read_text(encoding='utf-8'))
    diagnosis=json.loads((REPORT/'posthoc_diagnosis.json').read_text(encoding='utf-8'))
    contrasts=json.loads((REPORT/'attribute_selected_contrasts.json').read_text(encoding='utf-8'));contrasts={r['comparison']:r for r in contrasts['comparisons']}
    comparisons=[]
    for a,b in [('gold','finetune'),('gold','region_only'),('gold','automatic'),('gold','shuffled'),('gold_region','gold')]:
        aa=torch.load((ROOT/lookup[a]['checkpoint']).parent/'final_predictions.pt',weights_only=True);bb=torch.load((ROOT/lookup[b]['checkpoint']).parent/'final_predictions.pt',weights_only=True)
        comparisons.append(dict(comparison=a+' minus '+b,H_difference=lookup[a]['metrics']['H']-lookup[b]['metrics']['H'],H_bootstrap_95pct=bootstrap_h_difference(aa,bb,p),attribute_map_difference_pp=100*(lookup[a]['metrics']['attribute_map']-lookup[b]['metrics']['attribute_map'])))
    write_json(REPORT/'paired_comparisons.json',dict(comparisons=comparisons,scope='Paired within-class image bootstrap, 2000 replicates. Conditional on one training seed and fixed classes; exploratory comparisons without multiplicity adjustment.'))
    with (REPORT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['variant','selection','step','ZSL','U','S','H','attribute_mAP','unseen_attribute_mAP','regional_attribute_mAP'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for selection,records in [('classification',rows),('attribute',attr_rows)]:
            for r in records:
                v=r['metrics'];writer.writerow(dict(variant=r['variant'],selection=selection,step=r['step'],**{k:v[k] for k in ['ZSL','U','S','H']},attribute_mAP=100*v['attribute_map'],unseen_attribute_mAP=100*v['unseen_attribute_map'],regional_attribute_mAP='' if r['variant'] in ['native','finetune'] else 100*v['regional_attribute_map']))
    def table(records):
        lines=['| Variant | Selected update | ZSL | GZSL H | Attribute mAP | Unseen attribute mAP |','|---|---:|---:|---:|---:|---:|']
        for r in records:
            v=r['metrics'];lines.append(f'| {NAMES[r["variant"]]} | {r["step"]} | {v["ZSL"]:.2f} | {v["H"]:.2f} | {100*v["attribute_map"]:.2f} | {100*v["unseen_attribute_map"]:.2f} |')
        return '\n'.join(lines)
    gold=lookup['gold']['metrics'];fine=lookup['finetune']['metrics'];regional=lookup['gold_region']['metrics'];auto=lookup['automatic']['metrics'];shuffled=lookup['shuffled']['metrics'];pair=comparisons[0]
    best_probe=100*probe['linear_probe']['attribute_map'];base_probe=100*probe['native']['attribute_map'];configs=[json.loads(f.read_text(encoding='utf-8')) for f in OUT.glob('*/result.json')];assert len(configs)==12
    with (REPORT/'development_runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','lr','classification_step','development_H','attribute_step','development_attribute_mAP','seconds','peak_allocated_MiB']);writer.writeheader()
        for r in configs:writer.writerow(dict(variant=r['config']['variant'],lr=r['config']['lr'],classification_step=r['best_step'],development_H=r['development']['H'],attribute_step=r['attribute_best_step'],development_attribute_mAP=100*r['attribute_development']['attribute_map'],seconds=r['seconds'],peak_allocated_MiB=r['peak_allocated_mib']))
    diagnostic_text='\n'.join(f'- {NAMES[r["variant"]]}（属性最优第 {r["selected_step"]} 步）：清零融合文字改变 {r["interventions"]["zero_text"]["changed_gzsl_predictions"]}/320 个分类，错配文字改变 {r["interventions"]["permuted"]["changed_gzsl_predictions"]}/320 个分类；有效区域的聚合属性文字向量平均两两余弦相似度 {r["mean_pairwise_attribute_text_cosine"]:.4f}，融合门控 tanh 值 {r["fusion_gate"]:.4f}。' for r in diagnosis['records'])
    learning=f'冻结原始 CLIP 特征，仅训练一个线性属性预测头后，最终属性 mAP 从 {base_probe:.2f} 提高到 {best_probe:.2f}。这说明这些属性在当前特征中具有可学习信号；它不是物种分类或 AG 的提升。'
    conclusion=f'以分类指标选择的检查点中，可靠属性组相对普通微调的 H 差值为 {gold["H"]-fine["H"]:+.2f} 个百分点，图像重采样 95% 区间 [{pair["H_bootstrap_95pct"][0]:+.2f}, {pair["H_bootstrap_95pct"][1]:+.2f}]。相对自动监督，属性 mAP 差值 {100*(gold["attribute_map"]-auto["attribute_map"]):+.2f}；相对打乱监督，属性 mAP 差值 {100*(gold["attribute_map"]-shuffled["attribute_map"]):+.2f}。'
    if lookup['gold']['step']==0:conclusion+=' 可靠属性组按分类指标选择了第 0 步（尚未接受属性训练），因此这项主表差值不能算作属性监督带来的分类收益。应结合下面已训练的属性最优检查点判断训练效果。'
    elif pair['H_bootstrap_95pct'][0]<=0:conclusion+=' 当前结果未建立可靠属性相对普通微调的稳定分类优势。'
    else:conclusion+=' 当前固定种子、固定类别中出现正向分类信号，仍需独立随机种子及更大评估复核，不能直接宣称 AG-CLIP 普遍显著有效。'
    md=f'''# CUB 可靠属性监督诊断

本轮回答两个不同问题：模型能否学会逐图可见属性，以及学会属性是否带来额外物种分类收益。数据集改变后，与之前 iNaturalist 的数值不能直接进行算法优劣比较。

## 数据与监督

官方 CUB-200-2011 完整压缩包已下载至 E 盘并通过官方 MD5 校验。20 个物种来自 10 组共同英文名称后缀，每组随机选两类；该分组是固定抽样规则，不宣称严格属级分类。10 个已见类的 200 张图片训练；开发集为已见 100 张＋另外 4 类的 80 张；最终集为已见 200 张＋另外 6 类的 120 张。共 700 张互不重复图片。

从预声明的 8 个属性组中，仅依据训练集的正负样本支持度选择 24 条属性。保留数据集众包标注中 confidence=3（probably）或 4（definitely）、对应部位可见且位于 CLIP 实际中心裁剪内的条目。看不见／低置信度标为 unknown，不按负例处理。训练全图有 {coverage['train_known_global']}/{coverage['train_possible_global']} 条可用标注；预测区域内另有 {coverage['train_known_regional']} 条可用属性监督。众包标注仍可能有误，gold 是人工标注来源组的简称，不代表无噪声真值。

## 模型与对照

沿用原始 OpenAI CLIP ViT-B/32，两个视觉分支训练末端两层。OWL-ViT 对每张照片使用相同的五条解剖部位／鸟主体提示，最多保留两个预测框；所有区域组使用完全相同的框。可训练属性头预测 24 个属性，预测分数加权固定属性文本向量，再通过 CAF 与全图融合。训练和推理均融合预测属性，真实属性只进入损失或评分。

- Fine-tuning：仅全图分类。
- Regions：区域分支和 CAF，属性文本置零、无属性监督。
- Automatic：冻结原始 CLIP 的正／负属性提示相似度产生软伪标签，不使用逐图人工属性。这不是前轮的 OWL 属性标签，且 CLIP 否定提示不保证校准或语义正确。
- Human attributes：用经过可见性、置信度筛选的逐图全图属性监督。
- Shuffled：同样的已知／未知位置与正负数量，在训练图片间打乱属性值。
- Human + region：在 Human 基础上额外监督预测框内可见部位的属性。部位点只筛选训练标签，不作为模型输入或裁剪依据；点位于框内也不保证完整部位都可见。

6 种变体 × 2 个学习率 × seed 42，共 12 组，每组 160 次更新。微批 2、有效批 16、224 像素、BF16 和两遍梯度缓存。检查点按开发集 GZSL H 选择；独立保留开发属性 mAP 最好的检查点作属性诊断，两类选择均在最终测试前锁定。第 0 步允许被选中，表示未训练初始点，不能算训练收益。

## 按分类指标选择的结果

所有指标为百分比。ZSL 为仅 6 个未见候选类；GZSL 同时提供 10 个已见＋6 个未见类，H 是已见／未见平均每类准确率的调和均值。属性 mAP 仅对可见、已知且有正负样本的属性评分；未知条目不计分。

{table(rows)}

{conclusion}

## 按属性识别选择的结果

下表是另一套预先锁定的选择标准，不能从两张表中事后挑最高分类成绩作为主结果。

{table(attr_rows)}

{learning}

在属性最优检查点的预先独立选择下，人工属性比打乱监督的属性 mAP 高 {contrasts['gold minus shuffled']['attribute_mAP_difference_pp']:.2f} 个百分点，配对图像 bootstrap 95% 区间为 [{contrasts['gold minus shuffled']['attribute_mAP_bootstrap_95pct'][0]:.2f}, {contrasts['gold minus shuffled']['attribute_mAP_bootstrap_95pct'][1]:.2f}]。该结果支持正确配对的人工属性对属性识别有帮助。

纯未见六分类也有一个需谨慎解释的信号：人工属性组相对原始 CLIP 提高 {contrasts['gold minus native']['ZSL_difference_pp']:.2f} 个百分点，未校正 McNemar p={contrasts['gold minus native']['exact_mcnemar_p']:.5f}；但相对区域无文字／打乱监督仅多识别正确 1/120 张，p=1.0。多个探索性比较未经多重校正，不能将对原始模型的变化解释为 AG 特有的显著收益。与此同时，混合分类 H 明显退化，已见类偏置仍然存在。

## 已训练模型是否利用融合文字

以下分析针对确实训练过的“属性最优”检查点，未据此调整模型或重新选择结果。

{diagnostic_text}

文字聚合后的高相似度可能削弱属性差异，是后续排查候选；它不是因果证明。清零／错配只干预融合文字，不能抹去编码器已经学到的属性知识，因此需结合打乱监督、区域无文字对照与分类退化一并解释。

## 验证和边界

官方档案校验、类别与图片隔离、未知标签零梯度、未见类训练拒绝、打乱标签保持边际分布、梯度缓存一致性、两视觉分支／属性头实际更新、无区域回退全部通过。六个分类检查点及两个已训练的人工属性最优检查点，共八个检查点通过真实图片的重载推理一致性检查。实测训练最大分配显存 {max(r['peak_allocated_mib'] for r in configs):.0f} MiB；12 组训练和开发验证累计 {sum(r['seconds'] for r in configs)/60:.1f} 分钟，不包含下载、模型载入、定位及最终评估。

推理只接收图片，固定文本库仍是系统内部组件。没有提供测试图片的正确属性、部位点、真实框或类别标签。推理时将属性文字清零／错配，只改变融合输入，不会消除视觉编码器已学到的信息；完整干预数值见 final_results.json。

这是单种子小规模机制诊断。CUB 官方提示其图像与 ImageNet 存在重叠，因此“未见”只指当前适配训练；也不能排除 CLIP 预训练重叠。配对 bootstrap 仅描述固定类别和种子下的图像抽样不确定性，不替代多种子或独立数据集验证。

运行入口：`cub_models.cmd`。单图：`.\\.venv\\Scripts\\python.exe predict_cub.py "E:\\path\\bird.jpg"`，默认 Human attributes 分类检查点，固定 20 个候选物种。模型及图片保存在本机，GitHub 仅同步代码、配置和统计报告。

来源：[CUB 官方数据及 MD5](https://data.caltech.edu/records/65de6-vp158)、[官方数据说明及重叠提醒](https://www.vision.caltech.edu/datasets/cub_200_2011/)。
'''
    (REPORT/'summary_zh.md').write_text(md,encoding='utf-8')
    en=f'''# CUB image-level attribute supervision pilot

Official archive verified; 20 species, 200 seen training images, 180 development images and 320 final evaluation images. Twenty-four attributes were selected using training-only support. Confidence>=3, visible-part and input-center-crop masks exclude uncertain targets. Crowd annotations are not noise-free ground truth. No annotated parts, boxes or true attributes enter model inference.

Classification-selected results:

{table(rows)}

Attribute-selected results (a separate locked criterion, not alternative classification model selection):

{table(attr_rows)}

Frozen-feature linear attribute probe: original {base_probe:.2f} mAP, supervised probe {best_probe:.2f}. This is an attribute prediction diagnostic, not an AG species classification gain.

Human attributes minus fine-tuning H: {pair['H_difference']:+.2f} percentage points, conditional paired bootstrap 95% interval {pair['H_bootstrap_95pct']}. There is one training seed. CUB/ImageNet overlap and unknown CLIP pretraining overlap limit claims about unseen species. Automatic targets are uncalibrated frozen-CLIP positive/negative-prompt scores; shuffled targets preserve training support and marginals. Human+region adds losses for annotated visible parts inside the same predicted detector crops, without oracle input crops.

The classification-selected human model is update {lookup['gold']['step']}; update zero means it received no attribute training, so its classification difference is not an attribute-training gain. For the trained attribute-selected models, human supervision improves attribute mAP over shuffled targets by {contrasts['gold minus shuffled']['attribute_mAP_difference_pp']:.2f} points, paired image-bootstrap interval {contrasts['gold minus shuffled']['attribute_mAP_bootstrap_95pct']}. Unseen-only accuracy improves over original CLIP by 5 points (uncorrected McNemar p=0.03125), but beats region-only/shuffled controls by only one of 120 images (p=1.0). Mixed seen/unseen performance deteriorates. These exploratory contrasts are not multiplicity-corrected.

Gradient caching, annotation isolation, masked losses, parameter updates and eight checkpoint/image-only deployment checks passed (six classification-selected and two trained human-attribute-selected models). This is a CLIP/CAF adaptation, not an exact AG-CLIP/CoCa reproduction. Interpret attribute learning and additional species classification utility separately.
'''
    (REPORT/'summary_en.md').write_text(en,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(14,5.6));x=np.arange(len(rows));labels=[NAMES[r['variant']] for r in rows]
    for ax,key,title,factor in [(axes[0],'H','Species classification: GZSL H',1),(axes[1],'attribute_map','Image-level attribute recognition: mAP',100)]:
        values=[factor*r['metrics'][key] for r in rows];bars=ax.bar(x,values,color=['#718399','#879eaa','#93bdb8','#c6b68d','#247b6a','#be947a','#395b7b']);ax.set_xticks(x,labels,rotation=30,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Percent');ax.set_title(title);ax.bar_label(bars,fmt='%.1f',padding=3);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('CUB | classification-selected checkpoints | seed 42\nFine-tuning: update 40; all regional variants: update 0 (untrained)',fontsize=12);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(REPORT/'comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));names=['Original readout']+[NAMES[r['variant']] for r in attr_rows if r['variant'] in ['automatic','gold','shuffled','gold_region']]+['Frozen linear probe'];values=[100*rows[0]['metrics']['attribute_map']]+[100*r['metrics']['attribute_map'] for r in attr_rows if r['variant'] in ['automatic','gold','shuffled','gold_region']]+[best_probe]
    bars=ax.bar(np.arange(len(values)),values,color=['#879eaa','#c6b68d','#247b6a','#be947a','#395b7b','#93bdb8']);ax.set_xticks(np.arange(len(values)),names,rotation=25,ha='right');ax.set_ylim(0,75);ax.set_ylabel('Attribute mAP (%)');ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_title('Attribute learning | checkpoints selected by development attribute mAP');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(REPORT/'attribute_learning.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6));palette=plt.get_cmap('tab10')
    for vi,variant in enumerate(p['variants']):
        for li,lr in enumerate(p['learning_rates']):
            history=json.loads((OUT/f'{variant}_seed42_lr{lr:g}'/'history.json').read_text(encoding='utf-8'));points=[h for h in history if 'development' in h]
            ax.plot([100*h['development']['attribute_map'] for h in points],[h['development']['H'] for h in points],color=palette(vi),marker='o' if li==0 else 's',linestyle='-' if li==0 else '--',label=f'{NAMES[variant]} / {lr:g}',alpha=.85)
    ax.set_xlabel('Development attribute mAP (%)');ax.set_ylabel('Development GZSL H (%)');ax.set_title('Training trajectories: attribute learning vs. classification');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(REPORT/'development_tradeoff.png',dpi=160);plt.close(fig)
    export=Path('E:/Codex/2026-09-27/yo/outputs');export.mkdir(parents=True,exist_ok=True)
    for name in ['summary_zh.md','summary_en.md','comparison.csv','comparison.png','attribute_learning.png','development_tradeoff.png']:shutil.copy2(REPORT/name,export/f'CUB_Attribute_Diagnostic_{name}')
    print(conclusion);print(learning)

if __name__=='__main__':main()
