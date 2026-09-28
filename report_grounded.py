"""Export the completed starter experiment without images or model weights."""
import csv,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json
from multimodal.grounded_experiment import OUT,REPORT,provenance,measures

LABELS={'native':'Original CLIP','finetune':'CLIP fine-tuning','region_only':'Regions, zero text','ag':'Attribute-guided','shuffled':'Shuffled attributes'}

def bootstrap_h_difference(a,b,p,replicates=2000):
    # Paired, within-class bootstrap. This estimates image-sampling uncertainty
    # conditional on these classes and this seed, not training-seed variability.
    assert a['paths']==b['paths'] and torch.equal(a['labels'],b['labels'])
    y=a['labels'].numpy();ca=(a['predictions']==a['labels']).numpy();cb=(b['predictions']==b['labels']).numpy();rng=np.random.default_rng(20260928)
    acc_a=[];acc_b=[]
    for c in p['seen_classes']+p['eval_unseen_classes']:
        ids=np.where(y==c)[0];resampled=rng.choice(ids,size=(replicates,len(ids)),replace=True)
        acc_a.append(ca[resampled].mean(1));acc_b.append(cb[resampled].mean(1))
    def h(values):
        x=np.stack(values);s=x[:10].mean(0);u=x[10:].mean(0)
        return 200*s*u/np.maximum(s+u,1e-12)
    low,high=np.quantile(h(acc_a)-h(acc_b),[.025,.975]);return [float(low),float(high)]

def main():
    payload=json.loads((REPORT/'final_results.json').read_text(encoding='utf-8'));assert payload['provenance']==provenance()
    p=json.loads((ROOT/'configs/grounded_v1.json').read_text(encoding='utf-8'));smoke=json.loads((REPORT/'smoke.json').read_text(encoding='utf-8'))
    deployment=json.loads((REPORT/'deployment_verification.json').read_text(encoding='utf-8'))
    assert smoke['status']=='passed' and deployment['status']=='passed' and len(deployment['checks'])==4
    assert smoke['provenance']==deployment['provenance']==provenance()
    regions=json.loads((ROOT/'cache/grounded_v1/regions_development.json').read_text(encoding='utf-8'))['images']
    prompts=json.loads((ROOT/'configs/expanded_attributes.json').read_text(encoding='utf-8'))['prompts']
    manifest=json.loads((ROOT/'data/grounded_v1/manifest.json').read_text(encoding='utf-8'))
    count=Counter(i for r in regions.values() for i in r['attribute_ids']);coverage=[]
    for c in manifest['classes']:
        subset=[r for r in manifest['rows'] if r['label']==c['label'] and r['path'] in regions]
        if subset:coverage.append(dict(name=c['name'],images=len(subset),with_regions=sum(bool(regions[r['path']]['boxes']) for r in subset)))
    write_json(REPORT/'region_coverage.json',dict(scope='Development-only descriptive audit; not used for tuning. Proposals are not verified attribute labels.',vocabulary_size=len(prompts),used_attributes=len(count),total_regions=sum(count.values()),top_attributes=[dict(attribute_id=i,prompt=prompts[i],count=n) for i,n in count.most_common(10)],per_class=coverage))
    results=payload['results'];rows=[]
    for r in results:
        rows.append(dict(variant=r['variant'],selected_step=r['best_step'],lr=r.get('lr',''),**{k:r['metrics'][k] for k in ['ZSL','U','S','H']}))
    with (REPORT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    ag=next(r for r in results if r['variant']=='ag');native=results[0];fine=next(r for r in results if r['variant']=='finetune')
    ag_pred=torch.load((ROOT/ag['checkpoint']).parent/'final_predictions.pt',weights_only=True)
    native_pred=torch.load(OUT/'native_predictions.pt',weights_only=True)
    fine_pred=torch.load((ROOT/fine['checkpoint']).parent/'final_predictions.pt',weights_only=True)
    ci_native=bootstrap_h_difference(ag_pred,native_pred,p);ci_fine=bootstrap_h_difference(ag_pred,fine_pred,p)
    statistics=dict(ag_minus_native_H=ag['metrics']['H']-native['metrics']['H'],ag_minus_finetune_H=ag['metrics']['H']-fine['metrics']['H'],paired_bootstrap_95pct_H_difference_vs_native=ci_native,paired_bootstrap_95pct_H_difference_vs_finetune=ci_fine,replicates=2000,scope='Within-class paired image bootstrap, conditional on selected 16 classes and seed 42; not seed variability or independent held-out-dataset evidence.')
    write_json(REPORT/'uncertainty.json',statistics)
    configs=[json.loads(f.read_text(encoding='utf-8')) for f in OUT.glob('*/result.json')];assert len(configs)==8
    max_memory=max(r['peak_allocated_mib'] for r in configs);total_seconds=sum(r['seconds'] for r in configs)
    def table():
        lines=['| Model | Selected update | ZSL (6 classes) | GZSL U | GZSL S | GZSL H |','|---|---:|---:|---:|---:|---:|']
        for r in results:lines.append(f'| {LABELS[r["variant"]]} | {r["best_step"]} | '+ ' | '.join(f'{r["metrics"][k]:.2f}' for k in ['ZSL','U','S','H'])+' |')
        return '\n'.join(lines)
    conclusion=f'AG 相对原始 CLIP 的 H 差值为 {statistics["ag_minus_native_H"]:+.2f} 个百分点，图像重采样 95% 区间为 [{ci_native[0]:+.2f}, {ci_native[1]:+.2f}]；相对普通微调为 {statistics["ag_minus_finetune_H"]:+.2f} 个百分点。'
    if ci_native[0]<=0:conclusion+=' 当前结果不足以证明 AG 稳定优于原始 CLIP。'
    region=next(r for r in results if r['variant']=='region_only');shuffled=next(r for r in results if r['variant']=='shuffled')
    conclusion+=f' AG 与仅区域对照的 H 差值为 {ag["metrics"]["H"]-region["metrics"]["H"]:+.2f}，与打乱属性对照为 {ag["metrics"]["H"]-shuffled["metrics"]["H"]:+.2f}；因此不能把相对原始 CLIP 的改善全部归因于属性语义。本轮未建立属性带来独立分类增益的证据。'
    md=f'''# 本机 AG 起步配置与实测结果

部署目录：`E:\\ELEC4240\\SpeciesRecognition`。本轮是结构更完整的 CLIP 属性引导改编，使用 OWL-ViT 实际区域、独立属性视觉编码器和 CAF；不是原论文 CoCa 的严格复现。

## 已完成配置

- OpenAI CLIP ViT-B/32，全图和属性分支分别训练最后两个 Transformer block、最终归一化与投影；文字编码固定。
- 每图最多 2 个区域，分辨率 224，microbatch 2，通过两遍梯度缓存组成有效 batch 16。梯度缓存与完整计算图的梯度最大误差为 {smoke['gradient_cache_max_error']:.1g}。
- BF16 混合精度、激活检查点；检测与训练顺序运行。训练矩阵实际峰值显存分配 {max_memory:.0f} MiB，8 组训练/开发验证总用时 {total_seconds/60:.1f} 分钟（不含加载模型、定位与最终推理）。显卡是 RTX 5060 Laptop 8 GB；桌面/其他程序占用另计。
- 4 种变体 × 2 个视觉学习率（1e-5、3e-6）× 1 个种子，每组 120 次更新。开发集在 0/40/80/120 次更新选择检查点；**0 表示未训练初始点被选中，不能视为训练带来提升**。
- CE、有效批次图文对比、原始特征保持；AG 额外加入置信度加权区域属性对齐。重复类别的图片在对比损失中作为多正样本。
- 四个选中检查点均已重载并完成实际单图推理验证，预测类别与批量评估全部一致，最大 logits 误差 {max(r['max_logit_error'] for r in deployment['checks']):.8f}。单图复制到固定微批大小后丢弃重复输出，以保持 BF16 计算形状一致，不是多图投票。

## 数据与评估

已有 iNaturalist 实际图片，20 个物种（昆虫与植物）。本轮 10 个已见训练类、4 个开发未见类、6 个最终未见类；训练图片从 200 增为 **300**，开发验证 180，最终评估 220，共 700 张唯一图片。物种数量未增加。

开发集使用 10 个已见类＋4 个开发未见类的混合候选分类，按 H 选择学习率/检查点。全部选择锁定后才对最终 220 张图定位和评估。ZSL 只在 6 个未见候选类上测试；GZSL 同时提供 10 个已见＋6 个未见候选类，U/S 分别为未见/已见平均每类准确率，H 为两者调和均值。所有数值为百分比。

{table()}

{conclusion}

原图/类别此前用于过项目探索，当前属于复用数据的探索实验；未见仅指本轮适配训练未见，不能排除 CLIP 预训练重叠。与前轮的已见验证划分不同，不能把跨轮差值全部归因于算法改动。只有一个训练种子，上述 bootstrap 仅估计固定类别内图片采样不确定性。

## 属性对照与限制

区域对照使用同样的检测框、独立视觉分支和 CAF，但将属性文字置零；打乱对照使用相同框和错配的文字/辅助目标。两者均保留了属性驱动的检测过程，不能称为完全无语义信息的区域选择。

本轮固定 58 条已有来源属性，OWL-ViT 根据统一词表定位，不接收真实类别。开发图片有 105/480（21.9%）没有有效区域；这些图片精确退回全图分支。固定抽查的 20 张训练图片可见正确主体、粗略整虫框、背景花朵和错误描述。因此区域及其文字只是弱标签，不能声称是专家确认的图片级可见属性真值。

开发集的 607 个检测区域仅使用了 {len(count)}/58 条属性，最频繁的 6 条占 {sum(n for _,n in count.most_common(6))/sum(count.values())*100:.1f}%。例如 Briza minor 仅有 11/40 张图片检出区域；属性覆盖率和准确性仍是当前限制，梯度通过不等于语义监督有效。

AG 测试时置零文字：H={ag['interventions']['zero_text']['H']:.2f}；错配文字：H={ag['interventions']['permuted']['H']:.2f}。这些是同一检查点的事后敏感性分析，不用于重新挑选参数。它们只改变融合输入，不会消除视觉编码器在训练时已经学到的属性信息，需结合独立训练的区域／打乱对照解释。

用户预测只输入图片，但部署内部仍需要固定的类别/属性文字库和 OWL-ViT，区别于早期“文字只在训练期使用”的方案。

## 操作

双击根目录 `grounded_models.cmd` 检查并运行完整起步流程；已完成单元会保留，未完成的训练单元按相同种子从头重跑。不要同时启动两个相同任务。单图预测：

```powershell
cd E:\\ELEC4240\\SpeciesRecognition
.\\.venv\\Scripts\\python.exe predict_grounded.py "E:\\path\\photo.jpg"
```

默认使用开发集选出的 AG 检查点，并从固定 20 类中给出前五个匹配分数，不是任意物种识别器。`--candidates gzsl` 复现 16 类混合候选；`--candidates unseen` 复现 6 类 ZSL。不能按图片真实类别临时选择候选集合。分数未做概率校准。

代码、配置和报告可同步私有 GitHub；图片、模型权重和像素缓存只保留本地。详细命令、损失与复现边界见 `docs/grounded_starter.md`。
'''
    (REPORT/'summary_zh.md').write_text(md,encoding='utf-8')
    en=f'''# Grounded CLIP starter results

This local CLIP ViT-B/32 implementation trains the last two blocks of independent whole-image and attribute-region visual towers, using OWL-ViT proposals and CAF self-attention. It is an AG-inspired structural reconstruction, not an exact CoCa-paper reproduction.

The reused 20-species iNaturalist subset contains 300 training images, 180 development images and 220 final evaluation images. Four variants, two learning rates and seed 42 yield eight 120-update runs. Development GZSL H selects checkpoints before final evaluation. Update zero is eligible and indicates an untrained selected checkpoint. ZSL uses six unseen candidates; GZSL uses ten seen plus six unseen candidates.

{table()}

AG minus original CLIP H: {statistics['ag_minus_native_H']:+.2f} percentage points; paired within-class bootstrap 95% interval [{ci_native[0]:+.2f}, {ci_native[1]:+.2f}]. AG minus fine-tuning H: {statistics['ag_minus_finetune_H']:+.2f}. This interval does not estimate training-seed variability; these are reused exploratory data and pretraining overlap is unknown.

AG minus region-only H: {ag['metrics']['H']-region['metrics']['H']:+.2f}; AG minus shuffled H: {ag['metrics']['H']-shuffled['metrics']['H']:+.2f}. The improvement over original CLIP must not be attributed solely to semantic attributes. These controls do not establish an independent classification benefit from the attributes.

Microbatch 2 and exact two-pass gradient caching produce an effective contrastive batch of 16. BF16 and activation checkpointing use a measured maximum allocation of {max_memory:.0f} MiB. Eight runs including development validation took {total_seconds/60:.1f} minutes, excluding model loading and region detection. Smoke tests prove gradient equivalence, updates to both visual towers and CAF, unchanged frozen early layers, unseen-training rejection and exact missing-region fallback.

The 58 fixed attribute prompts are weak semantic supervision. OWL-ViT found no valid region for 105/480 development images, and visual inspection found background crops and incorrect attributes. Region-only and shuffled controls share the attribute-driven detector; neither removes all semantic guidance. At inference users supply only an image, while the deployed system still uses fixed text banks and the detector. The resulting predictor covers the configured candidate species, not arbitrary wildlife.
'''
    (REPORT/'summary_en.md').write_text(en,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5));x=np.arange(len(results));names=[LABELS[r['variant']] for r in results]
    for ax,key,title in zip(axes,['ZSL','H'],['Unseen-only classification (6 classes)','Mixed seen/unseen classification (GZSL H)']):
        values=[r['metrics'][key] for r in results];bars=ax.bar(x,values,color=['#687c91','#a4b1bf','#81b9b4','#217b6c','#d3a25f'])
        ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Percent');ax.set_title(title);ax.bar_label(bars,fmt='%.2f',padding=3);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Grounded CLIP starter | seed 42 | reused exploratory dataset');fig.tight_layout();fig.savefig(REPORT/'comparison.png',dpi=170);plt.close(fig)
    target=Path('E:/Codex/2026-09-27/yo/outputs');target.mkdir(parents=True,exist_ok=True)
    for file in ['summary_zh.md','summary_en.md','comparison.csv','comparison.png']:shutil.copy2(REPORT/file,target/f'Grounded_AGCLIP_{file}')
    print(md)

if __name__=='__main__':main()
