"""Report the locked class-disjoint pilot, including negative results and controls."""
import csv,json,shutil,statistics
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json,digest
from report_expanded import paired_statistics,holm

def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    root=ROOT/'runs/zsl_v1';out=ROOT/'reports/zsl_v1';out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'configs/zsl_v1.json').read_text(encoding='utf-8'));lock=json.loads((root/'selection_locked.json').read_text(encoding='utf-8'))
    rows=json.loads((root/'final_results.json').read_text(encoding='utf-8'));dev=json.loads((root/'development_results.json').read_text(encoding='utf-8'))
    native=json.loads((out/'native_clip_diagnostic.json').read_text(encoding='utf-8'))
    assert len(rows)==60 and len(dev)==120
    flat=[]
    for r in rows:
        item={k:r[k] for k in ['backbone','variant','lr','seed','gzsl_unseen','gzsl_seen','gzsl_h']}
        item.update(zsl_accuracy=r['zsl']['per_class_accuracy'])
        for mode,values in r['interventions'].items():item.update({f'{mode}_{k}':values[k] for k in ['accuracy','flips']})
        flat.append(item)
    csv_write(out/'final_results.csv',flat)
    csv_write(out/'development_results.csv',[dict(backbone=r['backbone'],variant=r['variant'],lr=r['lr'],seed=r['seed'],best_step=r['best_step'],dev_accuracy=r['validation']['per_class_accuracy'],dev_loss=r['validation']['loss']) for r in dev])
    summary=[];comparisons=[]
    for name in protocol['backbones']:
        ensembles={};identity=None
        for variant in protocol['variants']:
            group=[r for r in flat if r['backbone']==name and r['variant']==variant]
            item=dict(backbone=name,variant=variant,lr=group[0]['lr'])
            for key in ['zsl_accuracy','gzsl_unseen','gzsl_seen','gzsl_h']:
                item[key]=statistics.mean(r[key] for r in group);item[key+'_sd']=statistics.stdev(r[key] for r in group)
            summary.append(item);predictions=[]
            for r in group:
                p=torch.load(root/f'{name}_seed{r["seed"]}_lr{r["lr"]:g}'/variant/'final_predictions.pt',weights_only=True)['zsl']
                current=(p['paths'],p['labels'].tolist(),p['candidates'])
                if identity is not None:assert identity==current
                identity=current;predictions.append(p['logits'].softmax(-1))
            ensembles[variant]=torch.tensor(p['candidates'])[torch.stack(predictions).mean(0).argmax(-1)].tolist()
        for reference in ['baseline','region_only']:
            comparisons.append(dict(backbone=name,reference=reference,**paired_statistics(ensembles[reference],ensembles['ag_aux'],identity[1])))
    holm(comparisons);write_json(out/'summary.json',summary);write_json(out/'paired_comparisons.json',comparisons);write_json(out/'selection_locked.json',lock)
    get=lambda n,v:next(r for r in summary if r['backbone']==n and r['variant']==v)
    lines=['# Class-disjoint zero-shot pilot: five backbones','',
        '120 development cells and 60 locked evaluations completed. This is a reused-data exploratory pilot, not a strict AG-CLIP reproduction or an independent benchmark. Unseen classes are absent from this round of adaptation training; pretraining exposure is not ruled out.',
        '', '## Protocol','',
        '- Original five backbones: EfficientNet-B0, ResNet-18, ConvNeXt-Tiny, ViT-B/16, OpenAI CLIP ViT-B/32. Original frozen pretrained features are reused; all task projections/fusion modules are reinitialized. No earlier all-20-class task checkpoint is loaded.',
        '- Ten seen species (five insects, five plants): 20 training images each = 200. Four other species: 20 images each = 80 development images for checkpoint/LR selection. Six final unseen species: 20 images each = 120. An additional 200 seen-class images are used only for final GZSL evaluation. Class sets are disjoint; image hashes across roles are unique.',
        '- Species names and class attributes for unseen classes are permitted semantic side information. Unseen images and image labels never participate in gradient updates. Training CE uses only seen candidate classes.',
        '- Two learning rates, three optimization seeds, 200 alignment updates plus 200 comparison updates. The initial alignment weights are shared among variants for each model/LR/seed. All variants have class CE plus 0.25 cosine alignment to the seen class name. The AG variant adds 0.1 cosine alignment to the seen class attribute prototype.',
        '- Compare global baseline, region-only capacity control, attribute-guided fusion/auxiliary alignment, and wrongly paired attributes. Every image receives the same text vocabulary; true labels cannot choose inference crops or text. Fixed five overlapping crops are not anatomical detections.',
        '- Development-unseen macro class accuracy selects checkpoint and LR; CE breaks ties. All 20 model/variant LR choices are written to a fingerprinted selection lock before final evaluation. There is no final refit or test-calibrated seen-class bias correction.',
        '', '## Results','',
        f'**Critical additional reference:** original CLIP without task adaptation obtains {100*native["zsl"]["per_class_accuracy"]:.2f}% ZSL; GZSL U/S/H = {100*native["gzsl_unseen"]:.2f}/{100*native["gzsl_seen"]:.2f}/{100*native["gzsl_h"]:.2f}%. This reference was checked post hoc with identical prompts/global-image preprocessing and no tuning. In the table, baseline means a seen-class-adapted baseline, not original CLIP. A gain against that baseline can recover adaptation damage without outperforming native CLIP.',
        '', '| Backbone | Adapted baseline ZSL mean ± SD (%) | Region-only (%) | AG mean ± SD (%) | Shuffled (%) | AG − adapted baseline (pp) | AG GZSL unseen / seen / H (%) |','|---|---:|---:|---:|---:|---:|---:|']
    for n in protocol['backbones']:
        b=get(n,'baseline');a=get(n,'ag_aux');r=get(n,'region_only');s=get(n,'shuffled_attributes')
        lines.append(f'| {n} | {100*b["zsl_accuracy"]:.2f} ± {100*b["zsl_accuracy_sd"]:.2f} | {100*r["zsl_accuracy"]:.2f} | {100*a["zsl_accuracy"]:.2f} ± {100*a["zsl_accuracy_sd"]:.2f} | {100*s["zsl_accuracy"]:.2f} | {100*(a["zsl_accuracy"]-b["zsl_accuracy"]):+.2f} | {100*a["gzsl_unseen"]:.2f} / {100*a["gzsl_seen"]:.2f} / {100*a["gzsl_h"]:.2f} |')
    lines+=['', 'ZSL uses six unseen candidates (uniform chance 16.67%). GZSL uses sixteen candidates: ten seen plus six unseen. U and S are mean per-class accuracies; H is their harmonic mean, computed per run then averaged. The four development classes are excluded from final candidate sets. Seed SD describes optimization variation on one fixed class/image split, not variability over different unseen species.',
        '', '## Paired exploratory comparisons','', '| Backbone | Reference | Ensemble gain (pp) | Paired image bootstrap 95% CI (pp) | Holm-adjusted McNemar p |','|---|---|---:|---:|---:|']
    for c in comparisons:
        lo,hi=c['paired_stratified_bootstrap_95ci_pp'];lines.append(f'| {c["backbone"]} | {c["reference"]} | {c["improvement_percentage_points"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {c["holm_adjusted_p"]:.4f} |')
    lines+=['', 'These compare one three-seed ensemble prediction per image. Holm correction covers ten planned comparisons. Confidence intervals are unadjusted, image-level and conditional on these six classes; they do not measure transfer to a wider population of species. Prior reuse of these images and possible pretraining overlap further limit interpretation.',
        '', '## Limitations and next dataset','',
        'This experiment fixes the seen/unseen class protocol while retaining a small frozen-feature AG-inspired method. It does not implement CoCa-L, a separate trainable attribute visual encoder, OWL-ViT grounding, or the exact CAF/training of the AG-CLIP paper. Improved performance would support this adaptation only; no improvement would not refute the original method. Earlier supervised accuracies with twenty candidate classes are not directly comparable to six-way ZSL.',
        '', 'AwA2 offers 50 animal categories, 37,322 images and 85 class-level attributes; use the proposed split designed to avoid ImageNet category overlap. It is broader than birds but not all labels are strict biological species, and not all attributes are visually observable. CUB offers 200 bird species, 11,788 images, 312 image attributes and 15 part locations; its official page warns of possible ImageNet image overlap. Neither automatically guarantees that CLIP pretraining never encountered a class or image.',
        '', '- [AwA2 official data and proposed splits](https://cvml.ista.ac.at/AwA2/)', '- [CUB official data](https://www.vision.caltech.edu/datasets/cub_200_2011/)',
        '', 'All development and selected final cells, class identities, controls and declines are retained. Photos and model weights remain local.','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    cn=['# 五模型类别隔离零样本小规模测试','',
        '“未见”指类别没有参与本轮任务训练。正确流程是用已见类别图片训练，用未见类别图片评估；若将未见类别的有标签图片拿来训练，就变成少样本学习。未见类别名称和文字属性可以作为语义信息提前提供。',
        '', '这轮使用已有 iNaturalist 图片重新划分：10 个物种用于训练（200 张）、4 个不同物种用于开发验证（80 张）、6 个不同物种用于最终未见类别评估（120 张）。额外 200 张已见类别图片用于混合分类评估。所有任务适配模块重新初始化，未复用之前用全部 20 类训练的检查点。图片此前已参与研究，所以结果属于探索，不能称为全新独立测试。',
        '', '完成 120 组开发训练与 60 组参数锁定后的评估。下面是六个未见物种之间分类的平均准确率；三个随机种子只改变优化，不改变图片或类别划分。',
        '', f'**重要对照：原始 CLIP 未经本轮任务适配，已经达到 {100*native["zsl"]["per_class_accuracy"]:.2f}% ZSL。** 表中的“适配基线”经过已见类别训练，性能可能低于原始模型。因此不能把 AG 相对适配基线的提升，写成超过原始 CLIP 的提升。原始 CLIP 的混合分类 GZSL U/S/H 为 {100*native["gzsl_unseen"]:.2f}% / {100*native["gzsl_seen"]:.2f}% / {100*native["gzsl_h"]:.2f}%。该对照是在最终成绩出来后补查的，没有改变模型或调参。',
        '', '| 模型 | 全图适配基线 | 仅区域对照 | 属性引导 AG | 打乱属性 | AG 相对适配基线 |','|---|---:|---:|---:|---:|---:|']
    for n in protocol['backbones']:
        b=get(n,'baseline');a=get(n,'ag_aux');r=get(n,'region_only');s=get(n,'shuffled_attributes')
        cn.append(f'| {n} | {100*b["zsl_accuracy"]:.2f}% | {100*r["zsl_accuracy"]:.2f}% | {100*a["zsl_accuracy"]:.2f}% | {100*s["zsl_accuracy"]:.2f}% | {100*(a["zsl_accuracy"]-b["zsl_accuracy"]):+.2f} 个百分点 |')
    cn+=['', '六分类随机猜测的期望准确率为 16.67%。这些数值不能与此前二十分类监督实验直接比较。完整英文报告还给出已见／未见混合分类（GZSL）的 U、S 和调和平均 H，以及区域对照和配对统计。',
        '', f'CLIP＋AG 的六分类 ZSL 为 {100*get("clip_vit_b32","ag_aux")["zsl_accuracy"]:.2f}%，但混合分类中未见类别准确率仅为 {100*get("clip_vit_b32","ag_aux")["gzsl_unseen"]:.2f}%，说明模型仍强烈偏向已见类别。当前结果支持“属性引导可以缓解某些适配损失”，不足以声称获得优于原始 CLIP 的通用物种识别器。',
        '', '当前实现仍是冻结主干、固定区域裁剪与属性融合／辅助对齐的轻量 AG 改编，尚非原论文的 CoCa＋区域定位＋属性视觉编码器完整复现。四个 ImageNet 模型只在少量已见类别上学习到文字空间的映射，不保证具备通用 CLIP 的迁移能力。',
        '', '适合后续正式实验的数据集：AwA2（50 类动物、85 个类别属性，范围不限于鸟，但标签不全是严格的物种）；CUB（200 种鸟、312 个图片属性及部位标注，更接近细粒度属性研究）。它们都有预训练重叠方面需要核查的限制。此次未另行下载这两个数据集。',
        '', '所有学习率、负结果和打乱属性对照均保留。参数先在开发类别上选定并锁定，再评估最终类别；不根据最终成绩重新训练。','']
    (out/'Summary_CN.md').write_text('\n'.join(cn),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(5);width=.19
    for j,v in enumerate(protocol['variants']):
        values=[get(n,v) for n in protocol['backbones']]
        ax.bar(x+(j-1.5)*width,[100*r['zsl_accuracy'] for r in values],width,label='adapted baseline' if v=='baseline' else v,yerr=[100*r['zsl_accuracy_sd'] for r in values],capsize=2)
    ax.scatter([4],[100*native['zsl']['per_class_accuracy']],marker='D',color='black',s=45,label='Native CLIP (no adaptation)',zorder=5)
    ax.axhline(100/6,color='gray',linestyle='--',label='Six-way chance');ax.set_ylim(0,100);ax.set_xticks(x,protocol['backbones']);ax.set_ylabel('Unseen per-class accuracy (%)');ax.set_title('Class-disjoint pilot: fixed 10 seen / 4 dev / 6 final unseen species')
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13));ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    export=Path('E:/Codex/2026-09-27/yo/outputs');export.mkdir(parents=True,exist_ok=True)
    for source,target in [('README.md','ZeroShot_Results.md'),('Summary_CN.md','ZeroShot_Summary_CN.md'),('comparison.png','ZeroShot_Comparison.png'),('final_results.csv','ZeroShot_Final_Results.csv'),('development_results.csv','ZeroShot_Development_Results.csv')]:shutil.copy2(out/source,export/target)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
