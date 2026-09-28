"""Reproducible descriptive reporting; validation only, no significance claims."""
import csv,json,statistics,shutil
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from data_tools import ROOT,write_json

VARIANTS=['baseline','capacity_control','attribute_supervision','attribute_region','shuffled_supervision']
METRICS=['top1_accuracy','macro_f1','loss','attribute_map','attribute_macro_f1','grounding_peak_in_box','grounding_attention_mass','grounding_uniform_area_reference']
LABELS=['Global baseline','Capacity control','Attributes','Attributes + regions','Shuffled supervision']

def main():
    out=ROOT/'reports/visible_v1';out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'runs/visible_v1').glob('*/result.json'))]
    assert sum(r['backbone']!='fgclip_b16' for r in rows)==60
    flat=[]
    for r in rows:
        item={k:r[k] for k in ['backbone','variant','seed','lr','best_step','first_patch_adapter_gradient_l1']}
        item.update({f'val_{k}':r['validation'][k] for k in METRICS})
        item['train_accuracy']=r['training']['top1_accuracy']
        for v in r['interventions']:
            for k in ['top1_accuracy','flipped_predictions','mean_probability_l1']:item[f'{v["mode"]}_{k}']=v[k]
        flat.append(item)
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (out/'all_runs.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    groups=defaultdict(list)
    for r in rows:groups[(r['backbone'],r['variant'],r['lr'])].append(r)
    models=[m for m in ['clip_b16','siglip2_b16','fgclip_b16'] if any(r['backbone']==m for r in rows)]
    selected=[]
    for model in models:
        for variant in VARIANTS:
            candidates=[g for (m,v,lr),g in groups.items() if m==model and v==variant]
            if not candidates:continue
            g=max(candidates,key=lambda g:(statistics.mean(r['validation']['macro_f1'] for r in g),-statistics.mean(r['validation']['loss'] for r in g)))
            summary=dict(backbone=model,variant=variant,lr=g[0]['lr'],seeds=[r['seed'] for r in g],n=len(g))
            for k in METRICS:
                values=[r['validation'][k] for r in g];summary[k]=statistics.mean(values);summary[k+'_sd']=statistics.stdev(values) if len(values)>1 else None
            summary['attribute_evaluable_count']=g[0]['validation']['attribute_evaluable_count']
            summary['positive_grounding_pairs']=g[0]['validation']['positive_grounding_pairs']
            summary['interventions']={mode:{k:statistics.mean(next(i for i in r['interventions'] if i['mode']==mode)[k] for r in g) for k in ['top1_accuracy','flipped_predictions','mean_probability_l1']} for mode in ['zero','permuted']} if variant!='baseline' else {}
            selected.append(summary)
    write_json(out/'selected_summary.json',selected)
    get=lambda m,v:next(r for r in selected if r['backbone']==m and r['variant']==v)
    lines=['# Visible-attribute and region-alignment follow-up','',
        f'{len(rows)} completed training cells. Results below are exploratory **validation**, not held-out test performance.',
        '','Fixed 20 species; 200 training images (10/class) and 200 validation images. Sixty images received assistant-reviewed provisional attribute annotations: 40 train and 20 validation. One image in each split is entirely unknown, leaving 39/19 with known labels. Sixteen visual traits; unobserved and non-applicable entries are masked. Some attributes have only one positive training example.',
        '', 'CLIP and SigLIP 2 use three optimization seeds on the same images. FG-CLIP, when present, is a one-seed follow-up. Every variant has two learning rates and four checkpoint evaluations; select checkpoint by validation macro-F1, then CE, and select LR by the same metrics averaged over seeds. This development set has been reused; no significance or generalization claim is made.',
        '', '| Backbone | Condition | LR | Accuracy mean ± SD (%) | Attribute mAP (%) | Peak inside box (%) |',
        '|---|---|---:|---:|---:|---:|']
    for r in selected:
        sd=f' ± {100*r["top1_accuracy_sd"]:.2f}' if r['n']>1 else ' (one seed)'
        lines.append(f'| {r["backbone"]} | {r["variant"]} | {r["lr"]:g} | {100*r["top1_accuracy"]:.2f}{sd} | {100*r["attribute_map"]:.2f} | {100*r["grounding_peak_in_box"]:.2f} |')
    lines+=['','## Classification changes and branch interventions','', '| Backbone | Full AG minus global baseline (pp) | Full AG minus capacity control (pp) | Full AG with attribute branch zeroed (%) | Labels changed by zeroing / 200 | Labels changed by channel permutation / 200 |','|---|---:|---:|---:|---:|---:|']
    for m in models:
        b=get(m,'baseline');a=get(m,'attribute_region');c=get(m,'capacity_control');i=a['interventions']
        lines.append(f'| {m} | {100*(a["top1_accuracy"]-b["top1_accuracy"]):+.2f} | {100*(a["top1_accuracy"]-c["top1_accuracy"]):+.2f} | {100*i["zero"]["top1_accuracy"]:.2f} | {i["zero"]["flipped_predictions"]:.2f} | {i["permuted"]["flipped_predictions"]:.2f} |')
    lines+=['','These post-training interventions measure reliance of a trained model on its branch, not the causal benefit of retraining with attribute supervision. Permuting channels also disrupts the learned linear classifier, so changes alone do not prove semantic use.',
        '', '## Interpretation and limits','',
        '- Attribute prediction and species accuracy are different outcomes. Higher attribute mAP does not establish a species-classification improvement. Compare against the strong global baseline as well as the capacity control; a weak capacity control can exaggerate apparent gains.',
        '- Box localization is coarse. Many boxes enclose an organism or flower cluster, and shuffled annotations can still encourage generic foreground attention. High peak-in-box scores alone do not prove attribute-specific grounding. The CSV includes attention mass and the uniform-area reference.',
        '- Auxiliary labels are assistant-reviewed, not independent expert ground truth. Attribute metrics use only the small reviewed validation subset; only attributes with both positive and negative validation labels enter mAP. Rare-label AP is unstable.',
        '- Frozen encoders with trainable global/patch adapters and a 16-concept additive classification branch. This is an AG-inspired custom experiment, not a strict reproduction of the AG-CLIP paper. No validation labels or boxes are model inputs; at prediction time only an image is supplied, while fixed text embeddings remain inside the checkpoint.',
        '- The new image subset and preprocessing differ from earlier 225-run/96-run experiments. Cross-round scores are not controlled before/after comparisons.',
        '- FG-CLIP already received fine-grained region-text pretraining. It is a stronger alternative backbone, not a model devoid of attribute-related training. Web-pretraining overlap with iNaturalist has not been ruled out, so this is not a contamination-free benchmark.',
        '- All negative results and all learning rates are retained in all_runs.csv. No test images were encoded or evaluated in this follow-up.',
        '', '## Sources','',
        '- [CLIP](https://proceedings.mlr.press/v139/radford21a.html)',
        '- [SigLIP 2 official model](https://huggingface.co/google/siglip2-base-patch16-224)',
        '- [FG-CLIP official model and dense-feature API](https://huggingface.co/qihoo360/fg-clip-base)',
        '- [FG-CLIP paper](https://arxiv.org/abs/2505.05071)', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,4.7));x=np.arange(len(models));width=.15
    for j,v in enumerate(VARIANTS):
        for ax,key in zip(axes,['top1_accuracy','attribute_map']):
            means=[100*get(m,v)[key] for m in models];sd=[100*(get(m,v)[key+'_sd'] or 0) for m in models]
            ax.bar(x+(j-2)*width,means,width,label=LABELS[j],yerr=sd,capsize=2)
    for ax,title in zip(axes,['Species accuracy','Visible-attribute mAP (small reviewed subset)']):
        ax.set_xticks(x,models);ax.set_ylabel('Validation (%)');ax.set_ylim(0,100);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Fixed 20-species, 10-shot pilot; error bars: optimization-seed SD; FG-CLIP: one seed')
    fig.tight_layout(rect=(0,.12,1,.93));fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    export=Path('E:/Codex/2026-09-27/yo/outputs');export.mkdir(parents=True,exist_ok=True)
    for src,dest in [('README.md','Visible_Attribute_Results.md'),('all_runs.csv','Visible_Attribute_All_Runs.csv'),('comparison.png','Visible_Attribute_Comparison.png')]:shutil.copy2(out/src,export/dest)
    cn=['# 图片级可见属性与区域对齐：实验结果','',f'本轮已完成 {len(rows)} 组训练。20 个物种，每类 10 张训练图片；固定 200 张训练图、200 张验证图。CLIP 与 SigLIP 2 各三个优化随机种子；FG-CLIP 为追加的单种子小规模试验。结果均来自反复使用的开发验证集，不能当作独立测试结果，也不能据此声称统计显著。',
        '', '| 模型 | 全图基线准确率 | 属性监督 | 属性＋区域监督 | 完整方案相对基线 |', '|---|---:|---:|---:|---:|']
    for m in models:
        b=get(m,'baseline');a=get(m,'attribute_region');s=get(m,'attribute_supervision')
        cn.append(f'| {m} | {b["top1_accuracy"]*100:.2f}% | {s["top1_accuracy"]*100:.2f}% | {a["top1_accuracy"]*100:.2f}% | {(a["top1_accuracy"]-b["top1_accuracy"])*100:+.2f} 个百分点 |')
    cn+=['','属性识别是另一个指标，不能与物种分类准确率混用：','', '| 模型 | 基线属性 mAP | 完整方案属性 mAP | 移除属性分支后改变的预测数 / 200 |','|---|---:|---:|---:|']
    for m in models:
        b=get(m,'baseline');a=get(m,'attribute_region')
        cn.append(f'| {m} | {100*b["attribute_map"]:.2f}% | {100*a["attribute_map"]:.2f}% | {a["interventions"]["zero"]["flipped_predictions"]:.2f} |')
    cn+=['','此次新增了可见属性标签、未知项屏蔽和粗略证据框。60 张图片经助手查看并做暂定标注，其中训练 40 张、验证 20 张，各有一张全部未知；实际属性评价只覆盖 19 张验证图，某些属性极少出现，结果不稳定。尚未经过生物学专家复核。',
        '', '目前需要区分三个环节：①属性能否从图像识别；②关注位置是否对应属性；③属性是否提供全图分类器尚未掌握的物种区分信息。分类梯度、辅助监督和单图推理都已检查；属性指标改善但分类变化很小时，不能继续归咎于“AG 没有运行”。也不能仅凭这一轮断定唯一原因。',
        '', '粗属性如白色翅膀、绿色叶片可能在多个近似物种间共享；证据框又常包含整个主体。打乱监督仍能提高框内关注率，因此该指标不足以证明精确属性定位。当前最合理的下一步是为容易混淆的物种对增加真正区分它们的可见性状、更多独立图片级标注和更精确的部位框，再用未参与开发的数据确认。仅扩大模型或刻意削弱基线不能证明方法有效。',
        '', 'FG-CLIP 本身已有细粒度区域—文字预训练。本轮使用官方密集特征，将同一个属性适配模块加到它上面；这检验更适合局部对齐的主干能否受益，不是原论文 AG-CLIP 的严格复现，也不是完全未见过属性相关训练的基线。',
        '', '用户预测时只输入一张图片；模型内部仍保留固定属性文字向量。所有学习率和没有提升的结果均已保留。正式 100 物种实验及独立最终测试尚未完成。',
        '', '文件：`Visible_Attribute_All_Runs.csv` 为全部训练记录；`Visible_Attribute_Comparison.png` 为对比图；英文报告 `Visible_Attribute_Results.md` 含完整对照、分支干预与局限。','']
    (export/'Visible_Attribute_Summary_CN.md').write_text('\n'.join(cn),encoding='utf-8')
    (out/'Summary_CN.md').write_text('\n'.join(cn),encoding='utf-8')
    print(json.dumps(selected,indent=2))

if __name__=='__main__':
    main()
