"""Aggregate every completed diagnostic, including negative and step-zero results."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json

VERSIONS=['cub_tokens_v1','cub_bottleneck_v1','cub_rich_v1','cub_sparse_v1','cub_b16_v1','cub_regional_tokens_v1']
OUT=ROOT/'reports/cub_followup_v1'
LABELS={'cub_tokens_v1':'B/32 · 24 · dense','cub_bottleneck_v1':'B/32 · attribute-only','cub_rich_v1':'B/32 · 158 · dense','cub_sparse_v1':'B/32 · 158 · sparse/contrast','cub_b16_v1':'B/16 · 158 · dense','cub_regional_tokens_v1':'B/32 · regional fine-tuning'}


def mean_sd(values):return float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.


def paired_summary(version,results,control,seen_classes=None):
    ag={r['seed']:r for r in results if r['variant']=='tokens'};base={r['seed']:r for r in results if r['variant']==control}
    seeds=sorted(set(ag)&set(base))
    if not seeds:return None
    pp=[];bb=[]
    for seed in seeds:
        x=torch.load((ROOT/ag[seed]['checkpoint']).parent/'final_predictions.pt',weights_only=True)
        y=torch.load((ROOT/base[seed]['checkpoint']).parent/'final_predictions.pt',weights_only=True)
        assert torch.equal(x['labels'],y['labels'])
        pp.append(x['predictions'].numpy());bb.append(y['predictions'].numpy())
    labels=x['labels'].numpy();pp=np.stack(pp);bb=np.stack(bb);classes=sorted(set(labels));seen=set(range(0,20,2) if seen_classes is None else seen_classes)
    def h(pred,indices):
        per=np.stack([(pred[:,indices[labels[indices]==cl]]==cl).mean(-1)*100 for cl in classes],-1)
        s=per[:,[cl in seen for cl in classes]].mean(-1);u=per[:,[cl not in seen for cl in classes]].mean(-1)
        return np.divide(2*s*u,s+u,out=np.zeros_like(s),where=(s+u)>0).mean()
    rng=np.random.default_rng(1729);delta=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(np.flatnonzero(labels==cl),size=(labels==cl).sum(),replace=True) for cl in classes]);delta.append(h(pp,ix)-h(bb,ix))
    return dict(version=version,control=control,seeds=seeds,mean_H_delta=float(np.mean([ag[s]['metrics']['H']-base[s]['metrics']['H'] for s in seeds])),conditional_95_interval=np.percentile(delta,[2.5,97.5]).tolist(),changed_predictions_per_seed=[int((a!=b).sum()) for a,b in zip(pp,bb)],scope='Paired within-class image bootstrap, shared resamples across three fixed trained seeds. Conditional on this split/probe/checkpoint search; not a population-of-training-runs interval or multiplicity-adjusted confirmation.')


def main():
    OUT.mkdir(parents=True,exist_ok=True);completed={};rows=[];comparisons=[];cells=0
    for version in VERSIONS:
        path=ROOT/f'reports/{version}/final_results.json'
        if not path.exists():continue
        result=json.loads(path.read_text(encoding='utf-8'));completed[version]=result
        count=len(list((ROOT/'runs'/version).glob('*/result.json')));cells+=count
        variants=list(dict.fromkeys(r['variant'] for r in result['results']))
        for variant in variants:
            entries=[r for r in result['results'] if r['variant']==variant];row=dict(version=version,variant=variant,seeds=len(entries),selected_steps='/'.join(str(r['step']) for r in entries),step_zero_count=sum(r['step']==0 for r in entries))
            for key in ['H','ZSL','S','U']:row[key+'_mean'],row[key+'_sd']=mean_sd([r['metrics'][key] for r in entries])
            rows.append(row)
        for control in ['region_only','pooled','constant','identity','no_attribute_loss']:
            record=paired_summary(version,result['results'],control)
            if record:comparisons.append(record)
    write_json(OUT/'paired_contrasts.json',comparisons)
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=['| Experiment | Variant | H, mean ± SD | ZSL, mean ± SD | Selected steps |','|---|---|---:|---:|---|']
    for r in rows:table.append(f'| {r["version"]} | {r["variant"]} | {r["H_mean"]:.2f} ± {r["H_sd"]:.2f} | {r["ZSL_mean"]:.2f} ± {r["ZSL_sd"]:.2f} | {r["selected_steps"]} |')
    native_lines='; '.join(f'{version}: H={d["native"]["H"]:.2f}%, ZSL={d["native"]["ZSL"]:.2f}%' for version,d in completed.items() if version in ['cub_tokens_v1','cub_b16_v1'])
    pending=[v for v in VERSIONS if v not in completed]
    zh=f'''# 冻结全图分支与独立属性 token：连续诊断记录

已完成 {len(completed)} 个实验版本、{cells} 组训练。未完成版本：{', '.join(pending) or '无'}。本报告自动纳入全部已完成版本，不只展示最好的一组。

**当前结果不支持 AG 具有稳定、独立的分类增益。** 正确属性与无文字、平均融合、错误关联、常量属性等对照必须一起看。属性识别有提升，不等于物种分类有提升；更换主干和分数校准的收益也不能算作 AG 收益。

原始主干：{native_lines}。

## 本轮固定条件与公平性

复用 20 个 CUB 物种、200 张已见训练图片、180 张开发图片和 320 张评估图片；最终未见类别 6 个。所有学习率、检查点和校准系数只在开发集选取，再锁定后评估。但这批评估数据在此前实验中已经被查看，因此整轮属于探索性复测，不是全新的独立确认。

每个融合实验使用种子 42/43/44；这只覆盖融合初始化与训练批次的变化，数据划分、预训练权重和属性探针仍固定。均值后的 SD 是三个种子的样本标准差。不能因为 SD=0 就声称稳定推广：有些模型的分数不同，但 argmax 标签相同。

冻结特征版本的全图与区域视觉编码器均冻结，只有属性探针和融合模块按阶段训练。区域微调版本另外解冻区域编码器末端两层，全图分数保持冻结。所有前向推理只读图片、固定文字库与模型权重，不读真实类别、真实属性或真实部位位置。

## 主要诊断

1. **实现／缓存错误：** 独立 token、梯度、冻结参数、无效区域屏蔽和相同容量对照已核查；单张照片重新检测、提取特征的输出与缓存结果一致，已完成检查见各版本 deployment_verification.json。
2. **属性被平均：** 独立 token 确实保留各属性身份，但仅修复平均融合没有带来稳定分类增益。不同余弦诊断的计算对象不同，不能把跨图片平均文字余弦与视图内 token 余弦直接当作同一个指标比较。
3. **权重太小：** 对已训练 token 模型在开发集扫描更强的残差权重，H 反而从约 76.3% 降低到 43%–47%。这只排除该模型固定方向上简单放大权重的办法，不排除其他训练机制。
4. **视觉捷径：** 属性瓶颈分支没有视觉输入／视觉查询；减去常量属性输出后，只有属性变化能改变预测。更长训练仍未产生稳定收益；两个种子选中第 0 步，第三个训练后损害 GZSL。第 0 步不是成功的属性优化。
5. **属性覆盖不足：** 相同开发集和可见性规则下，仅用属性做已见类别原型分类，24→158 条属性的预测属性准确率为 87%→95%；真实属性诊断为 60%→84%。这是开发集诊断，真实属性不可作为部署输入，也不是理论性能上界。扩大属性集合后，融合分类仍没有稳定收益。
6. **低置信属性／损失函数：** 用 158 条属性做全部保留／每视图 top-8，以及分类损失／加入双向多正例对比损失的交叉对照；结果完整保留在 sparse 版本，不能只挑提升的一项。
7. **主干能力：** B/16 的原始模型优于 B/32，但正确、错误和常量属性融合仍相近。更换骨干同时改变权重与 patch 大小，不能声称这严格隔离了分辨率因素。
8. **已见类偏差：** B/32 原始分数经开发集选取 gamma=0.25 校准后，评估 H 为 78.84%。必须给所有对照相同校准机会；结果见 calibration.json。B/16 开发集选择 gamma=0。

以上是对具体实现和有限搜索范围的排查，并不意味着已经排除了所有可能的 AG 方法、数据划分或超参数，也不能据此否定原论文。

## 完整成绩（百分比）

H 为已见／未见类别平均准确率的调和平均；ZSL 只在最终 6 个未见候选类别间分类。两者不能混为一谈。

'''+ '\n'.join(table)+'''

## 证据、局限和论文定位

paired_contrasts.json 保存配对、按类别分层的图片 bootstrap 区间，条件是当前三个已训练种子与当前数据划分；它没有覆盖所有训练随机性、模型搜索或多重比较。大量后续实验属于假设探索，不能把某一次正向差值当成确认性显著提升。

使用的 CUB 人工属性是带噪声的众包标注。“确定度≥3且部位点在裁剪内”不保证完整部位可见，也不保证属性对物种具有足够区分性。更高维属性 mAP 与原 24 维 mAP 的类别组成不同，不应直接比较数值；coverage 结果另外报告共同 24 维指标。

原论文使用属性挖掘、OWL-ViT 区域对应、属性编码器、CAF 与图文对比训练，CUB 使用 150/50 类划分。本项目的冻结／局部微调实验是机制诊断改编，不能与论文完整训练成绩直接比较。[AG-CLIP 作者论文](https://www.researchgate.net/publication/399822501_AG-CLIP_Attribute-Guided_CLIP_for_Zero-Shot_Fine-Grained_Recognition)。
'''
    (OUT/'summary_zh.md').write_text(zh,encoding='utf-8')
    en=f'''# Frozen-global attribute-token diagnostics

Completed versions: {len(completed)}; training cells: {cells}. Pending: {', '.join(pending) or 'none'}.

No stable attribute-specific classification benefit is established. Attribute prediction, backbone replacement and seen/unseen calibration must be distinguished from AG gains. All controls and step-zero selections are retained below.

The study reuses 20 CUB species, 200 training, 180 development and 320 evaluation images. Evaluation had been inspected in earlier experiments; follow-ups are exploratory. Seeds 42/43/44 vary fusion initialization and batches, not the split or pretrained weights/probe. H is the harmonic mean of seen/unseen per-class accuracy; ZSL uses only six unseen candidates. Native results: {native_lines}.

Independent tokens, an attribute-only residual, 24 versus 158 attributes, confidence-based sparse tokens, symmetric contrastive loss, a B/16 backbone and regional visual fine-tuning are separate diagnostics. Conditional paired bootstrap intervals in paired_contrasts.json do not account for adaptive model search or all training randomness. Human-attribute oracle results in the coverage diagnostic are not deployable results or information-theoretic upper bounds.

'''+ '\n'.join(table)+'\n'
    (OUT/'summary_en.md').write_text(en,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');colors={'tokens':'#2563eb','region_only':'#d97706','pooled':'#08916b','constant':'#a855f7'}
    versions=list(completed)
    for col,metric in enumerate(['H','ZSL']):
        ax=axes[col]
        for i,version in enumerate(versions):
            d=completed[version];ax.scatter(i,d['native'][metric],marker='x',s=65,color='black',label='Native' if i==0 else None,zorder=5)
            for j,(variant,color) in enumerate(colors.items()):
                rr=[r for r in rows if r['version']==version and r['variant']==variant]
                if rr:ax.errorbar(i+(j-1.5)*.13,rr[0][metric+'_mean'],yerr=rr[0][metric+'_sd'],fmt='o',color=color,capsize=3,label=variant if i==0 else None)
        ax.set_xticks(range(len(versions)),[LABELS[v] for v in versions],rotation=30,ha='right');ax.set_ylim(0,100);ax.set_ylabel(f'{metric} (%)');ax.grid(axis='y',alpha=.2);ax.set_title(metric+' · mean ± seed SD')
    axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Exploratory reused-data diagnostics · native and matched controls')
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    write_json(OUT/'status.json',dict(completed=list(completed),pending=pending,training_cells=cells))
    print(f'Report updated: {cells} completed training cells; pending {pending}')


if __name__=='__main__':main()
