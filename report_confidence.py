"""Report predeclared confidence pilot and audit score sensitivity."""
import json,csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import digest,write_json
from confidence_experiment import ROOT,OUT,CACHE,RUN,CONFIG,source,definition,Head,pool_weights,load_cache,inputs,evaluate,split_indices
from report_expanded import paired_statistics,holm

def main():
    torch.set_num_threads(4)
    c,m,boxes,bank=definition();r=json.loads((OUT/'results.json').read_text());assert r['source']==source()
    selected=json.loads((OUT/'selection_locked.json').read_text());p=torch.load(RUN/'predictions.pt',weights_only=True)
    assert len(r['rows'])==42 and len(p)==42
    for item in selected['checkpoints']:
        assert digest(RUN/f"{item['backbone']}_{item['seed']}/{item['variant']}.pt")==item['checkpoint_sha256']
    summary=[];pairs=[];sensitivity=[];coverage=[];replay=0.
    for name in c['backbones']:
        d=load_cache(name);assert digest(CACHE/f'{name}.pt')==selected['feature_hashes'][name]
        ix=split_indices(d,'test');identity=None;ensemble={}
        for variant in c['variants']:
            rows=[row for row in r['rows'] if row['backbone']==name and row['variant']==variant];assert len(rows)==3
            summary.append(dict(backbone=name,variant=variant,accuracy_mean=float(np.mean([x['top1_accuracy'] for x in rows])),accuracy_sd=float(np.std([x['top1_accuracy'] for x in rows],ddof=1)),macro_f1_mean=float(np.mean([x['macro_f1'] for x in rows])),training_accuracy_mean=float(np.mean([x['training_accuracy'] for x in rows])),parameters=rows[0]['parameters']))
            probabilities=[]
            for seed in c['seeds']:
                key=f'{name}/{seed}/{variant}';x=p[key];current=(x['paths'],x['labels'].tolist())
                if identity is not None:assert identity==current
                identity=current;probabilities.append(x['probabilities'])
                model=Head(name,variant,bank,c).cuda();ck=torch.load(RUN/f'{name}_{seed}/{variant}.pt',weights_only=True);model.load_state_dict(ck['state_dict'])
                metrics,probs,y=evaluate(model,d,ix)
                torch.testing.assert_close(probs,x['probabilities'],atol=1e-6,rtol=1e-5)
                replay=max(replay,float((probs-x['probabilities']).abs().max()))
                reference_row=next(row for row in rows if row['seed']==seed)
                assert abs(metrics['top1_accuracy']-reference_row['top1_accuracy'])<1e-10
                if variant=='ag_confidence':
                    for intervention in ('ag_uniform','ag_shuffled'):
                        model.variant=intervention;im,ip,_=evaluate(model,d,ix)
                        sensitivity.append(dict(backbone=name,seed=seed,intervention=intervention,changed_predictions=int((ip.argmax(-1)!=probs.argmax(-1)).sum()),accuracy=im['top1_accuracy'],accuracy_change_pp=(im['top1_accuracy']-metrics['top1_accuracy'])*100,max_probability_change=float((ip-probs).abs().max()),gate_tanh=float(model.gate.detach().tanh())))
            ensemble[variant]=torch.stack(probabilities).mean(0).argmax(-1).numpy()
        for reference in ('ag_uniform','text','ag_shuffled'):
            pairs.append(dict(backbone=name,reference=reference,target='ag_confidence',**paired_statistics(ensemble[reference],ensemble['ag_confidence'],identity[1],seed=20261003)))
        if name==c['backbones'][0]:
            for split in ('train','val','test'):
                ids=split_indices(d,split);v=d['valid'][ids];conf=d['confidence'][ids];n=v.sum(-1)
                weights=pool_weights(conf,v,'ag_confidence');both=n==2
                coverage.append(dict(split=split,images=len(ids),zero=int((n==0).sum()),one=int((n==1).sum()),two=int(both.sum()),mean_larger_weight_two=float(weights[both].max(-1).values.mean()) if both.any() else None,mean_confidence=float(conf[v].mean()) if v.any() else None))
    holm(pairs)
    write_json(OUT/'summary.json',summary);write_json(OUT/'paired_comparisons.json',pairs)
    write_json(OUT/'verification.json',dict(replayed_cells=42,max_probability_error=replay,confidence_coverage=coverage,weighted_checkpoint_interventions=sensitivity,tests='test_confidence:3 passed; test_statistics:2 passed',scope='No new tuning. Intervention labels are posthoc diagnostics, not model selection.'))
    with (OUT/'all_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(r['rows'][0]));writer.writeheader();writer.writerows(r['rows'])
    names=dict(visual='纯视觉分类头',random_codes='随机类别向量',text='图文适配',region_only='区域无属性文字',ag_uniform='AG等权聚合',ag_confidence='AG置信度加权',ag_shuffled='AG打乱置信度')
    lookup={(x['backbone'],x['variant']):x for x in summary}
    table=['| 方法 | EfficientNet-B0 | CLIP ViT-B/32* |','|---|---:|---:|']
    for v in c['variants']:
        values=[lookup[(n,v)] for n in c['backbones']]
        table.append('| '+names[v]+' | '+' | '.join(f"{x['accuracy_mean']*100:.2f} ± {x['accuracy_sd']*100:.2f}%" for x in values)+' |')
    cn=['# 检测置信度属性聚合：小规模匹配对照','',
        '本轮没有观察到检测置信度加权带来的额外分类收益：EfficientNet-B0等权/加权AG均为92.00%，CLIP ViT-B/32均为93.33%；打乱置信度也持平。固定加权检查点后改用均匀或反序权重，在全部六个模型/种子设置中均未改变测试预测类别，但概率有小幅变化，说明权重参与了计算，尚未转化为纠错。不能把这一结果解释为所有属性加权方法无效。','',
        '## 范围与方法','',
        '使用此前grounded_v1十个训练类别的500张已有图片：300张训练候选、100张验证、100张历史测试图。每个种子实际训练200张（每类20张），三个种子42/43/44；两个主干、七个分支，共42个最终模型。全部主干冻结。这是新十类闭集实验，不能与历史二十类或未见类别成绩直接比较。', '',
        '纯视觉训练普通线性分类头；图文适配将图片特征映射到固定CLIP类别文字向量；随机类别向量保留图文头参数化但去掉文字语义。AG将OWL-ViT检测区域特征与对应属性文字编码为token，聚合后与全图token输入同一个CAF模块。等权、真实置信度、打乱置信度三种AG结构及参数量完全一致，只有聚合权重不同；不是从旧token注意力模型直接修改再混用成绩。区域无文字对照保留相同模块但属性文字置零。', '',
        '权重为有效区域的c_i/sum(c_j)，无有效区域时严格回退全图，有效区域分数全零时退回均匀权重。打乱对照交换两个有效检测的分数；仅一个检测时无法改变权重。本轮不调温度、不调阈值，不加置信度辅助损失或低置信度回退策略。检测配置继承grounded_v1（最多两个区域、阈值0.05）。', '',
        '每个种子先训练200步视觉/随机/文字初始化，再给每个分支200步。AG共享该种子的文字初始化，并匹配属性模块初始化；AdamW学习率0.001，验证macro-F1选点、CE打破平局。所有42个检查点先锁定后计算测试成绩。图片、区域和属性选择不读取预测图片的真实类别。', '',
        '*CLIP主干始终保留图文预训练；纯视觉仅表示当前分类头不使用文字。普通线性头与图文投影头参数量不同，短统一预算不代表各方法最优调参。本实现是AG-CLIP-inspired，不是原论文官方代码或严格复现。', '',
        '## 结果','', '以下为三个种子的平均准确率 ± 样本标准差。', '',*table,'',
        '## 置信度加权的额外收益','',
        '| 主干 | 加权−等权（种子均值，百分点） | 集成差值 | 纠正/误改 | 95%区间（未校正） | Holm p |',
        '|---|---:|---:|---:|---|---:|']
    for n in c['backbones']:
        q=next(x for x in pairs if x['backbone']==n and x['reference']=='ag_uniform')
        delta=(lookup[(n,'ag_confidence')]['accuracy_mean']-lookup[(n,'ag_uniform')]['accuracy_mean'])*100
        cn.append(f"| {n} | {delta:+.2f} | {q['improvement_percentage_points']:+.2f} | {q['ag_fixes']}/{q['ag_breaks']} | {q['paired_stratified_bootstrap_95ci_pp']} | {q['holm_adjusted_p']:.4f} |")
    cn += ['', '统计基于三个种子概率平均后的100张配对图片，不将300次预测当成独立样本；六项预先声明对比统一Holm校正（每个主干的加权AG对等权AG、图文适配、打乱置信度）。置信区间条件于已训练模型和历史测试图片，不能证明独立泛化；样本量小也限制检验能力。相同预测产生的[0,0]自助区间仅是当前样本重采样的退化结果，不代表总体效应确定为零。所有对比详见paired_comparisons.json。', '',
           '## 置信度能够改变多少图片','', '| 图片集 | 无区域 | 一个区域 | 两个区域 | 两区域中较大权重的均值 |','|---|---:|---:|---:|---:|']
    for x in coverage:cn.append(f"| {x['split']} ({x['images']}) | {x['zero']} | {x['one']} | {x['two']} | {x['mean_larger_weight_two']:.3f} |")
    cn += ['', '只有两个有效区域且分数不同的图片，置信度聚合才与等权不同。检测置信度不等同于属性正确性或物种区分能力；本轮没有人工验证框与属性是否准确对应。归一化也会丢失绝对可靠程度，因此所有分数偏低的问题仍然存在。', '',
           '## 固定加权模型后的权重干预','', '| 主干/种子 | 改为 | 改变预测数/100 | 准确率变化（百分点） |', '|---|---|---:|---:|']
    for x in sensitivity:cn.append(f"| {x['backbone']}/{x['seed']} | {names[x['intervention']]} | {x['changed_predictions']} | {x['accuracy_change_pp']:+.2f} |")
    cn += ['', '这些干预固定模型参数，仅改变推理权重，用来检查模型是否使用权重；它们不是新的选参结果。完整概率变化、门控值和42组检查点重放核验见verification.json。','',
           '## 复现','', '依赖本机既有expanded20图片、CLIP文字缓存和grounded_v1检测缓存；图片、特征、检测缓存和模型权重不上传GitHub。','', '```powershell', '.\\.venv\\Scripts\\python.exe -m unittest test_confidence test_statistics -v', '.\\.venv\\Scripts\\python.exe confidence_experiment.py all', '.\\.venv\\Scripts\\python.exe report_confidence.py','```','']
    (OUT/'summary_zh.md').write_text('\n'.join(cn),encoding='utf-8')
    en=['# Detector-confidence attribute pooling pilot','',
        'No additional classification gain was observed: uniform and confidence AG both achieved 92.00% for EfficientNet-B0 and 93.33% for CLIP ViT-B/32. Shuffled confidence tied these results. Fixed-checkpoint weight interventions changed probabilities slightly but no predicted labels in all six backbone/seed settings. This is evidence of low decision sensitivity in this pilot, not proof that attribute weighting is universally ineffective.','',
        'Ten classes, 500 existing images: 300 training candidates, 100 validation and 100 historically evaluated test images. Each run uses 20 training images per class. Two frozen backbones, three seeds, seven arms: 42 final models. Each receives 200 initialization steps plus 200 comparison steps; all selections are validation-only and sealed before scoring.', '',
        'Uniform, confidence-weighted and shuffled-confidence AG heads share identical architecture and initialization. Region/text tokens are pooled, then passed with a global token to CAF. Only pooling weights change. Actual cached OWL-ViT scores are used. Missing detections fall back to the global path. The original CLIP pretraining remains in every CLIP-backbone readout. This is a new ten-class closed-set adaptation, not an exact AG-CLIP replication or the prior twenty-class table.', '',
        '| Backbone | Method | Accuracy mean (%) | Seed SD (pp) |', '|---|---|---:|---:|']
    for x in summary:en.append(f"| {x['backbone']} | {x['variant']} | {x['accuracy_mean']*100:.2f} | {x['accuracy_sd']*100:.2f} |")
    en += ['', 'Paired tests use one three-seed ensemble prediction per image. Holm correction covers six predeclared contrasts: confidence AG versus uniform AG, text adaptation, and shuffled confidence for both backbones. See paired_comparisons.json for fixes/breaks, exact McNemar tests and conditional class-stratified bootstrap intervals. See verification.json for detector coverage and fixed-checkpoint weight interventions.', '',
           'Limitations: historically reused small test set, frozen features, short common budget rather than separately optimal tuning, at most two detections, uncalibrated detector confidence, no verified attribute correctness, and one fixed random-prototype bank. No post-result tuning was performed. No significance is not proof of no effect. A degenerate [0,0] bootstrap interval for identical observed predictions does not establish a zero population effect.','']
    (OUT/'summary_en.md').write_text('\n'.join(en),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    labels=['Visual','Random','Text','Regions','AG mean','AG conf.','AG shuffle']
    for ax,n in zip(axes,c['backbones']):
        x=[lookup[(n,v)] for v in c['variants']]
        ax.bar(range(7),[a['accuracy_mean']*100 for a in x],yerr=[a['accuracy_sd']*100 for a in x],capsize=3,color=['#8796a4','#c49a53','#477caf','#bda7c4','#529584','#256f51','#88805c'])
        ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.set_ylim(0,100);ax.set_title(n)
    axes[0].set_ylabel('Accuracy (%), mean +/- seed SD')
    fig.suptitle('10 classes | 20 shots | 3 seeds | Frozen features | Reused test set')
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summary=summary,paired=pairs,coverage=coverage,sensitivity=sensitivity),indent=2))

if __name__=='__main__':main()
