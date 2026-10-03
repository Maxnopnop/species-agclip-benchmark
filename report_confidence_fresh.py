"""Audit fresh holdout and report all locked confidence comparisons."""
import json,csv
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import confidence_fresh as f
from confidence_experiment import pool_weights
from report_expanded import paired_statistics,holm
from data_tools import write_json,digest

def main():
    torch.set_num_threads(4);c=json.loads(f.CONFIG.read_text());m=json.loads((f.DATA/'manifest.json').read_text());s=f.source(m)
    seal=f.check_seal(s);r=json.loads((f.OUT/'results.json').read_text());assert r['source']==s
    assert datetime.fromisoformat(seal['locked_at_utc'])<datetime.fromisoformat(r['scored_at_utc'])
    assert len(r['rows'])==42
    assert not ({x['source_id'] for x in m['classes']}&set(m['excluded_species']))
    assert len({x['image_id'] for x in m['rows']})==500
    assert len({x['sha256'] for x in m['rows']})==500
    assert len({x['pixels_sha256'] for x in m['rows']})==500
    for x in m['rows']:assert (x['split']!='test')==x['official_train']
    historical=json.loads((f.CACHE/'historical_fingerprints.json').read_text())['fingerprints']
    min_p=64;min_d=64
    # Independent replay of the fingerprint thresholds against prior and earlier new images.
    refs=historical.copy()
    for x in m['rows']:
        assert not any(x['sha256']==a['sha256'] or x['pixels_sha256']==a['pixels_sha256'] for a in refs)
        p=min((int(x['phash'])^int(a['phash'])).bit_count() for a in refs)
        d=min((int(x['dhash'])^int(a['dhash'])).bit_count() for a in refs)
        assert p>c['dedup_hamming_threshold'] and d>c['dedup_hamming_threshold']
        min_p=min(min_p,p);min_d=min(min_d,d);refs.append(x)
    b=f.text_bank(m,s);pred=torch.load(f.RUN/'predictions.pt',weights_only=True)
    summary=[];pairs=[];coverage=[];interventions=[];maxerror=0.
    for n in c['backbones']:
        data=f.cache(n,'test');ix=f.split_indices(data,'test');ensemble={};identity=None
        for v in c['variants']:
            rows=[x for x in r['rows'] if x['backbone']==n and x['variant']==v]
            summary.append(dict(backbone=n,variant=v,accuracy_mean=float(np.mean([x['top1_accuracy'] for x in rows])),accuracy_sd=float(np.std([x['top1_accuracy'] for x in rows],ddof=1)),macro_f1_mean=float(np.mean([x['macro_f1'] for x in rows])),training_accuracy_mean=float(np.mean([x['training_accuracy'] for x in rows]))))
            probs=[]
            for seed in c['seeds']:
                x=pred[f'{n}/{seed}/{v}'];current=(x['paths'],x['labels'].tolist())
                if identity is not None:assert identity==current
                identity=current;probs.append(x['probabilities'])
                ck=torch.load(f.RUN/f'{n}_{seed}/{v}.pt',weights_only=True)
                assert set(ck['train_paths'])=={x['path'] for x in m['rows'] if x['split']=='train'}
                model=f.Head(n,v,b,c).cuda();model.load_state_dict(ck['state_dict']);metrics,p,y=f.evaluate(model,data,ix)
                torch.testing.assert_close(p,x['probabilities'],atol=1e-6,rtol=1e-5);maxerror=max(maxerror,float((p-x['probabilities']).abs().max()))
                assert abs(metrics['top1_accuracy']-next(a['top1_accuracy'] for a in rows if a['seed']==seed))<1e-10
                if v=='ag_confidence':
                    for change in ('ag_uniform','ag_shuffled'):
                        model.variant=change;im,ip,_=f.evaluate(model,data,ix)
                        interventions.append(dict(backbone=n,seed=seed,intervention=change,changed_predictions=int((p.argmax(-1)!=ip.argmax(-1)).sum()),accuracy_change_pp=(im['top1_accuracy']-metrics['top1_accuracy'])*100,max_probability_change=float((p-ip).abs().max())))
            ensemble[v]=torch.stack(probs).mean(0).argmax(-1).numpy()
        for ref in ('ag_uniform','text','ag_shuffled'):
            pairs.append(dict(backbone=n,reference=ref,target='ag_confidence',**paired_statistics(ensemble[ref],ensemble['ag_confidence'],identity[1],seed=202610031)))
        if n==c['backbones'][0]:
            for stage in ('development','test'):
                data2=f.cache(n,stage)
                for split in (('train','val') if stage=='development' else ('test',)):
                    ids=f.split_indices(data2,split);valid=data2['valid'][ids];conf=data2['confidence'][ids];count=valid.sum(-1);both=count==2
                    w=pool_weights(conf,valid,'ag_confidence')
                    coverage.append(dict(split=split,images=len(ids),zero=int((count==0).sum()),one=int((count==1).sum()),two=int(both.sum()),mean_larger_weight=float(w[both].max(-1).values.mean()) if both.any() else None))
    holm(pairs)
    write_json(f.OUT/'summary.json',summary);write_json(f.OUT/'paired_comparisons.json',pairs)
    write_json(f.OUT/'verification.json',dict(replayed_cells=42,max_probability_error=maxerror,minimum_phash_distance=min_p,minimum_dhash_distance=min_d,prior_and_cross_split_duplicate_matches=0,source_class_overlap=0,coverage=coverage,interventions=interventions,tests_passed=7,seal_before_score=True))
    write_json(f.OUT/'classes.json',m['classes'])
    with (f.OUT/'all_results.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(r['rows'][0]));writer.writeheader();writer.writerows(r['rows'])
    names=dict(visual='纯视觉分类头',random_codes='随机类别向量',text='图文适配',region_only='区域无属性文字',ag_uniform='AG等权聚合',ag_confidence='AG置信度加权',ag_shuffled='AG打乱置信度')
    lookup={(x['backbone'],x['variant']):x for x in summary}
    table=['| 方法 | EfficientNet-B0 | CLIP ViT-B/32* |','|---|---:|---:|']
    for v in c['variants']:table.append('| '+names[v]+' | '+' | '.join(f"{lookup[(n,v)]['accuracy_mean']*100:.2f} ± {lookup[(n,v)]['accuracy_sd']*100:.2f}%" for n in c['backbones'])+' |')
    cn=['# 未使用物种与新图片：置信度聚合独立数据复测','',
        '新数据上仍未证明检测置信度加权的显著优势。EfficientNet-B0的等权AG为95.50%，加权为96.00%（种子均值+0.50个百分点）；CLIP ViT-B/32两者均为96.67%。六项预先声明的配对检验均未得到校正后显著正向提升。不能将小幅均值变化写成稳定收益，也不能据此判定所有加权方法无效。','',
        '## 数据隔离','',
        f"此前项目使用的120个CUB物种全部排除，从余下物种按固定随机顺序及数据数量要求选出10种。使用500张此前项目未用图片；每类20张训练、10张验证、20张测试，共200/100/200。训练和验证来自CUB官方训练部分，测试来自官方测试部分。按文件哈希、解码像素哈希及64位pHash/dHash阈值4筛除历史与新样本之间的重复/疑似近重复；对照历史图片共{len(historical)}张。筛查后重新审计无阈值内匹配，最小pHash/dHash距离为{min_p}/{min_d}。近重复判定仍是启发式，不能排除所有同一拍摄对象或观察事件。", '',
        '类名和划分根据元数据确定，未使用模型准确率或新测试属性标注。所有种子使用相同200张训练图片，变化来自初始化与批次顺序。新物种相对于旧项目实验未见，但在本轮闭集训练中可见，不是零样本评估。项目新图片也可能与基础模型预训练重叠。','',
        '## 方法与测试保护','',
        '保持confidence_v1的两个主干、七分支、三个种子、200+200步、AdamW学习率0.001、CAF结构和置信度公式不变；直接复用原Head/train函数。针对鸟类使用此前训练集筛选的固定24条鸟类属性，不根据新数据成绩选词。OWL-ViT所有图片使用同一词表，最多两个区域、阈值0.05。主干冻结。', '',
        '准备阶段只允许检查测试图片身份、重复性和官方划分，不运行测试模型预测。先对300张训练/验证图片检测及提特征；42个模型按验证macro-F1/CE选定后，锁定检查点及开发资产哈希，再允许对200张新测试图检测、提特征并作一次最终评分。重放用于核验同一固定模型，未据此改超参数或挑变体。', '',
        '*CLIP主干始终保留CLIP预训练，纯视觉指当前分类头不读文字。这里的AG为轻量改编，不是官方完整AG-CLIP。新鸟类数据与上一轮混合物种数据分布不同，不能把跨数据集绝对准确率变化归因于算法。统一短预算也不代表各基线最优调参。', '',
        '## 新测试集结果','', '均值 ± 三种子样本标准差；全部结果保留，包括负结果。', '',*table,'',
        '## 加权相对等权的收益','', '| 主干 | 种子均值差（百分点） | 集成差值 | 纠正/误改 | 95%区间 | Holm p |','|---|---:|---:|---:|---|---:|']
    for n in c['backbones']:
        p=next(x for x in pairs if x['backbone']==n and x['reference']=='ag_uniform')
        delta=(lookup[(n,'ag_confidence')]['accuracy_mean']-lookup[(n,'ag_uniform')]['accuracy_mean'])*100
        cn.append(f"| {n} | {delta:+.2f} | {p['improvement_percentage_points']:+.2f} | {p['ag_fixes']}/{p['ag_breaks']} | {p['paired_stratified_bootstrap_95ci_pp']} | {p['holm_adjusted_p']:.4f} |")
    cn += ['', '统计以三个种子概率集成后的200张配对图片为单位，六项预先声明对比统一Holm校正（加权AG对等权AG、文字适配、打乱置信度，各两个主干）。95%自助区间未做多重校正，只条件于固定模型与样本；相同预测产生的[0,0]不代表总体效应确定为零。未通过显著性检验不等于证明无效。','',
           '## 检测覆盖','', '| 集合 | 无区域 | 一个区域 | 两个区域 | 两区域较大权重均值 |','|---|---:|---:|---:|---:|']
    for x in coverage:cn.append(f"| {x['split']} ({x['images']}) | {x['zero']} | {x['one']} | {x['two']} | {x['mean_larger_weight']} |")
    cn += ['', '归一化权重仅能直接改变多个有效区域的聚合；检测置信度不保证正确属性或正确位置。固定加权检查点的均匀/反序权重干预保存在verification.json；这些诊断不用于模型选择。', '',
           '## 结论边界与复现','',
           '本轮降低了反复利用旧测试集的风险，但不能保证训练不过拟合，也不能证明适用于所有物种。若继续修改方法，这200张测试图也已成为用过的测试集，不能再作为完全独立的最终证据。预训练重叠、同一观察者/个体分组、定位可靠性与小样本统计能力仍有限制。', '',
           '```powershell','.\\.venv\\Scripts\\python.exe -m unittest test_confidence_fresh test_confidence test_statistics -v','.\\.venv\\Scripts\\python.exe confidence_fresh.py all','.\\.venv\\Scripts\\python.exe report_confidence_fresh.py','```','',
           '依赖本地官方CUB图片、已有基础模型与OWL-ViT权重、历史清单。GitHub仅包含代码、配置和结果，不包含照片、检测缓存、特征或权重。','']
    (f.OUT/'summary_zh.md').write_text('\n'.join(cn),encoding='utf-8')
    en=['# Fresh-data validation of confidence-weighted attribute aggregation','',
        'No significant confidence-weighting advantage was established. EfficientNet-B0 improved from 95.50% uniform AG to 96.00% weighted AG in seed-mean accuracy (+0.50 pp); CLIP ViT-B/32 tied at 96.67%. None of the six predeclared paired contrasts had a positive gain surviving Holm correction.','',
        'Ten CUB species excluded from all 120 previously used project CUB species; 500 project-unused images, with 200 training, 100 validation and 200 test images. Official source train/test boundaries respected. Exact file hashes, decoded-pixel hashes and pHash/dHash distance <=4 were screened against historical images and earlier accepted fresh images. This is heuristic duplicate protection, not observation-level or foundation-pretraining independence.', '',
        'Same two backbones, seven arms, three seeds, frozen features and 200+200-step training recipe as confidence_v1. The original Head/train implementation is reused. A fixed previously selected vocabulary of 24 bird attributes replaces the mixed-species vocabulary. All seeds use the same training images. Validation-only checkpoint choices are sealed before any test grounding or feature extraction. No tuning follows final scores.', '',
        '| Backbone | Method | Accuracy mean (%) | Seed SD (pp) |','|---|---|---:|---:|']
    for x in summary:en.append(f"| {x['backbone']} | {x['variant']} | {x['accuracy_mean']*100:.2f} | {x['accuracy_sd']*100:.2f} |")
    en += ['', 'Six predeclared paired comparisons use 200 three-seed ensemble predictions: confidence AG versus uniform AG, text, and shuffled confidence for both backbones. Exact McNemar, class-stratified bootstrap intervals, and Holm correction are reported in paired_comparisons.json. Replay and confidence interventions appear in verification.json.', '',
           'This is closed-set recognition on new project species, not zero-shot evaluation. CLIP retains language pretraining in the visual readout. New data reduces adaptive reuse of the old test set but does not guarantee absence of training overfitting. Bird-only scores are not directly comparable with the prior mixed-species dataset. Once inspected, this test set is no longer untouched for subsequent method development.','']
    (f.OUT/'summary_en.md').write_text('\n'.join(en),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    labels=['Visual','Random','Text','Regions','AG mean','AG conf.','AG shuffle']
    for ax,n in zip(axes,c['backbones']):
        x=[lookup[(n,v)] for v in c['variants']]
        ax.bar(range(7),[a['accuracy_mean']*100 for a in x],yerr=[a['accuracy_sd']*100 for a in x],capsize=3,color=['#8796a4','#c49a53','#477caf','#bda7c4','#529584','#256f51','#88805c'])
        ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.set_ylim(0,100);ax.set_title(n)
    axes[0].set_ylabel('Accuracy (%), mean +/- seed SD');fig.suptitle('10 previously unused CUB species | 200 fresh test images | 3 seeds')
    fig.savefig(f.OUT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summary=summary,paired=pairs,coverage=coverage,interventions=interventions),indent=2))

if __name__=='__main__':main()
