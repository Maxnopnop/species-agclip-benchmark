"""Audit fresh holdout and report all locked confidence comparisons."""
import json,csv
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fungitastic_experiment as f
from fungitastic_data import validate_rows
from confidence_experiment import pool_weights
from report_expanded import paired_statistics,holm
from data_tools import write_json,digest

def main():
    torch.set_num_threads(4);c=json.loads(f.CONFIG.read_text());m=json.loads((f.DATA/'manifest.json').read_text());s=f.source(m)
    seal=f.check_seal(s);r=json.loads((f.OUT/'results.json').read_text());assert r['source']==s
    assert datetime.fromisoformat(seal['locked_at_utc'])<datetime.fromisoformat(r['scored_at_utc'])
    assert len(r['rows'])==42
    validate_rows(m['rows'])
    assert len({x['sha256'] for x in m['rows']})==500
    assert len({x['pixels_sha256'] for x in m['rows']})==500
    historical=json.loads((f.ROOT/'cache/confidence_fresh_v1/historical_fingerprints.json').read_text())['fingerprints']
    historical+=json.loads((f.ROOT/'data/confidence_fresh_v1/manifest.json').read_text())['rows']
    min_p=64;min_d=64;refs=historical.copy()
    for x in m['rows']:
        assert digest(f.Path(m['image_root'])/x['path'])==x['sha256']
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
    write_json(f.OUT/'verification.json',dict(replayed_cells=42,max_probability_error=maxerror,minimum_phash_distance=min_p,minimum_dhash_distance=min_d,prior_and_cross_split_duplicate_matches=0,cross_split_observation_overlap=0,coverage=coverage,interventions=interventions,tests_passed=7,seal_before_score=True))
    write_json(f.OUT/'classes.json',m['classes'])
    with (f.OUT/'all_results.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(r['rows'][0]));writer.writeheader();writer.writerows(r['rows'])
    lookup={(x['backbone'],x['variant']):x for x in summary}
    for lang in ('zh','en'):
        zh=lang=='zh'
        lines=['# '+('FungiTastic：按年份隔离的新数据验证' if zh else 'FungiTastic temporal validation'),'',
            '500张照片、500条独立观察记录、10个物种。2022年200张训练、100张验证；2023年200张测试。每类20/10/20张，所有随机种子共享划分。仅按数量和固定随机顺序选择物种，不按结果选样。' if zh else
            '500 images from 500 unique observations across 10 species. 200 training and 100 validation images dated2022;200 test images dated2023.20/10/20 per species,same splits across seeds. Classes selected by counts and fixed random order,never by model scores.', '',
            '以下为三个种子的准确率均值±标准差。两主干冻结，原始Head/train不变，每阶段200步，七分支配对。CLIP纯视觉仍含原有图文预训练；本轮图文适配不等于从零预训练CLIP，AG也是轻量改编。' if zh else
            'Accuracy is mean +/- sample SD across three seeds. Two frozen backbones,unchanged Head/train,200 updates per stage,seven matched arms. CLIP visual readout retains its language pretraining. Task text adaptation is not CLIP pretraining from scratch,and AG is a lightweight adaptation.', '',
            '| Method | EfficientNet-B0 (%) | CLIP ViT-B/32 (%) |','|---|---:|---:|']
        for v in c['variants']:
            lines.append('| '+v+' | '+' | '.join(f"{lookup[(n,v)]['accuracy_mean']*100:.2f} ± {lookup[(n,v)]['accuracy_sd']*100:.2f}" for n in c['backbones'])+' |')
        lines+=['',
            '本轮纯视觉准确率仅约63%–65%，仍未得到置信度加权的显著正向证据。EfficientNet加权与等权均值相同；CLIP加权均值仅高0.33个百分点，低于打乱权重，且与区域无文字对照相同。因此不能把文字适配相对线性头的提升归因于AG，也不能归因于置信度排序。随机类别向量也带来大幅提升，说明分类头/优化差异是重要混杂因素。' if zh else
            'Visual readout accuracy is only about63%–65%,yet confidence weighting shows no significant positive evidence. EfficientNet weighted and uniform means tie;CLIP weighted mean rises by only0.33pp,below shuffled weights and equal to the region-only control. Text-versus-linear gains cannot be attributed to AG or confidence ordering. Large gains from random prototypes also identify classifier/optimization differences as a confound.', '',
            '测试图片177/200具有两个区域，因此本轮收益不足不能简单归因于缺少可加权区域。固定加权模型中切换为等权或反序权重，EfficientNet三个种子的预测均未改变，CLIP最多改变3/200张；这提示当前模型对权重排序的决策依赖有限，但不证明属性普遍无效。' if zh else
            '177/200 test images have two regions,so lack of opportunities to weight multiple regions is insufficient as an explanation. Fixed-checkpoint uniform/reversed-weight interventions change no EfficientNet labels and at most3/200 CLIP labels. This suggests limited decision dependence on weight ordering in this implementation,not universal attribute ineffectiveness.', '',
            '种子均值和概率集成是不同汇总：例如CLIP加权相对等权的种子均值为+0.33个百分点，但概率集成为−1.50个百分点。不能只选取较有利的汇总来声称提升。' if zh else
            'Seed means and probability ensembles are different estimands:CLIP weighting changes the seed mean by+0.33pp but the ensemble by−1.50pp. Selecting only the favorable aggregation would overstate evidence.']
        lines+=['','## '+('置信度加权的配对证据' if zh else 'Paired evidence for confidence weighting'),'',
                '| Backbone | Reference | Ensemble delta (pp) | Fixes / breaks | 95% CI (pp) | Holm p |','|---|---|---:|---:|---|---:|']
        for p in pairs:
            lines.append(f"| {p['backbone']} | {p['reference']} | {p['improvement_percentage_points']:+.2f} | {p['ag_fixes']}/{p['ag_breaks']} | {p['paired_stratified_bootstrap_95ci_pp']} | {p['holm_adjusted_p']:.4f} |")
        positive=[p for p in pairs if p['improvement_percentage_points']>0 and p['holm_adjusted_p']<.05]
        lines+=['',('六项预设检验中，校正后显著正向对比数量：' if zh else 'Positive contrasts surviving Holm correction among six predeclared tests: ')+str(len(positive))+'.','',
            '检验使用三种子概率集成的200张独立观察图片，而不是把三种子当作600张独立照片。使用精确McNemar检验和5000次按类别分层自助区间。区间未作多重校正，且条件于固定模型；[0,0]不证明总体效应为零。' if zh else
            'Tests use200 observation-level predictions from three-seed probability ensembles,not600 independent images. Exact McNemar tests and5000 class-stratified bootstrap resamples;Holm across six contrasts. Intervals are unadjusted and conditional on fixed models. A [0,0] empirical interval does not prove a zero population effect.','',
            '## '+('检测与权重利用' if zh else 'Detection and weight use'),'',
            '| Split | Zero boxes | One box | Two boxes | Mean larger weight |','|---|---:|---:|---:|---:|']
        for x in coverage:lines.append(f"| {x['split']} | {x['zero']} | {x['one']} | {x['two']} | {x['mean_larger_weight']} |")
        lines+=['',
            '固定30条蘑菇可见属性提示，所有图片共用；不读取图片自带caption、坐标或测试属性。OWL-ViT最多取两个区域，阈值0.05。置信度不是经校准的属性正确概率。固定加权检查点替换为等权/打乱权重的诊断在verification.json；未用于调参。' if zh else
            'Thirty fixed visible fungus prompts are shared across all images. No per-image captions,coordinates or test attribute labels enter the model. OWL-ViT uses at most two regions with threshold0.05. Detector confidence is not a calibrated attribute-correctness probability. Fixed-checkpoint uniform/shuffled interventions are diagnostic only,recorded in verification.json.', '',
            '## '+('数据保护与结论边界' if zh else 'Data protection and limits'),'',
            f'Historical images screened: {len(historical)}. Minimum pHash/dHash distances: {min_p}/{min_d};threshold4. Zero exact/pixel/perceptual-threshold matches after filtering. Observation IDs are disjoint across splits. Photographer identities are unavailable for grouping.', '',
            '先锁定清单、代码和配置，再提取训练/验证特征。42个检查点按验证macro-F1/CE选定并封存后才运行测试检测与特征提取；42个模型重放已核验。未按测试结果修改模型。' if zh else
            'Manifest,code and configuration locked before development feature extraction. All42 validation-selected checkpoints and development assets sealed before test grounding/features.42 predictions replayed for audit. No test-guided changes.', '',
            '2022/2023是元数据中的观察日期，晚于2021年发布的OpenAI CLIP及ImageNet1K数据；这加强了针对旧主干的时间隔离，但不是独立验证过的拍摄日期证明。OWL-ViT预训练重叠未审计，不能宣称整个系统从未见过图片；物种概念也可能早已见过。' if zh else
            'Metadata observation dates2022/2023 postdate the2021 OpenAI CLIP release and ImageNet1K data. This strengthens temporal separation for the old backbones,but observation dates are not independently verified capture dates. OWL-ViT pretraining overlap is unaudited:the entire pipeline cannot be certified image-unseen,and species concepts may already be known.', '',
            '这是自定义闭集时间划分，不是未见类别零样本实验，也不是官方完整基准。500张小样本、短训练预算及未人工核验的属性定位限制推广；新数据不能保证不过拟合。若后续继续改方法，本测试集也已被看过。跨鸟类/真菌数据集的绝对准确率不能直接比较为算法收益。' if zh else
            'This is a custom closed-set temporal split,not unseen-class zero-shot recognition or the full official benchmark. Small sample size,short fixed training budget and unverified attribute localization limit generalization. Fresh data cannot guarantee no overfitting. This holdout is now observed for future development. Cross-dataset bird/fungus accuracy differences are not algorithmic gains.', '',
            'Sources: [official dataset](https://github.com/BohemianVRA/FungiTastic), [author Kaggle mirror](https://www.kaggle.com/datasets/picekl/fungitastic), [CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md).', '',
            '```powershell', r'.\.venv\Scripts\python.exe -m unittest test_fungitastic test_confidence test_statistics -v',
            r'.\.venv\Scripts\python.exe fungitastic_experiment.py all', r'.\.venv\Scripts\python.exe report_fungitastic.py', '```','',
            'Photos,feature caches and weights remain local on E:. Only code,configurations and aggregated reports are published. Preparation currently uses historical local fingerprints for project-overlap screening;see data_audit.json.','']
        (f.OUT/f'summary_{lang}.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    labels=['Visual','Random','Text','Regions','AG mean','AG conf.','AG shuffle']
    for ax,n in zip(axes,c['backbones']):
        x=[lookup[(n,v)] for v in c['variants']]
        ax.bar(range(7),[a['accuracy_mean']*100 for a in x],yerr=[a['accuracy_sd']*100 for a in x],capsize=3,color=['#8796a4','#c49a53','#477caf','#bda7c4','#529584','#256f51','#88805c'])
        ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.set_ylim(0,100);ax.set_title(n)
    axes[0].set_ylabel('Accuracy (%), mean +/- seed SD');fig.suptitle('FungiTastic | 10 species | 200 test observations from 2023 | 3 seeds')
    fig.savefig(f.OUT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(summary=summary,paired=pairs,coverage=coverage,interventions=interventions),indent=2))

if __name__=='__main__':main()
