"""Report all315 cells and predeclared paired contrasts, without new tuning."""
import csv
import json
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json,digest
from report_expanded import paired_statistics,holm
from three_stage_experiment import CONFIG,OUT,RUN


def main():
    c=json.loads(CONFIG.read_text());r=json.loads((OUT/'results.json').read_text())
    selected=json.loads((OUT/'selection_locked.json').read_text())['historical_ag_choices']
    p=torch.load(RUN/'predictions.pt',weights_only=True)
    assert p['source']==r['source']
    for path,sha in r['source'].items():assert digest(ROOT/path)==sha
    groups=defaultdict(list)
    for row in r['rows']:groups[(row['backbone'],row['shots'],row['variant'])].append(row)
    summaries=[];lookup={}
    for (name,shots,variant),rows in groups.items():
        assert len(rows)==3
        record=dict(backbone=name,shots=shots,variant=variant,
                    accuracy_mean=float(np.mean([x['top1_accuracy'] for x in rows])),
                    accuracy_std=float(np.std([x['top1_accuracy'] for x in rows],ddof=1)),
                    macro_f1_mean=float(np.mean([x['macro_f1'] for x in rows])),
                    macro_f1_std=float(np.std([x['macro_f1'] for x in rows],ddof=1)),parameters=rows[0]['parameters'])
        if variant in c['new_variants']:record['training_accuracy_mean']=float(np.mean([x['training_accuracy'] for x in rows]))
        summaries.append(record);lookup[(name,shots,variant)]=record
    pairs=[];table=[]
    for item in selected:
        name,shots,ag=item['backbone'],item['shots'],item['selected_ag']
        predictions={};identity=None
        for variant in ['visual_linear','random_codes','baseline','region_only',ag]:
            byseed=[]
            for seed in c['seeds']:
                x=p['predictions'][f'{name}/{shots}/{seed}/{variant}']
                current=(x['paths'],x['labels'].tolist())
                if identity is not None:assert identity==current
                identity=current;byseed.append(x['probabilities'])
            predictions[variant]=torch.stack(byseed).mean(0).argmax(-1).numpy()
        for reference,target in c['primary_comparisons']:
            target=ag if target=='selected_ag' else target
            pairs.append(dict(backbone=name,shots=shots,reference=reference,target=target,
                              contrast=('ag_vs_text' if reference=='baseline' else 'ag_vs_region' if reference=='region_only' else
                                        'text_vs_visual' if reference=='visual_linear' else 'text_vs_random'),
                              **paired_statistics(predictions[reference],predictions[target],identity[1],seed=20261002)))
        get=lambda v:lookup[(name,shots,v)]
        table.append(dict(backbone=name,shots=shots,selected_ag=ag,
                          visual_accuracy=get('visual_linear')['accuracy_mean']*100,
                          random_accuracy=get('random_codes')['accuracy_mean']*100,
                          text_accuracy=get('baseline')['accuracy_mean']*100,
                          ag_accuracy=get(ag)['accuracy_mean']*100,
                          text_minus_visual_pp=(get('baseline')['accuracy_mean']-get('visual_linear')['accuracy_mean'])*100,
                          text_minus_random_pp=(get('baseline')['accuracy_mean']-get('random_codes')['accuracy_mean'])*100,
                          ag_minus_text_pp=(get(ag)['accuracy_mean']-get('baseline')['accuracy_mean'])*100))
    assert len(pairs)==60;holm(pairs)
    counts={key:sum(x['significant_positive_gain'] for x in pairs if x['contrast']==key)
            for key in ['text_vs_visual','text_vs_random','ag_vs_text','ag_vs_region']}
    write_json(OUT/'summary.json',dict(cells=summaries,three_stage_table=table,positive_holm_counts=counts,
                                     statistic_unit='Paired200-image predictions from three-seed probability ensembles;15 settings per contrast.'))
    write_json(OUT/'paired_comparisons.json',pairs)
    for filename,rows in [('all_results.csv',r['rows']),('three_stage_table.csv',table)]:
        with (OUT/filename).open('w',newline='',encoding='utf-8') as stream:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    names=dict(efficientnet_b0='EfficientNet-B0',resnet18='ResNet-18',convnext_tiny='ConvNeXt-Tiny',vit_b_16='ViT-B/16',clip_vit_b32='CLIP ViT-B/32*')
    cn=['# 五模型三阶段对照：纯视觉、图文适配、属性引导', '',
        '本轮补齐过去缺失的无文字视觉分类对照，并加入参数量匹配的随机类别向量对照。新增训练90组，逐一核验并复用原有225组，共评估315组。没有重新选择旧AG变体或修改旧结果。', '',
        '## 实验定义与公平性', '',
        '- 数据：原expanded20的20种物种、1000张图片；600张训练候选、200张验证、200张已在历史实验中评估过的测试图片。每次每类5/10/20张训练图，总计100/200/400张。三个种子42/43/44。闭集分类，不能解释为未见类别能力。',
        '- 纯视觉：冻结原来的图像特征，训练D→20普通线性分类头；不读取文字、属性或局部区域。前四个原版为视觉预训练模型。',
        '- 图文适配：原来的D→512投影（CLIP主干为残差投影）匹配固定CLIP类别文字向量；属于任务内监督适配，不是从头完成CLIP图文预训练。',
        '- AG：复用验证集预先选定的ag_mean/ag_attention/ag_aux，并保留region_only对照。仍是冻结主干、固定几何裁剪的轻量AG改编，不是原论文完整方法。',
        '- 随机类别向量：保持图文适配头的架构、初始化、温度及参数量，将类别文字换成固定随机单位向量；不包含文字语义。它用于检查图文适配相对普通线性头的收益是否来自语义以外的结构、优化或几何因素。',
        '- 每个新增模型与旧模型使用完全相同的训练图片、两个200步阶段、AdamW学习率0.001、余弦调度和验证macro-F1/CE选点规则；阶段间重置优化器与随机批次生成。所有90个新检查点先锁定，再计算测试指标。',
        '- 标准线性头与投影头的参数量和归一化不同；统一学习率/更新次数不代表各结构已获得最优超参数。随机类别向量对照提供了额外的参数匹配参照，但不能排除所有原型几何差异。',
        '- *第五个模型即使换成纯视觉分类头，仍保留CLIP图文预训练知识，不能称为“无CLIP预训练”。本轮只比较它的分类读出方式。', '',
        '## 三阶段结果', '',
        '下表为三个训练种子的平均测试准确率（%）。完整标准差和macro-F1见summary.json；差值按未四舍五入的数值计算。', '',
        '| 模型 | 每类训练图 | 纯视觉 | 随机向量对照 | 图文适配 | 属性引导 | 图文−纯视觉 | 图文−随机向量 | AG−图文 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in table:
        cn.append(f"| {names[x['backbone']]} | {x['shots']} | {x['visual_accuracy']:.2f} | {x['random_accuracy']:.2f} | {x['text_accuracy']:.2f} | {x['ag_accuracy']:.2f} | {x['text_minus_visual_pp']:+.2f} | {x['text_minus_random_pp']:+.2f} | {x['ag_minus_text_pp']:+.2f} |")
    cn += ['', '## 统计比较与解释', '',
           '配对统计以三个种子概率平均后的200张测试图预测为单位，不把同一图片的三个预测当作三个独立样本。四种比较×15种模型/样本数组合共60项，统一执行Holm校正。种子平均准确率差与概率集成后的差值可能不同。', '',
           '| 对比 | 通过Holm校正的正向提升 |', '| --- | ---: |',
           f"| 图文适配 − 普通视觉线性头 | {counts['text_vs_visual']}/15 |",
           f"| 图文适配 − 参数匹配随机向量对照 | {counts['text_vs_random']}/15 |",
           f"| AG − 图文适配 | {counts['ag_vs_text']}/15 |",
           f"| AG − 仅增加区域信息 | {counts['ag_vs_region']}/15 |", '',
           '本轮随机向量对照在全部15个设置中都高于普通视觉线性头，且图文适配相对随机向量没有校正后显著优势。因此，普通视觉头与图文适配的差距不能全部归因于文字语义；投影参数化、归一化、温度、优化以及原型几何都是尚未逐项分离的因素。随机向量也固定为一个种子，当前检验不覆盖随机原型重新抽样的不确定性。', '',
           'AG相对图文适配的种子平均变化为−1.67到+1.50个百分点。最大正向变化出现在CLIP ViT-B/32的20-shot设置（83.67%→85.17%），仍未通过校正后的配对检验；不能据此声称AG显著有效，也不能由当前轻量改编的负结果推断完整原论文方法无效。', '',
           '没有显著正向提升不等于证明效果为零；这些统计条件于当前模型、固定图片和历史研究选择。旧测试集已多次查看，不属于新的确认性验证。原始CLIP/视觉预训练是否与图片重叠也未被排除。', '',
           '## 训练充分性与资源范围', '',
           '以下训练准确率来自验证集选中的第二阶段检查点；不是用于选参的新指标。普通线性头在20-shot下未完全拟合训练集，但仅凭训练准确率不能判断原因是优化不足、容量限制还是验证集选点。相同短预算不保证同等收敛。本轮未为它追加训练，也未据此改变超参数；不应将其成绩称为最优纯视觉基线或完整微调模型。', '',
           '| 模型 | 20-shot普通视觉训练准确率 | 20-shot随机向量训练准确率 | 普通视觉参数量 | 随机／图文投影参数量 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name in c['backbones']:
        a=lookup[(name,20,'visual_linear')];b=lookup[(name,20,'random_codes')]
        cn.append(f"| {names[name]} | {a['training_accuracy_mean']*100:.2f}% | {b['training_accuracy_mean']*100:.2f}% | {a['parameters']:,} | {b['parameters']:,} |")
    cn += ['', '全部主干冻结，复用本地特征，没有下载新模型或图片。新增训练90组×400步；原有225组只重放，不重复训练。所有负结果与全部AG变体均保留在all_results.csv。', '',
           '## 复现', '',
           '依赖本机已有expanded20数据、特征缓存、文字资产和历史检查点；仓库不包含图片/权重。', '',
           '```powershell', '.\\.venv\\Scripts\\python.exe -m unittest test_three_stage -v',
           '.\\.venv\\Scripts\\python.exe three_stage_experiment.py all',
           '.\\.venv\\Scripts\\python.exe report_three_stage.py', '```', '',
           '协议及源文件哈希见protocol.json，选定检查点哈希见selection_locked.json，旧结果逐项重放核验见verification.json。该比较回答当前冻结特征与固定预算下的差异，不回答全模型微调、未见物种识别或原论文AG-CLIP的最终有效性。', '']
    (OUT/'summary_zh.md').write_text('\n'.join(cn),encoding='utf-8')
    en=['# Five-backbone visual / text-adapted / attribute-guided comparison', '',
        '90 newly trained controls plus 225 individually replayed historical cells. 20 species, 1,000 images; 5/10/20 training images per class; 3 seeds. Closed-set, frozen-feature, exploratory follow-up on previously evaluated test data.', '',
        'Visual linear heads use no text or attributes. Random-code heads have the exact text-adaptation projection/temperature architecture but random fixed class prototypes. Existing text-adapted and AG heads retain their original validation-selected checkpoints. All controls receive the same 200+200-step training and checkpoint schedule. CLIP ViT-B/32 retains CLIP pretraining under every readout.', '',
        '| Backbone | Shots | Visual linear (%) | Random codes (%) | Text-adapted (%) | Selected AG (%) |', '|---|---:|---:|---:|---:|---:|']
    for x in table:en.append(f"| {names[x['backbone']]} | {x['shots']} | {x['visual_accuracy']:.2f} | {x['random_accuracy']:.2f} | {x['text_accuracy']:.2f} | {x['ag_accuracy']:.2f} |")
    en += ['',f'Positive paired ensemble gains surviving Holm correction over all 60 contrasts: text vs linear {counts["text_vs_visual"]}/15; text vs random {counts["text_vs_random"]}/15; AG vs text {counts["ag_vs_text"]}/15; AG vs region-only {counts["ag_vs_region"]}/15.', '',
           'Random prototypes also outperform the ordinary linear readout in all 15 settings. This prevents attribution of the full text-versus-linear gap to language semantics: parameterization, normalization, temperature, optimization and prototype geometry differ. The random bank uses one fixed seed, so uncertainty from resampling prototypes is not covered. The largest mean AG gain is 1.50 percentage points for CLIP ViT-B/32 at 20 shots (83.67% to 85.17%); it is not significant after correction.', '',
           'Linear heads have not fully fit the training examples at 20 shots; this alone does not distinguish limited optimization, limited capacity or validation checkpoint selection. No post-result tuning was performed. These are not best-tuned end-to-end visual baselines.', '',
           'Intervals and tests condition on fixed trained models and a historically reused test set. They do not establish independent generalization, zero-shot capability, or an exact AG-CLIP replication. See the Chinese report, all_results.csv and paired_comparisons.json for full details.','']
    (OUT/'summary_en.md').write_text('\n'.join(en),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True,layout='constrained')
    for ax,shots in zip(axes,c['shots']):
        x=np.arange(5)
        for offset,variant,label,color in [(-.27,'visual_linear','Visual linear','#8495a6'),(-.09,'random_codes','Random codes','#c69c53'),(.09,'baseline','Text-adapted','#4679b2'),(.27,'selected_ag','Selected AG','#278b76')]:
            values=[];errors=[]
            for name in c['backbones']:
                ag=next(i['selected_ag'] for i in selected if i['backbone']==name and i['shots']==shots)
                a=lookup[(name,shots,ag if variant=='selected_ag' else variant)]
                values.append(a['accuracy_mean']*100);errors.append(a['accuracy_std']*100)
            ax.bar(x+offset,values,.18,yerr=errors,capsize=2,label=label,color=color)
        ax.set_xticks(x,['EffNet','ResNet','ConvNeXt','ViT','CLIP*']);ax.set_ylim(0,100);ax.set_title(f'{shots} training images / class')
    axes[0].set_ylabel('Accuracy (%), mean +/- seed SD');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
    fig.suptitle('20-species closed-set follow-up | *CLIP pretraining retained | Reused test set')
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(table=table,positive_holm_counts=counts),indent=2))


if __name__=='__main__':main()
