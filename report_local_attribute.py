"""Transparent tables and paired exploratory intervals; no further selection."""
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_attribute_experiment import ROOT, RUN, OUT, CONFIG, save, metrics

CN={'repaired_local':'修复定位／144 局部属性','native_local':'原始定位／相同 144 属性',
    'wrong_local':'错误部位／相同 144 属性','uniform_local':'均匀池化／相同 144 属性',
    'whole_local':'全图特征／相同 144 属性','whole_global':'全图特征／另 86 全图属性',
    'native_calibration':'仅 SigLIP 2 偏置校准'}


def main():
    c=json.loads(CONFIG.read_text());r=json.loads((OUT/'results.json').read_text())
    training=json.loads((OUT/'training.json').read_text())
    e=torch.load(RUN/'evaluation.pt',weights_only=True)
    arrays={key:np.array([[x['metrics']['per_class'][str(i)] for i in range(75)] for x in cells]) for key,cells in r['crossfit'].items()}
    arrays['baseline']=np.array([[r['baseline']['per_class'][str(i)] for i in range(75)]])
    for v in c['variants']:
        arrays[v+'/standalone']=np.array([[x['metrics']['per_class'][str(i)] for i in range(75)] for x in training['records'] if x['variant']==v])
    rng=np.random.default_rng(c['bootstrap_seed'])
    si=rng.integers(0,50,(c['bootstrap_replicates'],50));ui=rng.integers(50,75,(c['bootstrap_replicates'],25))
    def sampled(a):
        s,u=a[:,si].mean(-1),a[:,ui].mean(-1)
        return (2*s*u/np.maximum(s+u,1e-12)).mean(0)
    def observed(a):
        s,u=a[:,:50].mean(-1),a[:,50:].mean(-1)
        return float((2*s*u/np.maximum(s+u,1e-12)).mean())
    comparisons={}
    pairs=[('repaired_local/guarded',v) for v in arrays if v!='repaired_local/guarded' and not v.endswith('/standalone')]
    pairs += [('repaired_local/standalone',v+'/standalone') for v in c['variants'] if v!='repaired_local']
    for left,right in pairs:
        diff=sampled(arrays[left])-sampled(arrays[right])
        comparisons[left+' vs '+right]=dict(delta_H=observed(arrays[left])-observed(arrays[right]),
                                            conditional_class_bootstrap_95=np.quantile(diff,[.025,.975]).tolist())
    save(OUT/'analysis.json',dict(comparisons=comparisons,scope='Conditional paired class bootstrap; reused development data; selected policies fixed; no multiplicity correction or retraining/selection uncertainty. Exploratory, not confirmatory.'))
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['method','seed','image_id','fold','true_label','native_prediction','prediction'])
        for key,preds in e['predictions'].items():
            for j,seed in enumerate(c['seeds']):
                for i,image_id in enumerate(e['row_ids']):
                    writer.writerow([key,seed,image_id,int(e['folds'][i]),int(e['labels'][i]),int(e['native'][i].argmax()),int(preds[j,i])])
    standalone={v:{k:float(np.mean([x['metrics'][k] for x in training['records'] if x['variant']==v])) for k in ['S','U','H']} for v in c['variants']}
    text=['# 纯局部属性隔离与未见类别误改约束', '',
          '本轮观察到积极结果：在排除显式全图属性分支后，144 项局部属性配合受约束融合，使 SigLIP 2 的未见类别准确率 U 从 72.40% 提升至 77.73%，综合 H 从 75.46 提升至 79.49。每个种子仅新增 1–2 个未见类错误，纠正 14–15 个原有未见类错误。但定位修复相对原始定位的融合优势只有 0.15 个 H 百分点，仍不足以证明修复本身有可靠的独立增益。', '',
          '## 如何隔离局部属性', '',
          '所有 18 个新读出层从头按固定预算训练：3 个种子、每个 400 次更新、相同 500 张已见类别训练图。局部分支仅保留头、翼、胸、尾对应的 144 个属性，训练标签、损失、类别属性原型均只含这 144 列；没有全图属性损失、全图特征跳接或直接类别分类头。', '',
          '对同一 144 属性分别使用修复定位、原始定位、错配部位、均匀池化和全图特征。另训一个使用剩余 86 项全图属性的补充分支；它的属性数不同，不能作为严格同容量对照。全图特征／相同 144 属性才是控制属性语义集合的全图对照。错配部位后重新训练的模型可能补偿错配，因此它不是随机噪声。', '',
          '这里“纯局部”指显式输入路径和属性列的隔离。FG-CLIP 2 的 Transformer patch 特征本身仍有全图上下文，不能声称模型只看到了局部像素。推理时独立的 SigLIP 2 全图分支仍保留。', '',
          '| 属性独立分类器（不融合 SigLIP 2） | S (%) | U (%) | H (%) |', '| --- | ---: | ---: | ---: |']
    for v,m in standalone.items():text.append(f"| {CN[v]} | {m['S']:.2f} | {m['U']:.2f} | {m['H']:.2f} |")
    text += ['', '## 减少误改的固定方法', '',
             '融合分数为 z(SigLIP2) + λ·z(局部属性) − γ·已见类别指示项；z 按每张图的 75 个候选类分数标准化。对原模型第一、第二名分差大于 0.5 的图，保留原预测。这里分差不是概率或经过校准的置信度。', '',
             '预先固定 λ∈{0,0.05,0.1,0.2,0.5}，γ∈{0,0.25,0.5,1}。分层五折选参：每次仅在其他四折上选择平均 H 最高、且平均 U/H 不低于原模型、平均新增未见类错误不超过拟合未见图片数 1% 的组合。零权重、零校准是回退选项。一个权重组合用于所有三个种子；留出折从不参与选参。拟合折的约束不保证在留出折或新数据上成立。', '',
             '普通融合仅搜索 λ；保护融合搜索 λ、γ 并使用上述约束，预算不同。因此两者对比检验整个改进方案，不能单独归因于门控、校准或约束中的某一项。各属性对照在相同策略下使用相同预算。额外的原生 SigLIP 2 校准对照只搜索 γ；三个语义列置乱固定且分别报告，没有选择最好的一次。', '',
             '修复局部分支五个拟合折均选择 λ=0.2、γ=0.25。全开发集选出的未来推理配置也保存在 inference_policy.json；该配置不是独立验证的生产模型。', '',
             '## 融合结果：三种子均值，汇总每张图被留出时的预测', '',
             '| 方法 | 策略 | S (%) | U (%) | H (%) | 整体准确率 (%) |', '| --- | --- | ---: | ---: | ---: | ---: |',
             '| SigLIP 2 | 原始 | 78.80 | 72.40 | 75.46 | 75.60 |']
    for key,m in r['summary'].items():
        v,p=key.split('/');name=CN.get(v,v.replace('semantic_shuffle_','局部属性语义列置乱 '))
        text.append(f"| {name} | {'误改保护' if p=='guarded' else '普通融合'} | {m['S']['mean']:.2f} | {m['U']['mean']:.2f} | {m['H']['mean']:.2f} ± {m['H']['std']:.2f} | {m['accuracy']['mean']:.2f} |")
    text += ['', 'H 是 S/U 的调和平均数；± 是三种子标准差，不是置信区间。部分置乱分支在留出数据上仍造成退化，表明选参约束不是泛化保证。', '',
             '## 未见类别：究竟纠正了多少，误改了多少', '',
             '| 种子 | 普通局部：纠正 / 新错 / 净增 | 保护局部：纠正 / 新错 / 净增 | 保护局部：已见类别纠正 / 新错 |',
             '| --- | --- | --- | --- |']
    for j,seed in enumerate(c['seeds']):
        a=r['crossfit']['repaired_local/plain'][j]['changes']['unseen'];b=r['crossfit']['repaired_local/guarded'][j]['changes']
        text.append(f"| {seed} | {a['corrected']} / {a['newly_wrong']} / {a['net_correct']} | {b['unseen']['corrected']} / {b['unseen']['newly_wrong']} / {b['unseen']['net_correct']} | {b['seen']['corrected']} / {b['seen']['newly_wrong']} |")
    text += ['', '未见类别共 250 张。上一轮“230 属性混合分支”每个种子新增 7–8 个未见错误；本轮局部普通融合为 4–5 个，局部保护融合为 1–2 个。上一轮到本轮的变化包含重新训练和属性子集变化，不能当作单因素因果对照。本轮纯局部普通→保护是同一读出层上的策略比较。', '',
             '保护策略存在取舍：已见类别 S 从普通局部融合的 84.67% 降至 81.33%，仍高于原模型的 78.80%；已见类别新增错误变多。总体准确率从基线 75.60% 提升至 79.53%，但不能声称所有类型的错误都减少。', '',
             '## 能否归因于局部属性／定位修复', '',
             '局部证据不再只是全图属性分支的替代说法：全部显式全图属性路径已移除；语义列置乱后效果约为基线水平；仅校准原模型的策略选择了不改动预测。这些结果支持语义一致的局部属性与受约束融合共同起作用。', '',
             '但修复、原始定位和错配部位的保护融合 H 分别为 79.49、79.35、79.00，差距很小。全图特征／相同 144 属性保护融合 H 为 77.43，提示局部池化可能有用；仍需控制额外定位监督、上下文和超参数预算，并在新数据上复核。单独定位修复的融合收益不能据此确认。', '',
             '剩余 86 项全图属性的普通融合 H 为 79.76，是另一种竞争方法，不应遗漏；但其 U 为 72.93%，没有达到本轮局部保护融合的未见类别表现。目标是可审计的错误取舍，不能只选一个指标最高的方法来证明假设。', '',
             '## 条件性不确定性', '', '| H 差值比较 | 差值（百分点） | 条件性 95% 区间 |', '| --- | ---: | --- |']
    for target in ['baseline','repaired_local/plain','native_local/guarded','wrong_local/guarded','whole_local/guarded','native_calibration/guarded']:
        m=comparisons['repaired_local/guarded vs '+target];lo,hi=m['conditional_class_bootstrap_95']
        text.append(f"| 修复局部保护 − {target} | {m['delta_H']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    text += ['', '这些配对类别 bootstrap 区间条件于已经训练的模型与已选权重，没有重新训练或重复选参，也没有多重比较校正。500 张开发图已经多次使用，五折不能消除历史研究选择带来的依赖。区间即使不含零，也只能作为探索性证据，不能宣称在独立数据上显著提升。', '',
             '## 已实现与复现', '',
             '本机代码：local_attribute_experiment.py（训练/评估），test_local_attribute.py（全图路径隔离、门控和选择约束），predict_local_attribute.py（图像输入推理），verify_local_attribute.py（重放及实际图片核对），report_local_attribute.py（报告）。', '',
             '推理命令：`.venv\\Scripts\\python.exe predict_local_attribute.py --image E:\\path\\bird.jpg --seed 42`。它使用已保存的研究配置及 75 个固定候选物种，输入不含真实类别、可见属性或部位坐标。不能把此配置直接解释为支持任意未知物种。', '',
             '一张有意选取的未见类别纠错图片仅用于检查实际图像推理与缓存预测一致，不是新增准确率证据。具体差值与所有重放检查见 verification.json。模型/缓存留在本机，仓库只保存代码、配置和结果。', '',
             '下一步应冻结此候选方案，用没有参与当前诊断的新图片评估，同时保留原始定位和同属性全图对照；当前不继续扩大超参数搜索或更换主干。', '']
    (OUT/'summary_zh.md').write_text('\n'.join(text),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    chosen=['repaired_local/plain','repaired_local/guarded','native_local/guarded','wrong_local/guarded','whole_local/guarded','native_calibration/guarded']
    short=['Local / plain','Local / guarded','Original / guarded','Wrong part / guarded','Whole image / guarded','Native calibration']
    axes[0].errorbar([r['summary'][k]['H']['mean'] for k in chosen],range(6),
                     xerr=[r['summary'][k]['H']['std'] for k in chosen],fmt='o',capsize=3,color='#247d77')
    axes[0].set_yticks(range(6),short);axes[0].invert_yaxis();axes[0].axvline(r['baseline']['H'],ls='--',c='#b54e49')
    axes[0].set_xlabel('H (%)');axes[0].set_title('Cross-fit fusion: mean +/- seed SD')
    for key,color in [('repaired_local/plain','#859bad'),('repaired_local/guarded','#247d77')]:
        m=r['summary'][key]
        axes[1].scatter(m['S']['mean'],m['U']['mean'],label=key.split('/')[1],s=80,c=color)
    axes[1].scatter(r['baseline']['S'],r['baseline']['U'],label='Native SigLIP 2',c='#b54e49',s=80)
    axes[1].set_xlabel('Seen accuracy S (%)');axes[1].set_ylabel('Unseen accuracy U (%)');axes[1].set_title('Seen / unseen tradeoff');axes[1].legend()
    x=np.arange(3)
    for shift,key,label,color in [(-.18,'repaired_local/plain','Plain','#859bad'),(.18,'repaired_local/guarded','Guarded','#247d77')]:
        damage=[a['changes']['unseen']['newly_wrong'] for a in r['crossfit'][key]]
        axes[2].bar(x+shift,damage,.36,label=label,color=color)
    axes[2].set_xticks(x,c['seeds']);axes[2].set_xlabel('Training seed');axes[2].set_ylabel('New unseen errors (250 images)');axes[2].legend()
    axes[2].set_title('Wrong corrections on unseen images')
    fig.suptitle('144 local attributes, no explicit global attribute path | Reused development data; exploratory')
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in comparisons.items() if k.endswith('baseline') or k.endswith('native_local/guarded') or k.endswith('wrong_local/guarded')},indent=2))


if __name__=='__main__':main()
