"""Separate report for the predeclared 100-new-species follow-up."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json
from report_cub_followup import paired_summary,mean_sd


def main():
    out=ROOT/'reports/cub100_v1';result=json.loads((out/'final_results.json').read_text());p=json.loads((ROOT/'configs/cub100_v1.json').read_text())
    variants=['region_only','pooled','tokens','permuted','constant'];rows=[]
    for variant in variants:
        entries=[r for r in result['results'] if r['variant']==variant];row=dict(variant=variant,seeds=[r['seed'] for r in entries])
        for key in ['S','U','H','ZSL']:row[key+'_mean'],row[key+'_sd']=mean_sd([r['metrics'][key] for r in entries])
        rows.append(row)
    contrasts=[paired_summary('cub100_v1',result['results'],c,p['seen_classes']) for c in ['region_only','pooled','permuted','constant']]
    write_json(out/'summary.json',dict(native=result['native'],rows=rows,paired_contrasts=contrasts))
    table=['| 模型 | H 均值 ± SD | ZSL 均值 ± SD |','|---|---:|---:|',f'| 原始 CLIP B/16 | {result["native"]["H"]:.2f} | {result["native"]["ZSL"]:.2f} |']
    for r in rows:table.append(f'| {r["variant"]} | {r["H_mean"]:.2f} ± {r["H_sd"]:.2f} | {r["ZSL_mean"]:.2f} ± {r["ZSL_sd"]:.2f} |')
    ci=['| 独立属性 token 对照差值 | ΔH | 条件 95% 区间 |','|---|---:|---:|']
    for r in contrasts:ci.append(f'| tokens − {r["control"]} | {r["mean_H_delta"]:+.2f} | [{r["conditional_95_interval"][0]:+.2f}, {r["conditional_95_interval"][1]:+.2f}] |')
    text='''# 100 个新物种：扩大候选空间后的复核

该实验在查看结果前固定配置并提交协议，抽取此前 20 物种以外的 100 个 CUB 物种。50 类用于训练，25 类仅用于开发，25 类仅用于最终评估。500 张训练、500 张开发、750 张最终评估，共 1750 张不同图片；与旧实验的类别和图片哈希均不重叠。每个训练类别仅 10 张图片。预训练模型是否曾见过 CUB 图片无法完全核实。

CLIP ViT-B/16 冻结，使用 230 条训练集支持的属性，5 种融合对照 × 2 个学习率 × 3 个随机种子，共 30 组训练。学习率和检查点先在开发集选择并锁定，再计算最终特征与评估。三种子只覆盖融合训练的随机性，属性探针和类别划分固定。

**整体适配有小幅收益，但没有证明独立属性的额外收益。** 无文字、平均融合、错误关联和常量属性对照的结果与正确属性相近。不能把相对于原始主干的全部提升归因于 AG。

H 是 50 个已见与 25 个最终未见类别的宏平均准确率的调和平均，候选类别共 75 个。ZSL 只在 25 个最终未见类别间预测。下列数值都是百分比，差值是百分点。

'''+ '\n'.join(table)+'\n\n'+ '\n'.join(ci)+'''

区间来自按类别配对重采样图片的 1000 次 bootstrap，同次重采样用于三个已训练种子。区间条件于固定类别划分、属性探针、选择网格与三次训练，未覆盖类别抽样、全部训练随机性或多重比较，不能当作确认性显著性检验。

事后校准单独保存在 calibration.json，各对照都获得相同开发集网格；不替代以上预先锁定的主要成绩。单张图片部署复核见 deployment_verification.json，重新检测及编码的 logits 与缓存最大误差约 1.5e-6，预测一致。
'''
    (out/'summary_zh.md').write_text(text,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric in zip(axes,['H','ZSL']):
        ax.errorbar(range(5),[r[metric+'_mean'] for r in rows],yerr=[r[metric+'_sd'] for r in rows],fmt='o',capsize=4,color='#2563eb')
        ax.axhline(result['native'][metric],color='black',ls='--',label='Native B/16');ax.set_xticks(range(5),variants,rotation=25,ha='right');ax.set_ylabel(metric+' (%)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('100 new CUB species · three fusion seeds · mean ± SD');fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(contrasts))


if __name__=='__main__':main()
