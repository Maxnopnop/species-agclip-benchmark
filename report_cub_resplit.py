"""Keep partition variation distinct from fusion seed variation."""
import json
import numpy as np
from data_tools import ROOT,write_json
from report_cub_followup import paired_summary


def main():
    c=json.loads((ROOT/'configs/cub_resplit_v1.json').read_text());out=ROOT/'reports/cub_resplit_v1';out.mkdir(parents=True,exist_ok=True);rows=[];contrasts=[]
    for backbone in c['backbones']:
        for split in c['class_partition_seeds']:
            version=f'cub_resplit_{backbone}_{split}';d=json.loads((ROOT/f'reports/{version}/final_results.json').read_text());protocol=json.loads((ROOT/f'reports/{version}/protocol.json').read_text());row=dict(backbone=backbone,split_seed=split,attributes=len(protocol['attributes']),native_H=d['native']['H'],native_ZSL=d['native']['ZSL'])
            for r in d['results']:row[r['variant']+'_H']=r['metrics']['H'];row[r['variant']+'_ZSL']=r['metrics']['ZSL'];row[r['variant']+'_step']=r['step']
            rows.append(row)
            for control in ['region_only','pooled','permuted','constant']:
                q=paired_summary(version,d['results'],control,protocol['seen']);q['scope']='Paired within-class image bootstrap, conditional on one trained fusion seed and this class partition. Overlapping partitions are not independent datasets. No aggregate significance claim.';contrasts.append(q)
    table=['| 主干 | 类别划分种子 | 属性数 | 原始 H | 无文字 H | token H | 平均融合 H | 错误关联 H | 常量 H |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:table.append(f'| {r["backbone"]} | {r["split_seed"]} | {r["attributes"]} | {r["native_H"]:.2f} | {r["region_only_H"]:.2f} | {r["tokens_H"]:.2f} | {r["pooled_H"]:.2f} | {r["permuted_H"]:.2f} | {r["constant_H"]:.2f} |')
    means=[]
    for b in c['backbones']:
        rr=[r for r in rows if r['backbone']==b];means.append(dict(backbone=b,mean_token_vs_region_H=float(np.mean([r['tokens_H']-r['region_only_H'] for r in rr])),mean_token_vs_constant_H=float(np.mean([r['tokens_H']-r['constant_H'] for r in rr])),token_step_zero_count=sum(r['tokens_step']==0 for r in rr)))
    write_json(out/'summary.json',dict(rows=rows,descriptive_means=means,conditional_contrasts=contrasts))
    text='''# 更换训练／未见物种划分后的复核

保持同一 100 个物种，预先固定三种类别随机划分。每次 50 类训练、25 类开发未见、25 类最终未见；重新从该次训练图片确定属性词表、训练探针和融合模块。一个类别在某次运行中为最终未见，就不会参与该次训练或参数选择。所有初始化重新开始，不跨划分传递已训练模型。

每次 500 张训练、500 张开发、750 张最终图片，遵守官方图片 train/test 边界。为支持类别角色轮换，总图片目录增加到 2500 张，包含此前 100 类实验之外的 750 张图片；每次实际仍用 1750 张。固定视觉编码可提前计算，但所有六个主干／划分组合的选择先锁定，再统一计算最终分类成绩。

CLIP B/16 和修正小写后的 SigLIP 2 各跑三种划分，每次 5 对照 × 2 学习率、固定融合种子 42，共 60 组融合训练。这三个“类别划分种子”不同于先前三个“融合初始化种子”。相同 100 物种和不少照片跨运行重叠，不能将其当作三个独立数据集或声称新的确认性显著结果。

H 为已见／未见类别宏平均准确率的调和平均，单位百分比。ZSL 和每次条件配对区间保存在 summary.json；跨划分均值仅作描述，不作独立样本显著性检验。

'''+ '\n'.join(table)+'\n\n'+ '\n'.join(f'{r["backbone"]}：token 相对无文字的跨划分平均 ΔH={r["mean_token_vs_region_H"]:+.3f} pp，相对常量为 {r["mean_token_vs_constant_H"]:+.3f} pp；{r["token_step_zero_count"]}/3 个划分选择第 0 步（未得到适配收益）。' for r in means)+'''

主协议继承的 selection 文本保留了旧实验“30 cells”的描述；本阶段每个主干／划分实际是 10 cells、5 个锁定选择，总数 60，以配置 variants/seeds/learning_rates 及实际结果文件为准。该文字不参与选择。
'''
    (out/'summary_zh.md').write_text(text,encoding='utf-8');print(json.dumps(means))


if __name__=='__main__':main()
