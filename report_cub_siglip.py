"""Report the corrected SigLIP experiment, excluding the defective v1."""
import json
from data_tools import ROOT,write_json
from report_cub_followup import paired_summary,mean_sd


def main():
    version='cub_siglip_v2';out=ROOT/'reports'/version;result=json.loads((out/'final_results.json').read_text());p=json.loads((ROOT/'configs/cub_siglip_v2.json').read_text());rows=[]
    for v in ['region_only','pooled','tokens','permuted','constant']:
        entries=[r for r in result['results'] if r['variant']==v];row=dict(variant=v)
        for k in ['H','ZSL','S','U']:row[k+'_mean'],row[k+'_sd']=mean_sd([e['metrics'][k] for e in entries])
        rows.append(row)
    contrasts=[paired_summary(version,result['results'],c,p['seen_classes']) for c in ['region_only','pooled','permuted','constant']]
    write_json(out/'summary.json',dict(native=result['native'],rows=rows,paired_contrasts=contrasts))
    table=['| 模型 | H 均值 ± SD | ZSL 均值 ± SD |','|---|---:|---:|',f'| 原始 SigLIP 2 B/16 | {result["native"]["H"]:.2f} | {result["native"]["ZSL"]:.2f} |']
    for r in rows:table.append(f'| {r["variant"]} | {r["H_mean"]:.2f} ± {r["H_sd"]:.2f} | {r["ZSL_mean"]:.2f} ± {r["ZSL_sd"]:.2f} |')
    ci=['| tokens 对照 | ΔH | 条件 95% 区间 |','|---|---:|---:|']
    for r in contrasts:ci.append(f'| tokens − {r["control"]} | {r["mean_H_delta"]:+.2f} | [{r["conditional_95_interval"][0]:+.2f}, {r["conditional_95_interval"][1]:+.2f}] |')
    text='''# SigLIP 2：修正文字预处理后的 100 物种复核

使用与 cub100_v1 完全相同的类别与图片划分，500 张训练、500 张开发、750 张最终评估，230 条属性。模型换为本地 SigLIP 2 Base/16，采用其原生 224×224 缩放和归一化，重新计算裁剪内可见性。维度为 768。全图和区域编码器冻结；五种融合对照有相同参数量，各跑两个学习率、三个种子和 400 步。推理只输入图片和固定本地模型／文字资产。

**文字大小写缺陷已单独处理。** v1 保留了大写物种名，而快速 Gemma 分词器没有落实配置中的 do_lower_case。开发集审计中，仅显式转为小写，原始 H 从 9.70% 到 75.46%。v1 的 30 组结果全部排除科学比较；不能把这个修复算成 AG 收益。v2 是重新编码文字并从头训练探针／融合模块的修正版。官方 SiglipProcessor/模型前向与手动特征相似度的 logits 最大误差约 1e-6。审计证据保存在 cub_siglip_v1/inference_audit.json。

所有 v2 选择先在开发集锁定再评估，但此前 CLIP 与有缺陷的 v1 最终数据已被查看，因此属于探索性复测。固定分数温度 20 对原始类别排序无影响，也用于所有融合对照；输出不是校准概率。SigLIP 与 CLIP 的对比同时改变预训练、图像预处理和维度，不能解释为只改变某一个结构因素。

H 为 50 个已见类与 25 个最终未见类宏平均准确率的调和平均；ZSL 只在 25 个最终未见类别间分类。数值为百分比，差值为百分点。

'''+ '\n'.join(table)+'\n\n'+ '\n'.join(ci)+'''

区间是固定三个训练种子上的、按类别配对重采样图片的条件 bootstrap 区间，未覆盖类别划分随机性或多重模型搜索。正确属性必须优于匹配对照，才能讨论属性特有收益；单纯换主干或修复大小写带来的提升不属于 AG。
'''
    (out/'summary_zh.md').write_text(text,encoding='utf-8');print(json.dumps(contrasts))


if __name__=='__main__':main()
