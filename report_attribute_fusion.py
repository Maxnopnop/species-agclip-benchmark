"""Audit pooled cross-fit predictions and render the bounded fusion experiment."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from attribute_fusion_experiment import ROOT, OUT, CONFIG, load_scores, standardized, stratified_folds, metrics, changes, save, digest


def main():
    torch.set_num_threads(4)
    c = json.loads(CONFIG.read_text())
    r = json.loads((OUT/'results.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert all(digest(ROOT/p) == sha for p, sha in protocol['source'].items())
    native, attributes, labels, rows, checks = load_scores(c)
    assert checks == r['verification']
    folds = stratified_folds(labels, c['folds'], c['fold_seed'])
    native_pred = native.argmax(-1)
    case_rows = []
    transition_counts = []
    for variant in c['variants']:
        az = standardized(attributes[variant])
        out = torch.full((3, 500), -1, dtype=torch.long)
        for selection in r['fold_selections'][variant]:
            mask = folds == selection['fold']
            out[:, mask] = (standardized(native[mask])[None] + selection['weight'] * az[:, mask]).argmax(-1)
        for j, seed in enumerate(c['seeds']):
            assert metrics(out[j], labels) == r['crossfit'][variant][j]['metrics']
            assert changes(out[j], native_pred, labels) == r['crossfit'][variant][j]['changes']
            if variant == 'learned_part':
                damaged = (native_pred == labels) & (out[j] != labels)
                transition_counts.append(dict(seed=seed, unseen_new_errors=int((damaged & (labels >= 50)).sum()),
                                              unseen_new_errors_predicted_seen=int((damaged & (labels >= 50) & (out[j] < 50)).sum())))
            for i, row in enumerate(rows):
                case_rows.append(dict(variant=variant, seed=seed, image_id=row['image_id'], role=row['role'],
                                      label=int(labels[i]), fold=int(folds[i]), native_prediction=int(native_pred[i]),
                                      fused_prediction=int(out[j, i])))
    with (OUT/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader();writer.writerows(case_rows)
    names = dict(global_='Global attributes', uniform='Uniform pooling', native_part='Original localization',
                 position_prior='Fixed position prior', learned_part='Repaired localization', wrong_part='Wrong part control')
    names['global'] = names.pop('global_')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout='constrained')
    order = c['variants']
    means = [r['summary'][v]['H']['mean'] for v in order]
    stds = [r['summary'][v]['H']['std'] for v in order]
    colors = ['#70859b' if v != 'learned_part' else '#16877a' for v in order]
    axes[0].barh(range(len(order)), means, xerr=stds, color=colors, capsize=3)
    axes[0].set_yticks(range(len(order)), [names[v] for v in order]);axes[0].invert_yaxis()
    axes[0].axvline(r['baseline']['H'], color='#a43a3a', linestyle='--', label='Native SigLIP 2')
    axes[0].set_xlim(73, 81);axes[0].set_xlabel('GZSL H (%)');axes[0].legend(fontsize=8)
    axes[0].set_title('Cross-fit fusion: mean +/- seed SD')
    for v in order:
        axes[1].plot(c['weights'], [g['summary']['H']['mean'] for g in r['descriptive_full_development_grid'][v]],
                     marker='o', label=names[v], linewidth=2 if v == 'learned_part' else 1)
    axes[1].set_xlabel('Attribute residual weight');axes[1].set_ylabel('GZSL H (%)')
    axes[1].set_title('Fixed weights on full development set\nDescriptive only');axes[1].legend(fontsize=7)
    x = np.arange(3)
    learned = r['crossfit']['learned_part']
    axes[2].bar(x-.17, [a['changes']['all']['corrected'] for a in learned], .34, label='Corrected', color='#16877a')
    axes[2].bar(x+.17, [a['changes']['all']['newly_wrong'] for a in learned], .34, label='New errors', color='#a43a3a')
    axes[2].set_xticks(x, c['seeds']);axes[2].set_xlabel('Attribute training seed');axes[2].set_ylabel('Images (out of 500)')
    axes[2].set_title('Repaired branch: errors fixed vs introduced');axes[2].legend(fontsize=8)
    fig.suptitle('Frozen SigLIP 2 + attribute score fusion | Reused development data; exploratory', fontsize=12)
    fig.savefig(OUT/'comparison.png', dpi=180);plt.close(fig)
    cn = {'global':'全图属性', 'uniform':'均匀区域池化', 'native_part':'原始定位属性', 'position_prior':'固定位置属性',
          'learned_part':'修复定位属性', 'wrong_part':'错误部位属性'}
    text = [
        '# 修复后的局部属性能否帮助 SigLIP 2 纠错？',
        '',
        '结论：能纠正部分错误，但当前证据不支持“定位修复带来特有、显著的融合优势”。修复分支融合的三种子平均 H 从 75.46 提升至 77.85（+2.38 个百分点）；原始定位、全图属性、固定位置和错误部位对照同样提升，且其平均 H 均高于修复分支。',
        '',
        '## 固定实验设计',
        '',
        '冻结 SigLIP 2 全图分支和此前训练完成的全部属性分支，不更新模型、不下载图片、不访问最终评估集。属性分支使用 FG-CLIP 2 特征和明确的“属性证据 × 类别属性原型”评分路径；它是跨主干分数融合，不是 AG-CLIP 原论文的完整复现。',
        '',
        '使用 500 张开发图片：50 个已见类别各 5 张，25 个开发未见类别各 10 张，推理时竞争类别为全部 75 类。每张图分别将两分支在这 75 类上的分数标准化，融合公式为 z(SigLIP2) + λ z(属性)。λ 固定为 0、0.05、0.1、0.2、0.5，不按结果追加搜索。λ 不是概率。',
        '',
        '每类分为五折，留出一折时仅在其余四折上按三种子平均 H 选权重；并列选较小权重。相同权重用于这一折的全部种子，最后汇总各图被留出时的预测。六个分支使用相同融合搜索预算。修复与错误部位分支还拥有相同的定位训练预算；其他分支没有这一步额外监督。',
        '',
        '## 主结果：五折交叉选择，三种子均值',
        '',
        '| 方法 | S 已见类 (%) | U 未见类 (%) | H (%) | 整体准确率 (%) |',
        '| --- | ---: | ---: | ---: | ---: |',
        f"| SigLIP 2 原始基线 | {r['baseline']['S']:.2f} | {r['baseline']['U']:.2f} | {r['baseline']['H']:.2f} | {r['baseline']['accuracy']:.2f} |"
    ]
    for v in order:
        m = r['summary'][v]
        text.append(f"| SigLIP 2 + {cn[v]} | {m['S']['mean']:.2f} | {m['U']['mean']:.2f} | {m['H']['mean']:.2f} ± {m['H']['std']:.2f} | {m['accuracy']['mean']:.2f} |")
    text += ['', 'S、U 是每类准确率的平均值；H 是两者的调和平均数，不是整体准确率。± 为三个固定训练种子的标准差，不是置信区间。所有行均报告，未按融合结果删掉较差对照。',
             '', '## 修复分支实际纠错与误改', '',
             '| 种子 | 纠正原错误 | 把原正确改错 | 净增加正确 | 已见类净增加 | 未见类净增加 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for cell in learned:
        a = cell['changes']
        text.append(f"| {cell['seed']} | {a['all']['corrected']} | {a['all']['newly_wrong']} | {a['all']['net_correct']} | {a['seen']['net_correct']} | {a['unseen']['net_correct']} |")
    text += ['', 'SigLIP 2 原有 122 个错误（已见类 53、未见类 69）。修复分支每个种子纠正 24–27 个错误，同时新增 10–11 个错误，净增 13–17 张正确预测。已见类受益，但未见类每个种子净减少 3–4 张正确预测。不能把纠正数量单独当作最终收益。',
             '', '## 如何解释', '',
             '- 定位修复有效与融合收益属于不同命题。此前修复提高了属性定位及属性单分支分类，但本次没有形成优于原始定位或错误部位对照的融合表现。',
             '- 全图、均匀池化和错误部位分支也能提升基线，提示额外监督、另一主干的互补信息和已见类别偏好可能贡献收益；本轮没有因果分解这些因素。',
             '- 错误部位对照是在错配部位后重新训练读出头，并非随机噪声。网络可能借助部位相关性或全图属性补偿，不能解读为“解剖位置完全不重要”。',
             '- 属性分支的 230 项属性包含 144 项部位属性和 86 项全图属性。本次是整个属性分支融合，并未隔离纯局部属性的净作用。',
             '- 修复分支五折权重为 0.5、0.2、0.5、0.2、0.2。固定权重 0.5 的全开发集 H 为 78.69，但它使用全部开发标签选择，不能替代主结果 77.85。即使使用相同固定权重，修复定位也没有一致领先所有对照。',
             '', '## 不确定性与边界', '']
    for target in ['siglip2', 'native_part', 'wrong_part']:
        interval = r['exploratory_intervals'][target]['paired_class_bootstrap_95_percentile']
        delta = r['summary']['learned_part']['H']['mean'] - (r['baseline']['H'] if target == 'siglip2' else r['summary'][target]['H']['mean'])
        name = 'SigLIP 2 基线' if target == 'siglip2' else cn[target]+'融合'
        text.append(f'- 修复分支融合相对{name}：H 差值 {delta:+.2f} 个百分点；条件性配对类别 bootstrap 95% 区间 [{interval[0]:+.2f}, {interval[1]:+.2f}]。')
    text += ['', '这些区间跨过零；同时区间条件于现有模型和已选权重，没有重新执行选权重，也没有包含重新训练或类别划分的不确定性。开发集在此前研究中反复使用，五折不能消除这一历史依赖。因此不能声称独立数据上的统计显著提升。未见类别仅指项目训练未见，基础模型预训练是否见过无法确认。类别属性原型来自 CUB 基准语义资产，其上游可能聚合测试标注，不等同于独立语言描述。',
             '', '## 下一步判断', '',
             '保留 SigLIP 2 作为默认全图基线，不把当前修复融合升级为“已验证更优”的默认模型。下一轮若继续，应先固定可靠属性集合，隔离局部与全图属性，并加入类别属性原型置乱对照以核对语义作用；重点检查未见类别误改，最后在未参与调参的新图片上验证。当前没有理由仅凭本轮结果继续更换更大的主干，也不应通过压低基线来制造提升。',
             '', '## 可复现性', '',
             '运行 `python -m unittest test_attribute_fusion -v`、`python attribute_fusion_experiment.py`、`python report_attribute_fusion.py`。协议先于融合计算写入并锁定配置/代码哈希。复核 18 个原分支检查点的来源、预测及 H；零权重保持原预测；报告重放全部 18 组交叉选择预测及纠错统计。',
             '', '文件：`protocol.json`（预先固定的设计）、`results.json`（完整结果）、`folds.json`（折划分）、`predictions.csv`（可审计预测）、`comparison.png`（图表）、`verification.json`（复核）。仓库不包含照片、模型权重或特征缓存。', '']
    (OUT/'summary_zh.md').write_text('\n'.join(text), encoding='utf-8')
    save(OUT/'verification.json', dict(protocol_source_matches=True, checkpoint_reconstructions=18,
                                      crossfit_predictions_replayed=18, unit_tests='3 passed before protocol lock',
                                      unseen_error_transitions=transition_counts,
                                      report_source_sha256=digest(Path(__file__))))
    print(json.dumps(dict(verified=True, unseen_error_transitions=transition_counts), indent=2))


if __name__ == '__main__':
    main()
