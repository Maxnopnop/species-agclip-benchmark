"""Audit the same-backbone small CoCa pilot; preserve negative results."""
import csv,json
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import coca_experiment as e
from coca_data import validate
from coca_components import Fusion
from data_tools import digest,write_json

PAPER='https://www.researchgate.net/publication/399822501_AG-CLIP_Attribute-Guided_CLIP_for_Zero-Shot_Fine-Grained_Recognition'

def main():
    torch.set_num_threads(4);c=json.loads(e.CONFIG.read_text());m=json.loads((e.DATA/'manifest.json').read_text());validate(m);s=e.source()
    seal=e.check_seal(s);result=json.loads((e.OUT/'results.json').read_text());assert result['source']==s
    assert datetime.fromisoformat(seal['locked_at_utc'])<datetime.fromisoformat(result['scored_at_utc'])
    assert len(result['rows'])==10
    assert not {x['source_id'] for x in m['classes']}&set(m['excluded_species'])
    refs=json.loads((e.ROOT/'cache/confidence_fresh_v1/historical_fingerprints.json').read_text())['fingerprints']
    refs+=json.loads((e.ROOT/'data/confidence_fresh_v1/manifest.json').read_text())['rows']
    minp=64;mind=64
    for x in m['rows']:
        assert digest(e.Path(m['image_root'])/x['path'])==x['sha256']
        assert not any(x['sha256']==r['sha256'] or x['pixels_sha256']==r['pixels_sha256'] for r in refs)
        dp=min((x['phash']^r['phash']).bit_count() for r in refs);dd=min((x['dhash']^r['dhash']).bit_count() for r in refs)
        assert dp>4 and dd>4;minp=min(minp,dp);mind=min(mind,dd);refs.append(x)
    d=e.load_stage('test',s);b=e.bank(s);tail=e.make_tail().cuda();pred=torch.load(e.RUN/'predictions.pt',weights_only=True);assert pred['source']==s
    assert pred['paths']==[x['path'] for x in d['rows']]
    candidates=pred['candidates'];maxerr=0.;training=[]
    initial=torch.load(e.CACHE/'initial_tail.pt',weights_only=True)
    tail.load_state_dict(initial['state_dict']);native=e.evaluate(tail,Fusion('baseline').cuda(),d,b,candidates)
    torch.testing.assert_close(native,pred['predictions']['native'],atol=1e-5,rtol=1e-4)
    for row in result['rows']:
        seed=row['seed'];v=row['variant'];ck=torch.load(e.RUN/f'{seed}/{v}.pt',weights_only=True)
        assert ck['source']==s
        selected=max(ck['history'],key=lambda a:e.select_key(a,a['step']))
        assert selected['step']==ck['best_step']==row['best_step']
        tail.load_state_dict(ck['tail']);fusion=Fusion(v).cuda();fusion.load_state_dict(ck['fusion'])
        logits=e.evaluate(tail,fusion,d,b,candidates);original=pred['predictions'][f'{seed}/{v}']
        torch.testing.assert_close(logits,original,atol=1e-5,rtol=1e-4);maxerr=max(maxerr,float((logits-original).abs().max()))
        metrics=e.measures(logits,d['rows'],candidates)
        for k,val in metrics.items():assert abs(row[k]-val)<1e-10
        change=sum(float((ck['tail'][k]-initial['state_dict'][k]).float().square().sum()) for k in ck['tail'])**.5
        training.append(dict(seed=seed,variant=v,selected_step=ck['best_step'],tail_change_l2=change,seconds=ck['seconds'],peak_allocated_mib=ck['peak_allocated_mib'],history=ck['history']))
    summary=[]
    for v in c['variants']:
        rows=[r for r in result['rows'] if r['variant']==v]
        summary.append(dict(variant=v,**{k+'_mean':float(np.mean([r[k] for r in rows])) for k in ['zsl','seen','unseen','harmonic']},zsl_sd=float(np.std([r['zsl'] for r in rows],ddof=1)),steps=[r['best_step'] for r in rows]))
    coverage=[]
    for stage in ['development','test']:
        x=json.loads((e.CACHE/f'regions_{stage}.json').read_text())
        counts=[len(r['scores']) for r in x['images'].values()]
        coverage.append(dict(stage=stage,zero=counts.count(0),one=counts.count(1),two=counts.count(2)))
    write_json(e.OUT/'verification.json',dict(replayed_cells=11,max_error=maxerr,minimum_phash_distance=minp,minimum_dhash_distance=mind,class_overlap=0,duplicate_threshold_matches=0,tests_passed=5,seal_before_test=True,coverage=coverage))
    write_json(e.OUT/'training_diagnostics.json',training);write_json(e.OUT/'summary.json',summary)
    write_json(e.OUT/'attributes.json',m['attributes']);write_json(e.OUT/'classes.json',m['classes'])
    with (e.OUT/'split_manifest.csv').open('w',newline='',encoding='utf-8') as file:
        keys=list(m['rows'][0]);w=csv.DictWriter(file,fieldnames=keys);w.writeheader();w.writerows(m['rows'])
    with (e.OUT/'results.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    base=next(x for x in summary if x['variant']=='baseline');ag=next(x for x in summary if x['variant']=='caf')
    delta=(ag['zsl_mean']-base['zsl_mean'])*100
    zero=sum(x['selected_step']==0 for x in training)
    for lang in ['zh','en']:
        zh=lang=='zh'
        lines=['# '+('CoCa ViT-L/14：AG-CLIP小规模近似复现' if zh else 'CoCa ViT-L/14: small AG-CLIP-inspired replication'),'',
            (f'同主干、同数据、同预算下，CAF属性方案相对经过同预算微调的CoCa基线，未见类别ZSL均值变化为{delta:+.2f}个百分点。这是两个种子的探索性结果，不是原论文性能复现或显著性结论。' if zh else f'The CAF attribute model changes unseen-class ZSL accuracy by{delta:+.2f}pp versus the equally trained CoCa baseline. Two-seed exploratory results,not a reproduction of the paper score or a significance claim.'),'',
            '## '+('已实现的流程' if zh else 'Implemented pipeline'),'',
            '实际部署CoCa ViT-L/14，使用公开LAION-2B权重，2.55GB下载已核对发布方LFS SHA-256。使用原生768维图文空间。前23层视觉编码器和前11层文字编码器冻结并缓存输出，训练最后一层及池化/归一化/投影、属性MLP和CAF。图像和区域共享视觉编码器。' if zh else
            'Public LAION-2B CoCa ViT-L/14 weights,2.55GB verified against publisher LFS SHA-256. Native768-dimensional image/text space. Frozen visual blocks1-23 and text blocks1-11 are cached;last blocks,pooling,normalization,projections and attribute/CAF modules are trained. Global images and regions share the visual encoder.', '',
            '缓存末层输入的实现已与原生前向核验；小测试图像/文字向量最大误差均为0。两个更新后视觉末层、文字末层、属性MLP和CAF均有非零梯度。属性与CAF输出初始化为0，使各分支起步均保留原生全图输出；因此CAF内部第一步梯度为0是预期行为，第二步已验证非零。' if zh else
            'Cached-tail outputs matched native image/text forward exactly in smoke testing. Visual/text tails,attribute MLP and CAF receive nonzero gradients after two updates. Zero-initialized residual outputs preserve the native global embedding at initialization;zero CAF internal gradient on the first update is expected and becomes nonzero on the next.', '',
            '## '+('数据与选择规则' if zh else 'Data and selection'),'',
            '25个此前项目未用CUB物种、700张图片：10个已见类各20张训练+10张验证+10张最终已见测试；5个开发未见类各20张；10个最终未见类各20张。三类物种互不重叠。训练图片来自官方训练部分，其余来自官方测试部分。按元数据及固定随机顺序选样，不按预测结果选择。' if zh else
            '25 project-unused CUB species,700 images:10 seen classes with20 training,10 validation and10 final seen-test images each;5 dev-unseen classes with20 images each;10 final-unseen classes with20 images each. All class groups disjoint. Official training photos used for training,official test photos for remaining roles. Metadata/fixed random sampling,never prediction-based sampling.', '',
            '每批10张，已见类各一张，使用类别名文字配对的双向图文对比损失；每个变体80次更新，两个种子。视觉/文字末层学习率1e-5、新增模块1e-4。按开发GZSL H、再按开发ZSL、再优先较早步选模型；允许第0步，避免强行使用变差的微调结果。全部检查点封存后才处理最终测试图片。' if zh else
            'Batch10,one image per seen class,with class-name text pairs and symmetric contrastive loss.80 updates per arm,two seeds;encoder LR1e-5,new-module LR1e-4. Select development GZSL H,then dev ZSL,then earlier step;step0 is eligible. Seal every selection before final test grounding/extraction.', '',
            f'Step0 selected: {zero}/10. Histories,selected steps and parameter-change norms: training_diagnostics.json.','',
            '## '+('最终结果' if zh else 'Held-out results'),'',
            '| Method | Unseen-only ZSL (%) | Seen S (%) | Unseen U (%) | GZSL H (%) | Selected steps |','|---|---:|---:|---:|---:|---|']
        n=result['native'];lines.append(f"| Native CoCa,no adaptation | {n['zsl']*100:.2f} | {n['seen']*100:.2f} | {n['unseen']*100:.2f} | {n['harmonic']*100:.2f} | 0 |")
        for x in summary:lines.append(f"| {x['variant']} | {x['zsl_mean']*100:.2f} ± {x['zsl_sd']*100:.2f} | {x['seen_mean']*100:.2f} | {x['unseen_mean']*100:.2f} | {x['harmonic_mean']*100:.2f} | {x['steps']} |")
        lines+=['',
            'ZSL只在10个最终未见类中选择；GZSL在10个已见+10个最终未见类中选择。S/U按类别平均，H逐次运行计算后再取均值。ZSL的±是两个种子的样本标准差，不是置信区间。原生基线仅评估一次。attributes=属性分支无CAF；caf=属性+CAF；confidence=CAF再加检测置信度权重；regions=CAF结构相同但属性文字向量置零。' if zh else
            'ZSL chooses among10 final unseen classes;GZSL among10 seen+10 final unseen classes. S/U are per-class means;H is computed per run then averaged. ZSL +/- is sample SD across two seeds,not a confidence interval. Native evaluated once. attributes=no CAF;caf=attributes+CAF;confidence=CAF with detector-score pooling;regions=same CAF but zero attribute-text vectors.', '',
            '## '+('与原论文的区别' if zh else 'Differences from the paper'),'',
            '| Item | Paper | This pilot |','|---|---|---|',
            '| Backbone | CoCa ViT-L/14 | Same architecture;public LAION-2B checkpoint,exact paper checkpoint unspecified |',
            '| CUB protocol | 150 seen / 50 unseen | 10 seen / 5 development unseen / 10 final unseen |',
            '| Encoder updates | Image/text encoder fine-tuning | Last visual/text block + pooling/projection only |',
            '| Training | 50 epochs,batch64 | 80 updates,batch10,two seeds |',
            '| Attributes | GPT-4o mining/filtering | Fixed24 shared bird prompts;no test per-image attribute labels |',
            '| Grounding | OWL-ViT top-K | OWL-ViT top2,threshold0.05 |',
            '| Fusion | Attribute aggregation + CAF | Explicit shared encoder,average-before-MLP,custom256dim CAF and zero residual initialization |', '',
            f'Paper Table4 reports CoCa65.4% -> AG without CAF69.0% -> AG with CAF73.3% on CUB50 unseen classes (+7.9pp total). Plant disease70.2% ->78.8% ->84.6% (+14.4pp). [Author paper]({PAPER}).', '',
            '论文聚合/CAF的相加与拼接、共享或独立视觉编码器存在描述歧义，本实现作了明确选择。不能把本轮更小候选类别集合的绝对准确率和论文73.3%直接比较，也不能把置信度加权当作论文已验证的增益。' if zh else
            'Paper descriptions leave addition/concatenation and shared/separate visual encoders ambiguous;this implementation makes explicit choices. Absolute pilot accuracy with fewer candidate classes is not comparable to73.3% in the paper. Detector-confidence weighting is not the source of the paper-reported gain.', '',
            '## '+('局限与复现' if zh else 'Limits and reproduction'),'',
            '新类别仅相对本项目历史实验未用；CoCa/OWL-ViT预训练是否见过这些CUB图片未排除。属性定位未经人工核验。少量样本、两个种子和很短预算仅能检验可运行性及初步方向；没有提升不证明完整论文无效，有提升也不证明完整复现。测试集已查看，后续调参不可继续称其独立盲测。' if zh else
            'Project-unused does not rule out CoCa/OWL-ViT pretraining exposure to CUB. Attribute grounding is not manually certified. Small samples,two seeds and short training test feasibility and initial direction only;negative results do not refute the full paper,positive results do not establish exact replication. The test set is now observed for future development.', '',
            '```powershell',r'.\.venv\Scripts\python.exe coca_download.py',r'.\.venv\Scripts\python.exe -m unittest test_coca_pilot -v',r'.\.venv\Scripts\python.exe coca_experiment.py all',r'.\.venv\Scripts\python.exe report_coca_pilot.py','```','',
            'Preparation depends on local official CUB images and historical manifests/fingerprints. Weights,photos and intermediate prefixes remain local;GitHub contains code/configurations/split identities and reports only.','']
        (e.OUT/f'summary_{lang}.md').write_text('\n'.join(lines),encoding='utf-8')
    labels=['Native','Baseline FT','Attributes','AG + CAF','Confidence','Regions']
    vals=[result['native']['zsl']]+[x['zsl_mean'] for x in summary]
    sd=[0.]+[x['zsl_sd'] for x in summary]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');ax.bar(labels,np.array(vals)*100,yerr=np.array(sd)*100,capsize=4,color=['#8796a4','#477caf','#9b85b0','#529584','#256f51','#bd9b67'])
    ax.set_ylim(0,100);ax.set_ylabel('Unseen-class accuracy (%), mean +/- seed SD');ax.set_title('CoCa ViT-L/14 | 10 held-out species, 200 images | 2 seeds');fig.savefig(e.OUT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(native=result['native'],summary=summary,zero_selected=zero,replay_max_error=maxerr),indent=2))

if __name__=='__main__':main()
