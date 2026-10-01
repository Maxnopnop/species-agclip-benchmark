"""Replay all held-out decisions and verify one image-only inference example."""
import json
import torch
from local_attribute_experiment import ROOT, RUN, OUT, CONFIG, predict_fused, metrics, changes, source, save
from predict_local_attribute import export_policy, predict


def main():
    torch.set_num_threads(4)
    c=json.loads(CONFIG.read_text());r=json.loads((OUT/'results.json').read_text())
    e=torch.load(RUN/'evaluation.pt',weights_only=True)
    assert source()==r['source']==e['source']
    baseline=e['native'].argmax(-1)
    verified=[]
    for key,existing in e['predictions'].items():
        variant,policy=key.split('/')
        out=torch.full_like(existing,-1)
        for row in r['selections'][key]:
            held=e['folds']==row['fold'];choice=row['chosen']
            out[:,held]=predict_fused(e['native'][held][None],e['attributes'][variant][:,held],
                                     choice['weight'],choice['penalty'],policy=='guarded',c['native_margin_threshold'])
        assert torch.equal(out,existing)
        for j,seed in enumerate(c['seeds']):
            assert metrics(out[j],e['labels'])==r['crossfit'][key][j]['metrics']
            assert changes(out[j],baseline,e['labels'])==r['crossfit'][key][j]['changes']
        verified.append(key)
    policy=export_policy()
    # Deterministic example: first unseen error corrected by the evaluated seed42 model.
    pred=e['predictions']['repaired_local/guarded'][0]
    mask=(e['labels']>=50)&(baseline!=e['labels'])&(pred==e['labels'])
    index=int(mask.nonzero()[0])
    manifest=json.loads((ROOT/'data/cub100_v1/manifest.json').read_text())
    row=next(x for x in manifest['rows'] if x['image_id']==e['row_ids'][index])
    result,ns,scores,q=predict(ROOT/manifest['image_root']/row['path'])
    torch.testing.assert_close(ns,e['native'][index],atol=.02,rtol=.002)
    torch.testing.assert_close(scores,e['attributes']['repaired_local'][0,index],atol=.001,rtol=.0001)
    deployed_cached=predict_fused(e['native'][index:index+1],e['attributes']['repaired_local'][0,index:index+1],policy['weight'],policy['penalty'],True,policy['margin_threshold'])
    assert result['prediction']==int(deployed_cached[0])==int(pred[index])
    verification=dict(source=r['source'],replayed_variants=len(verified),replayed_seed_cells=len(verified)*3,
                      example_image_id=row['image_id'],example_true_label=row['label'],image_only_result=result,
                      native_score_max_difference=float((ns-e['native'][index]).abs().max()),
                      local_score_max_difference=float((scores-e['attributes']['repaired_local'][0,index]).abs().max()),
                      scope='One deliberately selected correction verifies plumbing, not additional accuracy evidence. No true image metadata was passed to prediction.')
    save(OUT/'verification.json',verification)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
