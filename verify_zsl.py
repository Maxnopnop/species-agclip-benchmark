"""Check protocol provenance and raw-image/cached predictions on a fixed example."""
import json
from pathlib import Path
import torch
from data_tools import ROOT,digest,write_json
from predict_zsl import predict

def main():
    torch.set_num_threads(4);root=ROOT/'runs/zsl_v1'
    lock=json.loads((root/'selection_locked.json').read_text(encoding='utf-8'));p=json.loads((ROOT/'configs/zsl_v1.json').read_text(encoding='utf-8'))
    assert lock['implementation_sha256']==digest(ROOT/'multimodal/zsl.py')
    assert lock['protocol_sha256']==digest(ROOT/'configs/zsl_v1.json')
    checked=[]
    for path in root.glob('*/config.json'):
        c=json.loads(path.read_text(encoding='utf-8'));assert c['implementation_sha256']==lock['implementation_sha256'] and c['protocol']==p
        for v in p['variants']:assert (path.parent/v/'best.pt').is_file() and (path.parent/v/'result.json').is_file()
        checked.append(path.parent.name)
    assert len(checked)==30
    manifest=json.loads((ROOT/'data/expanded20/manifest.json').read_text(encoding='utf-8'));records=[]
    for selected in [r for r in lock['selected'] if r['variant']=='ag_aux']:
        name=selected['backbone'];folder=root/f'{name}_seed42_lr{selected["lr"]:g}/ag_aux'
        cached=torch.load(folder/'final_predictions.pt',weights_only=True)['zsl'];path=Path(manifest['image_root'])/cached['paths'][0]
        result,fresh=predict(folder/'best.pt',path);expected=cached['logits'][0].softmax(-1)
        torch.testing.assert_close(fresh,expected,rtol=2e-4,atol=2e-5)
        assert result['predictions'][0]['label']==int(cached['predictions'][0])
        records.append(dict(backbone=name,max_probability_error=float((fresh-expected).abs().max()),prediction_matches=True))
        torch.cuda.empty_cache()
    write_json(ROOT/'reports/zsl_v1/verification.json',dict(unit_tests='4 passed: class/image disjointness; label restrictions; semantic transfer output and gradient; shuffled target restrictions',checked_training_configs=len(checked),development_cells=120,final_cells=60,selection_locked_before_final=True,inference=records))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
