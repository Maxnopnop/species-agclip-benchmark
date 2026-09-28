"""Fix SigLIP 2 text case; reuse the independently verified visual cache."""
import gc,json
import torch
from torch.nn import functional as F
from data_tools import ROOT,digest
from .cub100_data import prompt,DATA
from .cub_siglip_data import prepare,load_data,photo_views
from .visible_data import model_parts

CACHE=ROOT/'cache/cub_siglip_v2'


def text_bank():
    p,m=prepare();CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/'text.pt'
    meta=dict(source_sha256=digest(__file__),config_sha256=digest(ROOT/'configs/cub_siglip_v2.json'),manifest_sha256=digest(DATA/'manifest.json'),revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text())['siglip2_b16'],text_normalization='Explicit lowercase before fast Gemma tokenization. Tokenizer config do_lower_case is not applied by this local fast tokenizer.')
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;return saved
    positives=[prompt(a['name']).lower() for a in m['attributes']];negatives=[s.replace('a bird with','a bird without') for s in positives]
    texts=[f'a photo of a {c["name"]}.'.lower() for c in m['classes']]+positives+negatives;assert all(t==t.lower() for t in texts)
    model,tokenizer,_=model_parts('siglip2_b16');model=model.cuda().eval();features=[]
    with torch.no_grad():
        for i in range(0,len(texts),32):
            batch=tokenizer(texts[i:i+32],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda');assert (batch['input_ids'][:,-1]==tokenizer.pad_token_id).all()
            features.append(F.normalize(model.get_text_features(**batch).float(),dim=-1).cpu())
    text=torch.cat(features);a=len(positives);result=dict(metadata=meta,class_text=text[:100],attribute_text=text[100:100+a],negative_text=text[100+a:],positive_prompts=positives,negative_prompts=negatives)
    torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result
