"""Same CUB crops and prompts, encoded with the already installed CLIP B/16."""
import json
import torch
from torch.nn import functional as F
from data_tools import ROOT,digest
from .cub_data import prepare,pixels
from .cub_rich_data import text_bank as rich_bank,load_data as rich_data
from .visible_data import model_parts

CACHE=ROOT/'cache/cub_b16_v1'


def metadata():
    return dict(source_sha256=digest(__file__),model_revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text())['clip_b16'],
                crop_protocol='Exact existing 224 RGB bicubic shortest-side resize/center-crop/OpenAI normalization; whole+two OWL views; BF16 batch of three views per photo.',
                manifest_sha256=digest(ROOT/'data/cub_attributes_v1/manifest.json'),rich_bank_sha256=digest(ROOT/'cache/cub_rich_v1/text.pt'))


def text_bank():
    CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/'text.pt';meta=metadata()
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;return saved
    p,m=prepare();bank=rich_bank();model,tokenizer,processor=model_parts('clip_b16');model=model.cuda().eval()
    prompts=[f'a photo of a {c["name"]}.' for c in m['classes']]+bank['positive_prompts']+bank['negative_prompts']
    with torch.no_grad():
        features=[]
        for start in range(0,len(prompts),32):
            texts=tokenizer(prompts[start:start+32],padding='max_length',max_length=77,truncation=True,return_tensors='pt').to('cuda')
            features.append(F.normalize(model.get_text_features(**texts).float(),dim=-1).cpu())
    text=torch.cat(features);a=len(bank['attribute_text'])
    result=dict(metadata=meta,class_text=text[:20],attribute_text=text[20:20+a],negative_text=text[20+a:],positive_prompts=bank['positive_prompts'],negative_prompts=bank['negative_prompts'],columns=bank['columns'])
    torch.save(result,path);del model;torch.cuda.empty_cache();return result


def load_data(stage):
    CACHE.mkdir(parents=True,exist_ok=True);data=rich_data(stage);path=CACHE/f'features_{stage}.pt';meta=metadata()
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;data['native_views']=saved['features'];return data
    raw=pixels(stage);assert [r['path'] for r in raw['rows']]==[r['path'] for r in data['rows']]
    model,_,processor=model_parts('clip_b16');model=model.cuda().eval();features=[]
    assert processor.size['shortest_edge']==224 and processor.crop_size==dict(height=224,width=224)
    assert all(abs(a-b)<1e-6 for a,b in zip(processor.image_mean,[.48145466,.4578275,.40821073]))
    with torch.no_grad():
        for i in range(len(raw['images'])):
            with torch.autocast('cuda',dtype=torch.bfloat16):f=F.normalize(model.get_image_features(pixel_values=raw['images'][i].cuda()).float(),dim=-1).cpu()
            features.append(f)
            if i%80==0:print(f'CLIP B16 {stage} {i}/{len(raw["images"])}',flush=True)
    data['native_views']=torch.stack(features);torch.save(dict(metadata=meta,features=data['native_views']),path)
    del model,raw;torch.cuda.empty_cache();return data
