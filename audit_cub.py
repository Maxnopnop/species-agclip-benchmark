"""Audit only training/development annotations and detector proposals."""
import json,math,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from data_tools import ROOT,write_json
from multimodal.cub_data import prepare,pixels,VERSION,REPORT
from multimodal.cub_experiment import setup

def main():
    setup();p,m=prepare();data=pixels('development');regions=json.loads((ROOT/f'cache/{VERSION}/regions_development.json').read_text(encoding='utf-8'))['images'];train=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen'];target=data['targets'][train];coverage=[]
    for i,a in enumerate(m['attributes']):coverage.append(dict(name=a['name'],positive=int((target[:,0,i]==1).sum()),negative=int((target[:,0,i]==0).sum()),unknown=int((target[:,0,i]<0).sum()),region_known=int((target[:,1:,i]>=0).sum())))
    write_json(REPORT/'coverage.json',dict(attributes=coverage,train_photos=len(train),development_photos=len(data['rows'])-len(train),train_known_global=int((target[:,0]>=0).sum()),train_possible_global=target.shape[0]*target.shape[-1],train_known_regional=int((target[:,1:]>=0).sum()),images_with_regions=int(data['valid'].any(-1).sum()),development_total=len(data['rows']),scope='CUB crowd annotations with confidence>=3 and visible part in CLIP center crop; part points are not full part segmentations. Not expert re-annotation.'))
    chosen=[next(i for i in train if data['rows'][i]['label']==c) for c in p['seen_classes']];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);folder=ROOT/f'work/{VERSION}';folder.mkdir(parents=True,exist_ok=True)
    for page in range(math.ceil(len(chosen)/4)):
        canvas=Image.new('RGB',(1100,1200),'white');draw=ImageDraw.Draw(canvas)
        for j,i in enumerate(chosen[page*4:page*4+4]):
            row=data['rows'][i];rec=regions[row['path']]
            with Image.open(Path(m['image_root'])/row['path']) as im:im=im.convert('RGB')
            d=ImageDraw.Draw(im)
            for box in rec['boxes']:d.rectangle(box,outline='red',width=3)
            for points in row['attribute_points']:
                for x,y in points:d.ellipse((x-3,y-3,x+3,y+3),fill='cyan')
            im.thumbnail((460,280));canvas.paste(im,(0,j*300))
            positive=[m['attributes'][a]['name'] for a,v in enumerate(data['targets'][i,0]) if v==1];negative=[m['attributes'][a]['name'] for a,v in enumerate(data['targets'][i,0]) if v==0]
            text=f'{m["classes"][row["label"]]["name"]}\nPositive: '+', '.join(positive)+'\nKnown negative: '+', '.join(negative[:6])+'\nRed: predicted regions. Cyan: annotated visible parts.'
            draw.multiline_text((475,j*300+5),'\n'.join(textwrap.fill(line,65) for line in text.splitlines()),font=font,fill='black')
        canvas.save(folder/f'annotation_audit_{page+1}.jpg',quality=92)
    print(json.dumps(coverage,indent=2))

if __name__=='__main__':main()
