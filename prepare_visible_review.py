"""Choose images before annotation and create local, unlabelled visual review sheets."""
import json,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from data_tools import ROOT,write_json


def main():
    manifest=json.loads((ROOT/'data/expanded20/manifest.json').read_text(encoding='utf-8'));rows=[]
    for split,count in [('train',2),('val',1)]:
        for label in range(20):
            group=sorted([r for r in manifest['splits'][split] if r['label']==label],key=lambda r:r['path'])
            random.Random(20260929+label+(0 if split=='train' else 1000)).shuffle(group)
            for row in group[:count]:rows.append(dict(id=len(rows),split=split,path=row['path'],label=label,sha256=row['sha256']))
    target=ROOT/'configs/visible_review_selection.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=rows:raise ValueError('Fixed selection changed')
    write_json(target,rows);out=ROOT/'work/visible_v1';out.mkdir(exist_ok=True,parents=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for page in range(10):
        canvas=Image.new('RGB',(1500,1040),'white');draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(rows[page*6:page*6+6]):
            x=(k%3)*500;y=(k//3)*520
            with Image.open(Path(manifest['image_root'])/r['path']) as im:im=im.convert('RGB')
            size=im.size;im.thumbnail((480,470));canvas.paste(im,(x+10,y+40))
            draw.text((x+10,y+8),f'ID {r["id"]} | {r["split"]} | {size[0]}x{size[1]}',fill='black',font=font)
        canvas.save(out/f'review_{page+1:02d}.jpg',quality=95)
    print('Fixed 40 training + 20 validation images. Species names and model outputs omitted from review sheets.')


if __name__=='__main__':main()
