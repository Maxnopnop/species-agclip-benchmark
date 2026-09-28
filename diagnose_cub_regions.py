"""Read-only development diagnostics of detector coverage and crop redundancy."""
import argparse,json
from collections import Counter
import numpy as np
import torch
from data_tools import ROOT,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='cub_attributes_v1',choices=['cub_attributes_v1','cub100_v1']);a=parser.parse_args()
    if a.version=='cub100_v1':
        from multimodal.cub100_data import load_data,prepare
    else:
        from multimodal.cub_token_experiment import load_data
        from multimodal.cub_data import prepare
    p,m=prepare();data=load_data('development');records=json.loads((ROOT/f'cache/{a.version}/regions_development.json').read_text())['images'];areas=[];similarities=[];overlaps=[];queries=Counter();no_region=0;positive_retained=[]
    def iou(a,b):
        x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));intersection=x*y
        return intersection/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
    for i,row in enumerate(data['rows']):
        r=records[row['path']];w,h=row['size'];boxes=r['boxes'];no_region+=not bool(boxes)
        for j,box in enumerate(boxes):
            areas.append((box[2]-box[0])*(box[3]-box[1])/(w*h));similarities.append(float(data['native_views'][i,0]@data['native_views'][i,j+1]));queries[str(r['attribute_ids'][j])]+=1
        if len(boxes)==2:overlaps.append(iou(*boxes))
        gold=data['targets'][i];positive=gold[0]==1
        if positive.any():positive_retained.append(float(((gold[1:]==1).any(0)&positive).sum()/positive.sum()))
    def stats(x):
        return dict(mean=float(np.mean(x)),median=float(np.median(x)),p10=float(np.percentile(x,10)),p90=float(np.percentile(x,90))) if x else None
    report=dict(version=a.version,stage='development_only',images=len(data['rows']),regions=len(areas),no_region_images=no_region,query_counts=dict(queries),box_area_fraction=stats(areas),global_region_cosine=stats(similarities),global_region_cosine_above_095_fraction=float(np.mean(np.array(similarities)>.95)),two_region_iou=stats(overlaps),two_region_iou_above_080_fraction=float(np.mean(np.array(overlaps)>.8)),global_positive_attributes_retained_in_any_region=stats(positive_retained),scope='Descriptive diagnostics only. Keypoint-based label retention is not localization IoU against true part masks or proof that an attribute is visible. Similar frozen embeddings indicate redundancy, not detector correctness.')
    write_json(ROOT/f'reports/{a.version}/region_diagnosis.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
