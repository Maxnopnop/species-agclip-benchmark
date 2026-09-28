"""Fixed source-to-vocabulary mapping; never inspect experiment outcomes here."""
import json
from data_tools import ROOT,write_json,digest
from multimodal.cub_rich_data import names,definition


def main():
    cfg=json.loads((ROOT/'configs/cub_descriptions_v1.json').read_text());columns=definition()['extended'];naming=names()
    # Positive observations only, two alternative phenotypes per source. Unstated
    # attributes are unknown, not negative. No class-image attributes are read.
    indices=[
        [[131,132,143],[131,132]],
        [[12,31,24,30],[12,24,99,30]],
        [[72,31,57,101,4],[13,61,5,11,156]],
        [[125,72,145,4,141],[126,66,29,5]],
        [[24,67,1],[24,1,45,46]],
        [[130,99,59,67,1],[22,42,1]],
        [[13,26,29,49],[49,13,29]],
        [[13,24,29,49],[49,13,24]],
        [[131,62,68],[131,63,2]],
        [[131,24],[126]],
        [[33,24,53,150,137,142,143],[132,151,48,143]],
        [[33,23,150,58],[14,24,53,48]],
        [[40,32,150,90],[106,6,157]],
        [[107,34,10],[13,116,119]],
        [[133,127,7,10,156],[127,76,75,7,10,156]],
        [[4,128,131,49,10],[4,127,9,10]],
        [[130,128,48,65,9,10,156],[129,33,10,156,37]],
        [[34,150,65,19],[129,128,10,156]],
        [[13,24,50,143],[6,46]],
        [[32,24,50,48],[81,85,119]]
    ]
    # Omit features absent from the train-supported 158-attribute vocabulary,
    # e.g. green back, yellow bill, pink legs and bill curvature/crest presence.
    # No guessed negative attributes, numeric occurrence rates or male/female
    # frequencies are supplied. Some alternatives overlap and are not exclusive.
    species=[]
    for item,pairs in zip(cfg['species'],indices):
        species.append(dict(label=item['label'],name=item['name'],source=item['source'],positive_alternatives=[[naming[columns[i]] for i in alternative] for alternative in pairs]))
    write_json(ROOT/'configs/cub_description_attributes_v1.json',dict(description_sha256=digest(ROOT/'configs/cub_descriptions_v1.json'),mapping_source_sha256=digest(__file__),scope='AI-assisted source-backed positive-only mapping from cited field-guide descriptions, before experiment scoring. Not expert-validated. Unstated/unsupported attributes remain unknown. The same written information is available to the description-only baseline.',species=species,description_blends=[0,.25,.5,.75,1],attribute_weights=[0,.1,.25,.5,1,2],permutation_seeds=[42,43,44]))


if __name__=='__main__':main()
