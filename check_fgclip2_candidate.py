"""Frozen FG-CLIP 2 candidate check; development images only, no training."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
os.environ['HF_MODULES_CACHE'] = str(ROOT / 'cache/huggingface/modules')
import hashlib
import json
import time
import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from multimodal.grounded_experiment import measures

OUT = ROOT / 'reports/four_routes_v1'
CACHE = ROOT / 'cache/four_routes_v1'
MODEL = CACHE / 'models/fgclip2_base'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    metadata = json.loads((ROOT / 'work/four_routes_review_20260929/fgclip2_hf_metadata.json').read_text())
    expected = next(x['lfs']['sha256'] for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
    actual = digest(MODEL / 'model.safetensors')
    assert expected == actual, 'Weight integrity mismatch'
    m = json.loads((ROOT / 'data/cub100_v1/manifest.json').read_text())
    p = json.loads((ROOT / 'configs/cub100_v1.json').read_text())
    rows = [r for r in m['rows'] if r['role'] in ['dev_seen', 'dev_unseen']]
    assert len(rows) == 500
    prompts = [f"a photo of a {c['name']}.".lower() for c in m['classes']]
    protocol = dict(model='qihoo360/fg-clip2-base', revision=metadata['sha'],
                    weights_sha256=actual, images=len(rows), candidates=75,
                    max_num_patches=256, batch_size=4, dtype='float32',
                    prompts=prompts, training=False, evaluation='development only',
                    region_smoke='First development image, fixed full-image and upper-half boxes; no true labels/parts supplied.',
                    localization='First development image per class (75); head/wing/breast/tail prompts; visible CUB parts evaluated only after predictions. Hit radius 0.1 original-image diagonal. Controls: cyclic wrong query, image center, uniform valid patch center.',
                    source_sha256=digest(Path(__file__)),
                    manifest_sha256=digest(ROOT / 'data/cub100_v1/manifest.json'),
                    custom_code_sha256={f: digest(MODEL / f) for f in ['modeling_fgclip2.py', 'configuration_fgclip2.py']})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fgclip2_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    start = time.perf_counter()
    # Only the pinned, locally reviewed implementation is executed.
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    load_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    texts, features = [], []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(prompts), 16):
            tokens = tokenizer(prompts[offset:offset+16], padding='max_length', max_length=64,
                               truncation=True, return_tensors='pt').to('cuda')
            texts.append(F.normalize(model.get_text_features(**tokens, walk_type='short').float(), dim=-1).cpu())
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset+4]
            images = []
            for row in batch:
                path = Path(m['image_root']) / row['path']
                assert digest(path) == row['sha256']
                with Image.open(path) as im:
                    images.append(im.convert('RGB'))
            inputs = processor(images=images, max_num_patches=256, return_tensors='pt').to('cuda')
            features.append(F.normalize(model.get_image_features(**inputs).float(), dim=-1).cpu())
            if offset % 100 == 0:
                print(f'FG-CLIP2 images: {offset + len(batch)}/{len(rows)}', flush=True)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        with Image.open(Path(m['image_root']) / rows[0]['path']) as im:
            image = im.convert('RGB')
        inputs = processor(images=image, max_num_patches=256, return_tensors='pt').to('cuda')
        dense = model.get_image_dense_feature(**inputs)
        h, w = map(int, inputs['spatial_shapes'][0])
        assert int(inputs['pixel_attention_mask'].sum()) == h*w
        captions = ['the head of a bird', 'the wing of a bird', 'the tail of a bird']
        box_tokens = tokenizer(captions, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
        box_text = model.get_text_features(**box_tokens, walk_type='box')
        local_scores = F.normalize(dense[0, :h*w].float(), dim=-1) @ F.normalize(box_text.float(), dim=-1).T
        image_w, image_h = image.size
        regions = model.get_image_region_features(**inputs, image_sizes=[(image_h, image_w)],
                    region_infos=[[[0, 0, image_w, image_h], [0, 0, image_w, image_h/2]]])
        long_tokens = tokenizer(['a photograph of a bird with visible head, wings, and tail.'],
                    padding='max_length', max_length=196, truncation=True, return_tensors='pt').to('cuda')
        long_text = model.get_text_features(**long_tokens, walk_type='long')
        for tensor in [dense, box_text, local_scores, regions[0], long_text]:
            assert torch.isfinite(tensor).all()
        assert regions[0].shape == (2, texts[0].shape[-1])
        # Anatomical pointing diagnostic, not segmentation or color-attribute accuracy.
        part_groups = {'head': [2, 5, 6, 7, 10, 11, 15], 'wing': [9, 13],
                       'breast': [4], 'tail': [14]}
        location_tokens = tokenizer([f'the {part} of a bird' for part in part_groups],
                    padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
        location_text = F.normalize(model.get_text_features(**location_tokens, walk_type='box').float(), dim=-1)
        location_rows, visited, pair_checks = [], set(), []
        for row in rows:
            if row['label'] in visited:
                continue
            visited.add(row['label'])
            with Image.open(Path(m['image_root']) / row['path']) as im:
                location_image = im.convert('RGB')
            location_inputs = processor(images=location_image, max_num_patches=256, return_tensors='pt').to('cuda')
            lh, lw = map(int, location_inputs['spatial_shapes'][0])
            local = model.get_image_dense_feature(**location_inputs)[0, :lh*lw]
            scores = (F.normalize(local.float(), dim=-1) @ location_text.T).cpu()
            iw, ih = location_image.size
            yy, xx = torch.meshgrid(torch.arange(lh), torch.arange(lw), indexing='ij')
            grid = torch.stack([(xx.flatten()+.5)/lw*iw, (yy.flatten()+.5)/lh*ih], dim=-1)
            peaks = grid[scores.argmax(0)]
            centered = scores - scores.mean(0, keepdim=True)
            wing_tail_correlation = F.cosine_similarity(centered[:, 1], centered[:, 3], dim=0)
            pair_checks.append(dict(image_id=row['image_id'],
                wing_tail_same_peak=bool(scores[:, 1].argmax() == scores[:, 3].argmax()),
                wing_tail_heatmap_correlation=float(wing_tail_correlation)))
            tolerance = .1 * (iw**2+ih**2)**.5
            for j, (part, ids) in enumerate(part_groups.items()):
                targets = [row['parts'][str(pid)][:2] for pid in ids if row['parts'][str(pid)][2]]
                if not targets:
                    continue
                targets = torch.tensor(targets)
                distance = torch.cdist(grid.float(), targets.float()).min(-1).values
                location_rows.append(dict(image_id=row['image_id'], label=row['label'], part=part,
                    correct_query_hit=float(torch.cdist(peaks[j:j+1], targets).min() <= tolerance),
                    wrong_query_hit=float(torch.cdist(peaks[(j+1)%4:(j+1)%4+1], targets).min() <= tolerance),
                    center_hit=float(torch.cdist(torch.tensor([[iw/2, ih/2]]), targets).min() <= tolerance),
                    uniform_grid_expected_hit=float((distance <= tolerance).float().mean()),
                    query_hits={q: float(torch.cdist(peaks[k:k+1], targets).min() <= tolerance)
                                for k, q in enumerate(part_groups)},
                    normalized_point_distance=float(torch.cdist(peaks[j:j+1], targets).min() / (iw**2+ih**2)**.5)))
        assert len(visited) == 75
        localization = {}
        for part in part_groups:
            subset = [r for r in location_rows if r['part'] == part]
            localization[part] = dict(n=len(subset), **{key: sum(r[key] for r in subset)/len(subset)
                for key in ['correct_query_hit', 'wrong_query_hit', 'center_hit', 'uniform_grid_expected_hit', 'normalized_point_distance']})
        confusion = {part: {query: sum(r['query_hits'][query] for r in location_rows if r['part'] == part)
                        / localization[part]['n'] for query in part_groups} for part in part_groups}
        extra_diagnostic = dict(scope='Post-result exploratory check prompted by low wing pointing accuracy; no changes to model/prompts/primary metrics.',
            target_part_by_query_hit_rate=confusion,
            wing_tail_same_peak_fraction=sum(r['wing_tail_same_peak'] for r in pair_checks)/len(pair_checks),
            mean_wing_tail_heatmap_correlation=sum(r['wing_tail_heatmap_correlation'] for r in pair_checks)/len(pair_checks))
    features, texts = torch.cat(features), torch.cat(texts)
    assert torch.isfinite(features).all() and torch.isfinite(texts).all()
    labels = torch.tensor([r['label'] for r in rows])
    metrics, predictions = measures(20 * features @ texts.T, labels, p, 'development')
    smoke = dict(dense_shape=list(dense.shape), valid_spatial_shape=[h, w],
                 region_shape=list(regions[0].shape), long_text_shape=list(long_text.shape),
                 box_text_shape=list(box_text.shape), all_finite=True,
                 limitation='Interface smoke only; finite heatmaps do not establish correct anatomical localization.')
    report = dict(protocol=protocol, metrics=metrics, smoke=smoke,
                  localization=localization,
                  extra_diagnostic=extra_diagnostic,
                  parameter_count=sum(x.numel() for x in model.parameters()),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                  load_seconds=load_seconds, inference_seconds=inference_seconds,
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  caution='Frozen backbone candidate, not an AG ablation; different pretraining/resolution from CLIP. No foundation-data overlap guarantee.')
    torch.save(dict(features=features, text=texts, labels=labels, rows=rows, predictions=predictions,
                    dense_scores=local_scores.cpu(), smoke=smoke), CACHE / 'fgclip2_development.pt')
    (OUT / 'fgclip2_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT / 'fgclip2_pointing_records.json').write_text(json.dumps(location_rows, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'protocol'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
