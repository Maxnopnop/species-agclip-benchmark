"""Development-only DCLIP/Waffle-inspired controls on frozen CLIP B/16.

Fixed four descriptors per class, three randomization seeds, no fitting or tuning.
This is a matched-budget diagnostic, not an exact WaffleCLIP reproduction.
"""
import hashlib
import json
import random
import re
from pathlib import Path
import torch
from torch.nn import functional as F
from multimodal.visible_data import model_parts
from multimodal.grounded_experiment import measures

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/four_routes_v1'
CACHE = ROOT / 'cache/four_routes_v1'
SOURCE = ROOT / 'work/four_routes_review_20260929/WaffleCLIP/descriptors/descriptors_cub.json'


def canonical(s):
    return re.sub('[^a-z]', '', s.lower())


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / 'data/cub100_v1/manifest.json').read_text())
    p = json.loads((ROOT / 'configs/cub100_v1.json').read_text())
    raw = json.loads(SOURCE.read_text())
    lookup = {canonical(k): v for k, v in raw.items()}
    names = [c['name'] for c in m['classes']]
    assert len(lookup) == 200
    descriptors = [lookup[canonical(name)][:4] for name in names]
    assert all(len(d) == 4 for d in descriptors)
    def phrase(d):
        if d.startswith(('a ', 'an ')): return 'which is ' + d
        if d.startswith(('has ', 'is ')): return 'which ' + d
        return 'which has ' + d
    def prompts(lists):
        return [[f'a photo of a {n}, {phrase(d)}.' for d in ds] for n, ds in zip(names, lists)]
    variants = {'class_name': [[f'a photo of a {n}.'] for n in names],
                'dclip_four': prompts(descriptors),
                'generic_templates_four': [[s.format(n) for s in
                    ['a photo of a {}.', 'a close-up photo of a {}.',
                     'a photo of the bird {}.', 'a clear photo of a {}.']] for n in names]}
    vocabulary = sorted(set(re.findall('[a-z]+', ' '.join(d for ds in raw.values() for d in ds).lower())))
    for seed in [42, 43, 44]:
        rng = random.Random(seed)
        permutation = list(range(100))
        # Derangement keeps no species paired with its own description.
        while True:
            rng.shuffle(permutation)
            if all(i != j for i, j in enumerate(permutation)): break
        variants[f'shuffled_descriptions_{seed}'] = prompts([descriptors[j] for j in permutation])
        random_words = [' '.join(rng.choices(vocabulary, k=3)) for _ in range(2)]
        random_chars = [''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=5)) + ' ' +
                        ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=5)) for _ in range(2)]
        variants[f'waffle_inspired_four_{seed}'] = prompts([random_words + random_chars for _ in names])
    protocol = dict(scope='Development-only exploratory diagnostic; no final evaluation, no optimization.',
                    classes=100, evaluated_classes=75, descriptor_budget=4,
                    seeds=[42, 43, 44], variants=variants,
                    aggregation='Mean cosine per descriptor: mean of unit text vectors, WITHOUT renormalizing the mean.',
                    difference_from_waffle='Four prompts and descriptor-derived vocabulary instead of official external word list/default 30 prompts.',
                    descriptor_source='ExplainableML/WaffleCLIP@7a1b8ee48e31285f62ecd839fecb6b89cbef81f1',
                    descriptor_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'semantic_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    data = torch.load(ROOT / 'cache/cub100_v1/features_development.pt', weights_only=True)
    ids = [i for i, r in enumerate(data['rows']) if r['role'] != 'train_seen']
    images = data['native_views'][ids, 0]
    labels = data['labels'][ids]
    model, tokenizer, _ = model_parts('clip_b16')
    model = model.cuda()
    unique = sorted(set(t for v in variants.values() for ds in v for t in ds))
    encoded = []
    max_length = 0
    with torch.no_grad():
        for start in range(0, len(unique), 32):
            texts = unique[start:start+32]
            uncut = tokenizer(texts, padding=False, truncation=False)['input_ids']
            max_length = max(max_length, max(map(len, uncut)))
            assert max_length <= 77, 'Unexpected prompt truncation'
            tokens = tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt').to('cuda')
            encoded.append(F.normalize(model.get_text_features(**tokens).float(), dim=-1).cpu())
    bank = dict(zip(unique, torch.cat(encoded)))
    results, predictions = {}, {}
    for variant, lists in variants.items():
        text = torch.stack([torch.stack([bank[t] for t in ds]).mean(0) for ds in lists])
        logits = 20 * images @ text.T
        metrics, pred = measures(logits, labels, p, 'development')
        results[variant] = metrics
        predictions[variant] = pred
        print(variant, {k: round(metrics[k], 3) for k in ['S', 'U', 'H', 'ZSL']}, flush=True)
    # Verify the reused native baseline is encoded consistently.
    native = torch.load(ROOT / 'cache/cub100_v1/text.pt', weights_only=True)['class_text']
    ours = torch.stack([bank[f'a photo of a {n}.'] for n in names])
    error = float((native-ours).abs().max())
    assert error < 2e-5
    torch.save(dict(predictions=predictions, labels=labels), CACHE / 'semantic_predictions.pt')
    report = dict(protocol=protocol, results=results, max_token_length=max_length,
                  native_text_max_abs_error=error, photographs=len(ids),
                  caution='Three seeds vary random prompts, not independent trained models or datasets. '
                          'Descriptors are published LLM output, not guaranteed visible/accurate annotations.')
    (OUT / 'semantic_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
