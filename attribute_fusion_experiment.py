"""Predeclared cached-score fusion; no fitting on the held-out fold's labels."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/attribute_fusion_v1'
CONFIG = ROOT / 'configs/attribute_fusion_v1.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def standardized(scores):
    return (scores - scores.mean(-1, keepdim=True)) / scores.std(-1, correction=0, keepdim=True).clamp_min(1e-8)


def stratified_folds(labels, count, seed):
    rng = np.random.default_rng(seed)
    folds = np.full(len(labels), -1, dtype=int)
    for label in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels.numpy() == label)
        assert len(ids) % count == 0
        folds[rng.permutation(ids)] = np.arange(len(ids)) % count
    return torch.tensor(folds)


def metrics(pred, labels):
    per_class = {str(c): float((pred[labels == c] == c).double().mean() * 100) for c in range(75)}
    assert all(np.isfinite(list(per_class.values())))
    s = np.mean([per_class[str(c)] for c in range(50)])
    u = np.mean([per_class[str(c)] for c in range(50, 75)])
    return dict(S=float(s), U=float(u), H=float(2*s*u/max(s+u, 1e-12)),
                accuracy=float((pred == labels).double().mean()*100), per_class=per_class)


def changes(pred, native, labels):
    answer = {}
    for name, mask in [('all', torch.ones(len(labels), dtype=torch.bool)), ('seen', labels < 50), ('unseen', labels >= 50)]:
        n_ok = native[mask] == labels[mask]
        f_ok = pred[mask] == labels[mask]
        answer[name] = dict(corrected=int((~n_ok & f_ok).sum()), newly_wrong=int((n_ok & ~f_ok).sum()),
                            net_correct=int(f_ok.sum()-n_ok.sum()), original_errors=int((~n_ok).sum()),
                            wrong_to_different_wrong=int((~n_ok & ~f_ok & (pred[mask] != native[mask])).sum()))
    return answer


def select_weight(predictions, labels, weights):
    """Input is [weight, training seed, selection image], never held-out labels."""
    scores = [float(np.mean([metrics(p, labels)['H'] for p in by_seed])) for by_seed in predictions]
    best = 0
    for i in range(1, len(weights)):
        if scores[i] > scores[best] + 1e-10:
            best = i
    return best, scores


def locked_protocol(config):
    fingerprint = {str(p.relative_to(ROOT)): digest(p) for p in [CONFIG, Path(__file__)]}
    path = OUT / 'protocol.json'
    if path.exists():
        prior = json.loads(path.read_text(encoding='utf-8'))
        assert prior['source'] == fingerprint and prior['config'] == config, 'Protocol changed; use a new version.'
    else:
        save(path, dict(locked_at_utc=datetime.now(timezone.utc).isoformat(), source=fingerprint, config=config))


def load_scores(config):
    from multimodal.attribute_path_v1 import source as attribute_source
    from multimodal.grounded_experiment import measures
    manifest = json.loads((ROOT/'data/cub100_v1/manifest.json').read_text())
    p = json.loads((ROOT/'configs/cub100_v1.json').read_text())
    rows = [r for r in manifest['rows'] if r['role'] in ['dev_seen', 'dev_unseen']]
    sig_path = ROOT/'cache/cub_siglip_v1/features_development.pt'
    text_path = ROOT/'cache/cub_siglip_v2/text.pt'
    sig = torch.load(sig_path, weights_only=True)
    text = torch.load(text_path, weights_only=True)['class_text']
    ids = [i for i, r in enumerate(sig['rows']) if r['role'] != 'train_seen']
    assert [(sig['rows'][i]['image_id'], sig['rows'][i]['label']) for i in ids] == [(r['image_id'], r['label']) for r in rows]
    labels = sig['labels'][ids]
    assert len(labels) == 500 and torch.equal(labels, torch.tensor([r['label'] for r in rows]))
    native = (20 * sig['native_views'][ids, 0] @ text.T)[:, :75]
    native_metrics = metrics(native.argmax(-1), labels)
    assert abs(native_metrics['H'] - 75.4645503) < 1e-4
    current_source = attribute_source()
    scores, checks = {}, {}
    source_paths = [sig_path, text_path, ROOT/'data/cub100_v1/manifest.json']
    for variant in config['variants']:
        cells = []
        for seed in config['seeds']:
            path = ROOT/f'runs/attribute_path_v1/{variant}_{seed}.pt'
            ck = torch.load(path, weights_only=True)
            assert ck['source'] == current_source
            raw = 20 * F.normalize(ck['evidence'], dim=-1) @ ck['state']['profiles'].T
            m, pred = measures(raw, labels, p, 'development')
            assert torch.equal(pred, ck['predictions'])
            assert abs(m['H'] - ck['metrics']['classification']['H']) < 1e-6
            cells.append(raw[:, :75])
            checks[f'{variant}_{seed}'] = dict(predictions_reproduced=True, H=m['H'])
            source_paths.append(path)
        scores[variant] = torch.stack(cells)
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    return native, scores, labels, rows, dict(input_hashes=hashes, attribute_source=current_source, reconstruction=checks)


def summarize(cells):
    return {key: dict(mean=float(np.mean([c['metrics'][key] for c in cells])),
                      std=float(np.std([c['metrics'][key] for c in cells], ddof=1))) for key in ['S', 'U', 'H', 'accuracy']}


def bootstrap(cells, baseline, config):
    rng = np.random.default_rng(config['bootstrap_seed'])
    si = rng.integers(0, 50, (config['bootstrap_replicates'], 50))
    ui = rng.integers(50, 75, (config['bootstrap_replicates'], 25))
    def sampled(values):
        s, u = values[:, si].mean(-1), values[:, ui].mean(-1)
        return (2*s*u/np.maximum(s+u, 1e-12)).mean(0)
    arrays = {v: np.array([[c['metrics']['per_class'][str(k)] for k in range(75)] for c in runs]) for v, runs in cells.items()}
    arrays['siglip2'] = np.array([[baseline['per_class'][str(k)] for k in range(75)]])
    learned = sampled(arrays['learned_part'])
    return {v: dict(paired_class_bootstrap_95_percentile=np.quantile(learned-sampled(a), [.025, .975]).tolist())
            for v, a in arrays.items() if v != 'learned_part'}


def run(config):
    native, attr, labels, rows, checks = load_scores(config)
    folds = stratified_folds(labels, config['folds'], config['fold_seed'])
    base_pred = native.argmax(-1)
    nz = standardized(native)
    assert torch.equal(nz.argmax(-1), base_pred)
    cells, grids, selections, exports = {}, {}, {}, {}
    for variant, raw in attr.items():
        az = standardized(raw)
        predicted = torch.stack([(nz[None] + weight * az).argmax(-1) for weight in config['weights']])
        assert all(torch.equal(p, base_pred) for p in predicted[0])
        grids[variant] = []
        for i, weight in enumerate(config['weights']):
            runs = [dict(seed=seed, metrics=metrics(predicted[i, j], labels), changes=changes(predicted[i, j], base_pred, labels))
                    for j, seed in enumerate(config['seeds'])]
            grids[variant].append(dict(weight=weight, summary=summarize(runs), seeds=runs))
        oof = torch.full((len(config['seeds']), len(labels)), -1, dtype=torch.long)
        selections[variant] = []
        for fold in range(config['folds']):
            fit, held = folds != fold, folds == fold
            index, fit_h = select_weight(predicted[:, :, fit], labels[fit], config['weights'])
            oof[:, held] = predicted[index, :, held]
            selections[variant].append(dict(fold=fold, fit_images=int(fit.sum()), heldout_images=int(held.sum()),
                                             weight=config['weights'][index], fit_mean_H=fit_h))
        cells[variant] = [dict(seed=seed, metrics=metrics(oof[j], labels), changes=changes(oof[j], base_pred, labels))
                          for j, seed in enumerate(config['seeds'])]
        index, fit_h = select_weight(predicted, labels, config['weights'])
        exports[variant] = dict(weight=config['weights'][index], fit_mean_H=fit_h,
                                note='All-development selection for future external inference, not an independently evaluated deployment metric.')
    result = dict(protocol_source=json.loads((OUT/'protocol.json').read_text())['source'], verification=checks,
                  baseline=metrics(base_pred, labels), summary={v: summarize(c) for v, c in cells.items()},
                  crossfit=cells, fold_selections=selections, descriptive_full_development_grid=grids,
                  future_inference_selection=exports, exploratory_intervals=bootstrap(cells, metrics(base_pred, labels), config),
                  limitations=config['scope'] + ' ' + config['uncertainty'])
    save(OUT/'results.json', result)
    # IDs/fold membership enable auditing without publishing images, labels used as inputs, or weights.
    save(OUT/'folds.json', [dict(image_id=r['image_id'], label=r['label'], fold=int(f)) for r, f in zip(rows, folds)])
    print(json.dumps(dict(baseline=result['baseline']['H'], summary=result['summary'], selections=exports,
                         intervals=result['exploratory_intervals']), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(CONFIG.read_text())
    locked_protocol(config)
    if not args.lock_only:
        run(config)


if __name__ == '__main__':
    main()
