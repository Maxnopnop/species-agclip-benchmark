"""Reused-data exploratory token experiment, with prelocked model selection."""
import json, math, time
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT, digest, write_json
from run import seed_all
from .cub_data import pixels, text_bank, prepare
from .cub_model import masked_bce
from .cub_tokens import TokenFusion
from .grounded_experiment import measures, rank
from .visible_train import attribute_metrics

VERSION = 'cub_tokens_v1'
OUT = ROOT / 'runs' / VERSION
REPORT = ROOT / 'reports' / VERSION
CONFIG = ROOT / 'configs' / f'{VERSION}.json'


def setup():
    seed_all(42)
    torch.backends.mha.set_fastpath_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def provenance():
    paths = ['multimodal/cub_tokens.py', 'multimodal/cub_token_experiment.py',
             'multimodal/cub_data.py', 'multimodal/cub_model.py', 'multimodal/models.py',
             'multimodal/grounded_experiment.py', 'multimodal/visible_train.py',
             'configs/cub_tokens_v1.json', 'configs/cub_attributes_v1.json',
             'data/cub_attributes_v1/manifest.json', 'cache/cub_attributes_v1/text.pt']
    return {path: digest(ROOT / path) for path in paths}


def load_data(stage):
    raw = pixels(stage)
    # Validate the immutable cache with its original loader before dropping pixels.
    data = {k: raw[k] for k in ['native_views', 'valid', 'targets', 'labels', 'rows']}
    del raw
    return data


def subset(data, ids, device='cpu'):
    return {k: data[k][ids].to(device) for k in ['native_views', 'valid', 'targets', 'labels']}


def attribute_score(targets, probabilities):
    global_metrics = attribute_metrics(targets[:, 0], probabilities[:, 0])
    regional = attribute_metrics(targets[:, 1:].flatten(0, 1), probabilities[:, 1:].flatten(0, 1))
    return dict(global_map=global_metrics['attribute_map'], regional_map=regional['attribute_map'],
                selection_map=(global_metrics['attribute_map'] + regional['attribute_map']) / 2)


def train_probe(data, bank, c):
    path = OUT / 'probe.pt'
    if path.exists():
        result = torch.load(path, weights_only=True)
        assert result['provenance'] == provenance()
        return result
    ti = [i for i, r in enumerate(data['rows']) if r['role'] == 'train_seen']
    vi = [i for i, r in enumerate(data['rows']) if r['role'] != 'train_seen']
    x, y = data['native_views'], data['targets']
    history, best, best_state = [], None, None
    for lr in c['probe_learning_rates']:
        seed_all(42)
        model = nn.Linear(512, y.shape[-1])
        with torch.no_grad():
            model.weight.copy_(10 * (bank['attribute_text'] - bank['negative_text']))
            model.bias.zero_()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        for step in range(max(c['probe_steps']) + 1):
            if step in c['probe_steps']:
                with torch.no_grad(): metrics = attribute_score(y[vi], model(x[vi]).sigmoid())
                row = dict(lr=lr, step=step, **metrics)
                history.append(row)
                if best is None or row['selection_map'] > best['selection_map']:
                    best = row
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step == max(c['probe_steps']): break
            opt.zero_grad()
            pred = model(x[ti])
            loss = masked_bce(pred[:, 0], y[ti, 0]) + c['probe_regional_weight'] * masked_bce(pred[:, 1:], y[ti, 1:])
            loss.backward(); opt.step()
    model.load_state_dict(best_state)
    with torch.no_grad():
        probability = model(x[ti]).sigmoid()
        valid_views = torch.cat([torch.ones(len(ti), 1, dtype=torch.bool), data['valid'][ti]], 1)
        mean = (probability * valid_views[..., None]).sum(0) / valid_views.sum(0)[:, None].clamp_min(1)
    result = dict(state=best_state, mean_probability=mean, selection=best, provenance=provenance())
    torch.save(result, path)
    write_json(REPORT / 'probe_development.json', dict(selected=best, history=history))
    print('Attribute probe locked: ' + json.dumps(best), flush=True)
    return result


def build(variant, seed, bank, probe, c):
    seed_all(seed)
    return TokenFusion(bank, probe['state'], probe['mean_probability'], variant, c)


@torch.no_grad()
def evaluate(model, data, p, stage, intervention=None):
    model.eval()
    ids = [i for i, r in enumerate(data['rows']) if r['role'] != 'train_seen']
    outputs = []
    device = next(model.parameters()).device
    for start in range(0, len(ids), 64):
        b = subset(data, ids[start:start + 64], device)
        o = model(b['native_views'], b['valid'], intervention)
        outputs.append({k: v.cpu() for k, v in o.items()})
    output = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    labels = data['labels'][ids]
    metrics, pred = measures(output['logits'], labels, p, stage)
    metrics['residual_gate'] = float(model.max_residual * model.gate.tanh())
    return metrics, dict(**output, labels=labels, predictions=pred,
                         paths=[data['rows'][i]['path'] for i in ids])


def name(variant, seed, lr): return f'{variant}_seed{seed}_lr{lr:g}'


def train_cell(variant, seed, lr, data, p, c, bank, probe):
    folder = OUT / name(variant, seed, lr)
    folder.mkdir(exist_ok=True)
    config = dict(variant=variant, seed=seed, lr=lr, protocol=c, provenance=provenance(), probe_sha256=digest(OUT / 'probe.pt'))
    if (folder / 'result.json').exists():
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        assert result['config'] == config
        return result
    model = build(variant, seed, bank, probe, c).cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
    ti = torch.tensor([i for i, r in enumerate(data['rows']) if r['role'] == 'train_seen'])
    assert set(data['labels'][ti].tolist()) == set(p['seen_classes'])
    frozen = {k: v.clone() for k, v in model.state_dict().items() if k.startswith('probe.') or k in ['class_text', 'attribute_text']}
    gen = torch.Generator().manual_seed(seed + 191)
    best, history = None, []
    start = time.time()
    for step in range(c['updates'] + 1):
        if step in c['validation_steps']:
            metrics, _ = evaluate(model, data, p, 'development')
            history.append(dict(step=step, metrics=metrics))
            if best is None or rank(metrics) > rank(best):
                best, best_step = metrics, step
                torch.save(dict(state={k: v.cpu().clone() for k, v in model.state_dict().items()}, config=config, step=step), folder / 'best.pt')
        if step == c['updates']: break
        model.train()
        b = subset(data, ti[torch.randperm(len(ti), generator=gen)[:c['batch_size']]], 'cuda')
        o = model(b['native_views'], b['valid'])
        seen = torch.tensor(p['seen_classes'], device='cuda')
        labels = (b['labels'][:, None] == seen).long().argmax(-1)
        loss = F.cross_entropy(o['logits'][:, seen], labels) + c['preserve_weight'] * (1 - (o['embedding'] * o['global_embedding']).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups: g['lr'] = lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step + 1) / c['updates'])))
    for k, v in frozen.items(): assert torch.equal(v, model.state_dict()[k])
    result = dict(config=config, best_step=best_step, development=best, seconds=time.time() - start,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), frozen_assets_unchanged=True)
    write_json(folder / 'result.json', result); write_json(folder / 'history.json', history)
    print(f'{folder.name}: dev H={best["H"]:.2f}, step={best_step}, {result["seconds"]:.1f}s', flush=True)
    return result


def train_all():
    c = setup(); p, _ = prepare(); bank = text_bank(); data = load_data('development')
    probe = train_probe(data, bank, c)
    results = [train_cell(v, s, lr, data, p, c, bank, probe) for s in c['seeds'] for v in c['variants'] for lr in c['learning_rates']]
    assert len({r['trainable_parameters'] for r in results}) == 1
    selected = []
    for s in c['seeds']:
        for v in c['variants']:
            r = max([r for r in results if r['config']['seed'] == s and r['config']['variant'] == v], key=lambda r: rank(r['development']))
            selected.append(dict(variant=v, seed=s, lr=r['config']['lr'], step=r['best_step'], development=r['development'], checkpoint=str((OUT / name(v, s, r['config']['lr']) / 'best.pt').relative_to(ROOT))))
    lock = dict(selected=selected, provenance=provenance(), scope=c['scope'], current_run_evaluation_not_used_for_selection=True, prior_evaluation_inspected=True)
    target = OUT / 'selection_locked.json'
    if target.exists(): assert json.loads(target.read_text(encoding='utf-8')) == lock
    else: write_json(target, lock)
    print('All 30 cells complete; 15 selections locked.', flush=True)


def restore(path, device='cuda'):
    c = setup(); saved = torch.load(path, weights_only=True)
    assert saved['config']['provenance'] == provenance()
    assert saved['config']['probe_sha256'] == digest(OUT / 'probe.pt')
    probe = torch.load(OUT / 'probe.pt', weights_only=True)
    model = build(saved['config']['variant'], saved['config']['seed'], text_bank(), probe, c)
    model.load_state_dict(saved['state'])
    return model.to(device).eval(), saved


def final_evaluation():
    c = setup(); p, _ = prepare()
    lock = json.loads((OUT / 'selection_locked.json').read_text(encoding='utf-8'))
    assert lock['provenance'] == provenance()
    data = load_data('evaluation'); bank = text_bank()
    native_logits = 20 * data['native_views'][:, 0] @ bank['class_text'].T
    native, predictions = measures(native_logits, data['labels'], p, 'evaluation')
    torch.save(dict(logits=native_logits, labels=data['labels'], predictions=predictions), OUT / 'native_predictions.pt')
    probe = torch.load(OUT / 'probe.pt', weights_only=True)
    model = build('tokens', 42, bank, probe, c)
    with torch.no_grad(): attributes = attribute_score(data['targets'], model.probe(data['native_views']).sigmoid())
    results = []
    for chosen in lock['selected']:
        model, _ = restore(ROOT / chosen['checkpoint'])
        metrics, pred = evaluate(model, data, p, 'evaluation')
        folder = (ROOT / chosen['checkpoint']).parent
        torch.save(pred, folder / 'final_predictions.pt')
        interventions = {}
        if chosen['variant'] == 'tokens':
            for mode in ['zero_text', 'permuted', 'constant']:
                m, counter = evaluate(model, data, p, 'evaluation', mode)
                interventions[mode] = dict(metrics=m, changed_predictions=int((pred['predictions'] != counter['predictions']).sum()), max_logit_difference=float((pred['logits'] - counter['logits']).abs().max()))
            # Token diversity before attention, across attributes within each view.
            with torch.no_grad():
                token, _, _ = model.make_tokens(data['native_views'].cuda(), data['valid'].cuda())
                token = F.normalize(token.reshape(-1, 3, len(bank['attribute_text']), c['hidden_dim']), dim=-1)
                gram = token @ token.transpose(-1, -2)
                valid = torch.cat([torch.ones(len(token), 1, dtype=torch.bool), data['valid']], 1).cuda()
                a = len(bank['attribute_text'])
                diversity = ((gram.sum((-1, -2)) - a) / (a * (a - 1)))[valid].mean()
            interventions['mean_within_view_token_cosine'] = float(diversity)
        results.append(dict(**chosen, metrics=metrics, interventions=interventions))
        print(f'FINAL {chosen["variant"]} seed{chosen["seed"]}: H={metrics["H"]:.2f}, ZSL={metrics["ZSL"]:.2f}', flush=True)
    write_json(REPORT / 'final_results.json', dict(native=native, attributes=attributes, results=results, provenance=provenance(), selection_sha256=digest(OUT / 'selection_locked.json'), scope=c['scope']))
