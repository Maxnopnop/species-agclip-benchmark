"""Development-led gate/calibration tests and attribute sufficiency diagnostics."""
import json, math
import numpy as np
import torch
from torch.nn import functional as F
from data_tools import ROOT, write_json, digest
from multimodal import cub_token_experiment as exp
from multimodal.grounded_experiment import measures, rank


def calibration(logits, labels, p, stage, gamma):
    adjusted = logits.clone(); adjusted[:, p['seen_classes']] -= gamma
    return measures(adjusted, labels, p, stage)[0]


def main():
    exp.setup(); p, _ = exp.prepare(); data = exp.load_data('development'); bank = exp.text_bank()
    ids = [i for i, r in enumerate(data['rows']) if r['role'] != 'train_seen']
    labels = data['labels'][ids]; native_logits = 20 * data['native_views'][ids, 0] @ bank['class_text'].T
    lock = json.loads((exp.OUT / 'selection_locked.json').read_text()); records = []
    native_calibration = [dict(gamma=g, metrics=calibration(native_logits, labels, p, 'development', g)) for g in [0., .25, .5, 1., 2., 4.]]
    chosen_native = max(native_calibration, key=lambda r: rank(r['metrics']))
    for chosen in lock['selected']:
        if chosen['variant'] != 'tokens': continue
        model, saved = exp.restore(ROOT / chosen['checkpoint'])
        original = float(model.max_residual * model.gate.tanh()); options = []
        # Same learned network, stronger signed residual, selected on development.
        for magnitude in [0., abs(original), .05, .1, .15, .199]:
            signed = math.copysign(magnitude, original)
            with torch.no_grad(): model.gate.fill_(math.atanh(signed / model.max_residual))
            metrics, prediction = exp.evaluate(model, data, p, 'development')
            options.append(dict(gate=signed, metrics=metrics))
        selected = max(options, key=lambda r: rank(r['metrics']))
        model, _ = exp.restore(ROOT / chosen['checkpoint'])
        metrics, prediction = exp.evaluate(model, data, p, 'development')
        calibrated = [dict(gamma=g, metrics=calibration(prediction['logits'], labels, p, 'development', g)) for g in [0., .25, .5, 1., 2., 4.]]
        records.append(dict(seed=chosen['seed'], checkpoint=chosen['checkpoint'], original_gate=original, gate_grid=options, selected_gate=selected, calibration_grid=calibrated, selected_calibration=max(calibrated, key=lambda r: rank(r['metrics']))))
    selection = dict(native_calibration_grid=native_calibration, selected_native_calibration=chosen_native, records=records,
                     scope='Posthoc development-selected diagnostics on reused data, not a new confirmatory test.', source_sha256=digest(__file__))
    write_json(exp.OUT / 'diagnostic_selection_locked.json', selection)
    # Attribute sufficiency within ten seen species. Never used to tune the AG models.
    ti = torch.tensor([i for i, r in enumerate(data['rows']) if r['role'] == 'train_seen'])
    vi = torch.tensor([i for i, r in enumerate(data['rows']) if r['role'] == 'dev_seen'])
    targets = data['targets'][:, 0]; known = targets >= 0
    prototypes = []
    for cls in p['seen_classes']:
        ix = ti[data['labels'][ti] == cls]; counts = known[ix].sum(0)
        prototypes.append(((targets[ix].clamp_min(0).sum(0) + 1) / (counts + 2)).clamp(.01, .99))
    prototypes = torch.stack(prototypes)
    model, _ = exp.restore(ROOT / records[0]['checkpoint'], 'cpu')
    with torch.no_grad(): predicted = model.probe(data['native_views'][vi, 0]).sigmoid()
    candidates = torch.tensor(p['seen_classes'])
    def attr_acc(prob, mask):
        scores = (prob[:, None] * prototypes.log()[None] + (1 - prob[:, None]) * (1 - prototypes).log()[None]) * mask[:, None]
        pred = candidates[scores.sum(-1).argmax(-1)]
        return float((pred == data['labels'][vi]).float().mean() * 100)
    oracle = attr_acc(targets[vi].clamp_min(0), known[vi])
    predicted_acc = attr_acc(predicted, torch.ones_like(predicted))
    pred_matched = attr_acc(predicted, known[vi])
    native_seen = candidates[(20 * data['native_views'][vi, 0] @ bank['class_text'][candidates].T).argmax(-1)]
    sufficiency = dict(development_seen_images=len(vi), attributes=len(bank['attribute_text']), native_seen_accuracy=float((native_seen == data['labels'][vi]).float().mean() * 100), predicted_attribute_prototype_accuracy=predicted_acc, oracle_visibility_matched_predicted_accuracy=pred_matched, oracle_true_attribute_accuracy=oracle,
                       warning='Oracle values use true development image attributes/visibility as diagnostic inputs, never deployed or counted as model results. Simple naive-Bayes prototypes are not an information-theoretic upper bound. All class prototypes use training-seen images only.')
    # Only now inspect evaluation for the already locked adjustments.
    final = exp.load_data('evaluation'); final_labels = final['labels']
    native_final = 20 * final['native_views'][:, 0] @ bank['class_text'].T
    result = dict(selection=selection, attribute_sufficiency=sufficiency, native_calibrated=calibration(native_final, final_labels, p, 'evaluation', chosen_native['gamma']), tokens=[])
    for r in records:
        model, _ = exp.restore(ROOT / r['checkpoint'])
        with torch.no_grad(): model.gate.fill_(math.atanh(r['selected_gate']['gate'] / model.max_residual))
        gate_metrics, _ = exp.evaluate(model, final, p, 'evaluation')
        model, _ = exp.restore(ROOT / r['checkpoint']); _, pred = exp.evaluate(model, final, p, 'evaluation')
        calibrated = calibration(pred['logits'], final_labels, p, 'evaluation', r['selected_calibration']['gamma'])
        result['tokens'].append(dict(seed=r['seed'], selected_gate=r['selected_gate']['gate'], gate_adjusted=gate_metrics, gamma=r['selected_calibration']['gamma'], calibrated=calibrated))
    write_json(exp.REPORT / 'diagnosis.json', result)
    print(json.dumps(dict(attribute_sufficiency=sufficiency, native_gamma=chosen_native['gamma'], native_calibrated=result['native_calibrated'], tokens=result['tokens']), indent=2))


if __name__ == '__main__': main()
