"""Image-only EfficientNet-B0 training, evaluation and single-image inference."""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from data_tools import ROOT, digest, validate_manifest, write_json

os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache' / 'torch'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    # Stable seeds and fixed splits, without claiming bitwise CUDA reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run doctor or choose --device cpu.')
    return torch.device(name)


def new_model(classes, pretrained=True):
    model = efficientnet_b0(weights=WEIGHTS if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    return model


def load_manifest(path, require_images=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(data)
    if require_images:
        missing = [r['path'] for rows in data['splits'].values() for r in rows
                   if not (Path(data['image_root']) / r['path']).is_file()]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} images missing. Run download_data.cmd first. Example: {missing[0]}')
    return data


class SpeciesDataset(Dataset):
    def __init__(self, manifest, split, samples_per_class=None, seed=42):
        self.root = Path(manifest['image_root'])
        self.rows = manifest['splits'][split]
        if samples_per_class is not None:
            selected = []
            for label in range(len(manifest['classes'])):
                rows = sorted((r for r in self.rows if r['label'] == label), key=lambda r: r['path'])
                random.Random(seed + label).shuffle(rows)
                if not 1 <= samples_per_class <= len(rows):
                    raise ValueError(f'samples-per-class must be 1..{len(rows)} for class {label}')
                selected.extend(rows[:samples_per_class])
            self.rows = selected
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]) if split == 'train' else WEIGHTS.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as im:
            image = self.transform(im.convert('RGB'))
        return image, row['label'], row['path']


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, workers, device, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=workers > 0)


def metrics_from_confusion(confusion):
    c = confusion.astype(np.float64)
    tp = c.diagonal()
    denominator = c.sum(0) + c.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    return {'top1_accuracy': float(tp.sum() / max(c.sum(), 1)),
            'macro_f1': float(f1.mean())}


def run_epoch(model, loader, device, classes, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, count, top5 = 0.0, 0, 0
    confusion = np.zeros((classes, classes), dtype=np.int64)
    predictions = []
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels, paths) in enumerate(loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; lower learning rate or run without AMP.')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            scores = logits.detach().float().softmax(1)
            guesses = scores.argmax(1)
            batch_size = len(labels)
            total_loss += float(loss.detach()) * batch_size
            count += batch_size
            top5 += int((scores.topk(min(5, classes), dim=1).indices == labels[:, None]).any(1).sum())
            truth, pred = labels.cpu().numpy(), guesses.cpu().numpy()
            np.add.at(confusion, (truth, pred), 1)
            if not training:
                confidence = scores.max(1).values.cpu().tolist()
                predictions.extend({'path': path, 'true_label': int(y), 'predicted_label': int(p), 'score': float(s)}
                                   for path, y, p, s in zip(paths, truth, pred, confidence))
            if training and (batch_index + 1) % 25 == 0:
                print(f'  batch {batch_index + 1}/{len(loader)} loss={total_loss/count:.4f}', flush=True)
    metrics = dict(metrics_from_confusion(confusion), loss=total_loss / count,
                   top5_accuracy=top5 / count, images=count)
    return metrics, confusion, predictions


def save_checkpoint(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def train(args):
    seed_all(args.seed)
    device = device_for(args.device)
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    classes = len(manifest['classes'])
    run_dir = ROOT / 'runs' / (args.name or f'baseline_seed{args.seed}_{timestamp()}')
    if run_dir.exists():
        raise FileExistsError(f'Run directory exists; use a new --name: {run_dir}')
    run_dir.mkdir(parents=True)
    config = vars(args).copy()
    config.update(architecture='efficientnet_b0', manifest_sha256=digest(manifest_path),
                  synthetic=manifest.get('synthetic', False), torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__), python=sys.version,
                  device=str(device), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'classes.json', manifest['classes'])
    training = SpeciesDataset(manifest, 'train', args.samples_per_class, args.seed)
    validation = SpeciesDataset(manifest, 'val')
    train_loader = make_loader(training, args.batch_size, args.workers, device, True, args.seed)
    val_loader = make_loader(validation, args.batch_size, args.workers, device)
    # Save exact selected rows for low-data experiments.
    write_json(run_dir / 'training_rows.json', training.rows)
    model = new_model(classes, not args.no_pretrained).to(device)
    classifier_ids = {id(p) for p in model.classifier.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in classifier_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.head_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, bad_epochs, history = -1.0, 0, []
    print(f'Run: {run_dir}\nDevice: {device}; train={len(training)} val={len(validation)} classes={classes}', flush=True)
    if config['synthetic']:
        print('SYNTHETIC SMOKE TEST: metrics do not measure species recognition.', flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        train_metrics, _, _ = run_epoch(model, train_loader, device, classes, optimizer, scaler, amp)
        val_metrics, _, _ = run_epoch(model, val_loader, device, classes, amp=amp)
        scheduler.step()
        history.append({'epoch': epoch, 'seconds': time.monotonic()-start,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()}})
        with (run_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        score = val_metrics['macro_f1']
        if score > best:
            best, bad_epochs = score, 0
            save_checkpoint(run_dir / 'best.pt', {
                'architecture': 'efficientnet_b0', 'state_dict': model.state_dict(),
                'classes': manifest['classes'], 'manifest_sha256': config['manifest_sha256'],
                'config': config, 'epoch': epoch, 'val_metrics': val_metrics,
            })
        else:
            bad_epochs += 1
        print(f'Epoch {epoch}/{args.epochs}: train acc={train_metrics["top1_accuracy"]:.3f}, '
              f'val acc={val_metrics["top1_accuracy"]:.3f}, val F1={score:.3f}, '
              f'{history[-1]["seconds"]:.1f}s', flush=True)
        if bad_epochs >= args.patience:
            print('Early stopping using internal validation only.', flush=True)
            break
    plot_history(history, run_dir / 'learning_curves.png')
    write_json(run_dir / 'summary.json', {'status': 'complete', 'epochs_completed': len(history),
                                         'best_val_macro_f1': best, 'synthetic': config['synthetic'],
                                         'test_evaluated': False})
    print(f'Saved best model: {run_dir / "best.pt"}', flush=True)
    return run_dir


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('loss', 'top1_accuracy')):
        for split in ('train', 'val'):
            ax.plot([h['epoch'] for h in history], [h[f'{split}_{metric}'] for h in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def restore(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['architecture'] != 'efficientnet_b0':
        raise ValueError('Unsupported model architecture')
    model = new_model(len(saved['classes']), pretrained=False)
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved


def evaluate(args):
    seed_all(42)
    device = device_for(args.device)
    model, saved = restore(args.checkpoint, device)
    manifest = load_manifest(args.manifest)
    if saved['manifest_sha256'] != digest(args.manifest) or saved['classes'] != manifest['classes']:
        raise ValueError('Checkpoint and manifest differ; refusing a misaligned evaluation.')
    dataset = SpeciesDataset(manifest, args.split)
    loader = make_loader(dataset, args.batch_size, args.workers, device)
    metrics, confusion, predictions = run_epoch(model, loader, device, len(manifest['classes']))
    output = Path(args.checkpoint).resolve().parent / f'evaluation_{args.split}_{timestamp()}'
    output.mkdir(parents=True)
    metrics.update(split=args.split, synthetic=manifest.get('synthetic', False),
                   checkpoint_sha256=digest(args.checkpoint), manifest_sha256=digest(args.manifest))
    write_json(output / 'metrics.json', metrics)
    names = [c['name'] for c in manifest['classes']]
    for row in predictions:
        row['true_species'] = names[row['true_label']]
        row['predicted_species'] = names[row['predicted_label']]
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    np.savetxt(output / 'confusion_matrix.csv', confusion, fmt='%d', delimiter=',')
    per_class = []
    for i, c in enumerate(manifest['classes']):
        tp, support, predicted = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        per_class.append({'label': i, 'species': c['name'], 'support': support,
                          'precision': tp/max(predicted, 1), 'recall': tp/max(support, 1),
                          'f1': 2*tp/max(support+predicted, 1)})
    write_json(output / 'per_class_metrics.json', per_class)
    plot_evaluation(confusion, predictions, manifest, output)
    print(json.dumps(metrics, indent=2), flush=True)
    print('Evaluation saved:', output, flush=True)
    return output


def plot_evaluation(confusion, predictions, manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    normalized = confusion / np.maximum(confusion.sum(1, keepdims=True), 1)
    heat = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    ax.set(xlabel='Predicted class index', ylabel='True class index', title='Row-normalized confusion matrix')
    fig.colorbar(heat, ax=ax)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=170)
    plt.close(fig)
    correct = [r for r in predictions if r['true_label'] == r['predicted_label']][:4]
    errors = [r for r in predictions if r['true_label'] != r['predicted_label']][:4]
    examples = correct + errors
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, examples):
        with Image.open(Path(manifest['image_root']) / row['path']) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(f'True: {row["true_species"]}\nPred: {row["predicted_species"]}', fontsize=8)
    fig.suptitle('Prediction examples' + (' - SYNTHETIC TEST ONLY' if manifest.get('synthetic') else ''))
    fig.tight_layout()
    fig.savefig(output / 'prediction_examples.png', dpi=130)
    plt.close(fig)


def predict(args):
    device = device_for(args.device)
    model, saved = restore(args.checkpoint, device)
    with Image.open(args.image) as im:
        tensor = WEIGHTS.transforms()(im.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores, indices = model(tensor).float().softmax(1)[0].topk(min(args.topk, len(saved['classes'])))
    result = {'image': str(Path(args.image).resolve()), 'synthetic_checkpoint': saved['config']['synthetic'],
              'note': 'Closed-set prediction among selected species; softmax scores are not calibrated probabilities.',
              'predictions': [dict(species=saved['classes'][i]['name'],
                                   common_name=saved['classes'][i].get('common_name'), score=s)
                              for s, i in zip(scores.cpu().tolist(), indices.cpu().tolist())]}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def doctor():
    seed_all(42)
    device = device_for('auto')
    model = new_model(100).to(device).train()
    x = torch.randn(2, 3, 224, 224, device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 99], device=device))
    loss.backward()
    assert logits.shape == (2, 100) and bool(torch.isfinite(loss))
    report = {'python': sys.executable, 'python_version': platform.python_version(),
              'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
              'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'pretrained_weights_cached': True, 'forward_backward_100_classes': 'passed',
              'scope': 'Environment check only; no species accuracy measurement.'}
    write_json(ROOT / 'deployment_check.json', report)
    print(json.dumps(report, indent=2), flush=True)


def smoke(args):
    """Real train/save/reload/evaluate/predict code with clearly labeled artificial data."""
    base = ROOT / 'work' / f'smoke_{timestamp()}'
    base.mkdir(parents=True)
    rng = np.random.default_rng(42)
    classes = [{'label': i, 'id': i, 'name': f'synthetic_class_{i:03d}'} for i in range(100)]
    splits = {s: [] for s in ('train', 'val', 'test')}
    for split in splits:
        for i in range(100):
            name = f'{split}_{i:03d}.jpg'
            Image.fromarray(rng.integers(0, 256, (240, 240, 3), dtype=np.uint8)).save(base / name)
            splits[split].append({'path': name, 'label': i})
    manifest_path = base / 'manifest.json'
    write_json(manifest_path, {'synthetic': True, 'image_root': str(base), 'classes': classes, 'splits': splits})
    train_args = argparse.Namespace(command='train', manifest=str(manifest_path), device=args.device,
        seed=42, name=f'smoke_{timestamp()}', epochs=1, batch_size=8, workers=args.workers,
        samples_per_class=None, lr=1e-4, head_lr=1e-3, patience=5, no_amp=False, no_pretrained=False)
    run_dir = train(train_args)
    output = evaluate(argparse.Namespace(checkpoint=str(run_dir / 'best.pt'), manifest=str(manifest_path),
                         device=args.device, split='test', batch_size=8, workers=args.workers))
    result = predict(argparse.Namespace(checkpoint=str(run_dir / 'best.pt'), image=str(base/'test_000.jpg'),
                         device=args.device, topk=5, output=str(output/'single_prediction.json')))
    assert result['synthetic_checkpoint'] and len(result['predictions']) == 5
    write_json(ROOT / 'smoke_test_result.json', {'status': 'passed', 'synthetic': True,
        'checks': ['100-class fine-tuning', 'best checkpoint save/reload', 'evaluation metrics',
                   'confusion matrix and learning curves', 'single-image top-5 prediction'],
        'run': str(run_dir), 'evaluation': str(output), 'real_species_accuracy': None})
    print('SMOKE TEST PASSED. Synthetic data only; no real species accuracy reported.', flush=True)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    p = sub.add_parser('smoke')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--workers', type=int, default=0)
    p = sub.add_parser('train')
    p.add_argument('--manifest', default=str(ROOT/'data'/'manifest.json'))
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--name')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--samples-per-class', type=int)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-pretrained', action='store_true')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', default=str(ROOT/'data'/'manifest.json'))
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--output')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    for key in ('epochs', 'batch_size', 'patience', 'topk'):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if hasattr(args, 'workers') and args.workers < 0:
        parser.error('workers must be nonnegative')
    if args.command == 'doctor':
        doctor()
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
