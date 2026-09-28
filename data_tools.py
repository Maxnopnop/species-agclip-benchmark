"""Official iNaturalist 2021 metadata, fixed splits and resumable downloads."""
import csv
import hashlib
import json
import random
import shutil
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
BASE_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/'
ARCHIVES = {
    'train_mini.json.tar.gz': '395a35be3651d86dc3b0d365b8ea5f92',
    'val.json.tar.gz': '4d761e0f6a86cc63e8f7afc91f6a8f0b',
    'train_mini.tar.gz': 'db6ed8330e634445efc8fec83ae81442',
    'val.tar.gz': 'f6f6e0e242e3d4c9569ba56400938afc',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or ':' in name:
        raise ValueError(f'Unsafe dataset path: {name}')
    return p.as_posix().removeprefix('./')


def download_serial(name, directory):
    """Resume only range responses at the requested offset; verify official MD5."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    expected = ARCHIVES[name]
    if target.exists():
        if digest(target, 'md5') == expected:
            print(f'Checksum OK: {name}', flush=True)
            return target
        raise RuntimeError(f'Checksum mismatch: {target}. Move this corrupt file aside and retry.')
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'ELEC4240-SpeciesRecognition/1.0', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            req = urllib.request.Request(BASE_URL + name, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response:
                if offset and response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-'):
                        raise RuntimeError(f'Invalid range response: {content_range}')
                    mode = 'ab'
                else:
                    offset, mode = 0, 'wb'
                length = response.headers.get('Content-Length')
                total = offset + int(length) if length else None
                if total and shutil.disk_usage(directory).free < total - offset + 1024**3:
                    raise RuntimeError('Insufficient disk space for this archive plus 1 GiB reserve.')
                count, last = offset, 0.0
                with partial.open(mode) as f:
                    while block := response.read(1024 * 1024):
                        f.write(block)
                        count += len(block)
                        if time.monotonic() - last > 10:
                            suffix = f' / {total / 1024**2:.1f} MiB' if total else ''
                            print(f'{name}: {count / 1024**2:.1f} MiB{suffix}', flush=True)
                            last = time.monotonic()
                if total and count != total:
                    raise OSError(f'Incomplete response: {count}/{total}')
            break
        except (OSError, TimeoutError) as exc:
            if attempt == 4:
                raise
            print(f'Download interrupted ({exc}); retry {attempt + 1}/4', flush=True)
            time.sleep(2 ** attempt)
    if digest(partial, 'md5') != expected:
        raise RuntimeError(f'Checksum mismatch: {partial}. Move it aside before retrying.')
    partial.replace(target)
    return target


def download(name, directory):
    """Eight bounded range transfers, append in order, resume and verify MD5."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    expected = ARCHIVES[name]
    if target.exists():
        if digest(target, 'md5') != expected:
            raise RuntimeError(f'Checksum mismatch: {target}. Move it aside and retry.')
        print(f'Checksum OK: {name}', flush=True)
        return target
    partial = target.with_suffix(target.suffix + '.part')
    req = urllib.request.Request(BASE_URL + name, method='HEAD')
    with urllib.request.urlopen(req, timeout=45) as response:
        total = int(response.headers['Content-Length'])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise RuntimeError(f'Oversized partial archive: {partial}')
    if shutil.disk_usage(directory).free < total - offset + 1024**3:
        raise RuntimeError('Insufficient disk space for archive plus 1 GiB reserve.')
    print(f'{name}: resume at {offset / 1024**2:.1f}/{total / 1024**2:.1f} MiB; 8 connections', flush=True)

    def read_range(start, end):
        for attempt in range(5):
            try:
                request = urllib.request.Request(BASE_URL + name, headers={
                    'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    wanted = f'bytes {start}-{end}/{total}'
                    if response.status != 206 or response.headers.get('Content-Range') != wanted:
                        raise RuntimeError('Server did not honor byte range. Use download_serial instead.')
                    block = response.read(end - start + 1)
                    if len(block) != end - start + 1:
                        raise OSError('Incomplete range response')
                    return block
            except (OSError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    step = 1024 * 1024
    ranges = iter((start, min(start + step, total) - 1) for start in range(offset, total, step))
    # At most eight 1 MiB results are buffered, including a slow leading request.
    with ThreadPoolExecutor(max_workers=8) as pool, partial.open('ab') as out:
        pending = []
        for start, end in ranges:
            pending.append(pool.submit(read_range, start, end))
            if len(pending) == 8:
                break
        last = 0.0
        while pending:
            block = pending.pop(0).result()
            out.write(block)
            out.flush()
            offset += len(block)
            if time.monotonic() - last > 10:
                print(f'{name}: {offset / 1024**2:.1f}/{total / 1024**2:.1f} MiB', flush=True)
                last = time.monotonic()
            next_range = next(ranges, None)
            if next_range is not None:
                pending.append(pool.submit(read_range, *next_range))
    if partial.stat().st_size != total or digest(partial, 'md5') != expected:
        raise RuntimeError(f'Checksum mismatch: {partial}. Move it aside before retrying.')
    partial.replace(target)
    return target


def metadata(name):
    path = ROOT / 'data' / 'metadata' / f'{name}.json'
    if not path.exists():
        archive = download(f'{name}.json.tar.gz', ROOT / 'data' / 'archives')
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, 'r:gz') as tar:
            members = [m for m in tar if m.isfile() and PurePosixPath(m.name).name == path.name]
            if len(members) != 1:
                raise ValueError(f'Expected one {path.name} in archive')
            with tar.extractfile(members[0]) as src, path.with_suffix('.tmp').open('wb') as out:
                shutil.copyfileobj(src, out)
        path.with_suffix('.tmp').replace(path)
    return json.loads(path.read_text(encoding='utf-8'))


def select_species(categories, seed=42):
    """Five biological groups, ten randomly chosen genus pairs per group."""
    predicates = {
        'Plants': lambda c: c.get('kingdom') == 'Plantae',
        'Insects': lambda c: c.get('class') == 'Insecta',
        'Birds': lambda c: c.get('class') == 'Aves',
        'Mammals': lambda c: c.get('class') == 'Mammalia',
        'Fungi': lambda c: c.get('kingdom') == 'Fungi',
    }
    rng = random.Random(seed)
    chosen = []
    for group, predicate in predicates.items():
        genera = defaultdict(list)
        for c in categories:
            if predicate(c) and c.get('genus'):
                genera[(c.get('family'), c['genus'])].append(c)
        eligible = sorted(g for g, members in genera.items() if len(members) >= 2)
        if len(eligible) < 10:
            raise ValueError(f'{group} has only {len(eligible)} eligible genera; review selection.')
        for genus in rng.sample(eligible, 10):
            for c in rng.sample(sorted(genera[genus], key=lambda c: c['id']), 2):
                chosen.append(dict(c, study_group=group))
    return [dict(c, label=i) for i, c in enumerate(sorted(chosen, key=lambda c: c['id']))]


def image_records(meta, classes):
    labels = {c['id']: c['label'] for c in classes}
    annotations = {a['image_id']: a['category_id'] for a in meta['annotations']}
    result = defaultdict(list)
    for im in meta['images']:
        category = annotations[im['id']]
        if category in labels:
            result[labels[category]].append({
                'path': safe_relative(im['file_name']), 'label': labels[category],
                'category_id': category, 'image_id': im['id'],
                'license': im.get('license'), 'rights_holder': im.get('rights_holder'),
            })
    return result


def validate_manifest(manifest):
    classes, splits = manifest['classes'], manifest['splits']
    n = len(classes)
    if [c['label'] for c in classes] != list(range(n)):
        raise ValueError('Labels must be consecutive in class order.')
    seen = set()
    for split in ('train', 'val', 'test'):
        counts = Counter()
        for row in splits[split]:
            safe_relative(row['path'])
            if row['path'] in seen:
                raise ValueError(f'Duplicate/leaked image path: {row["path"]}')
            seen.add(row['path'])
            if not 0 <= row['label'] < n:
                raise ValueError('Label outside class range')
            counts[row['label']] += 1
        if set(counts) != set(range(n)):
            raise ValueError(f'Split {split} must contain every selected class.')


def prepare():
    manifest_path = ROOT / 'data' / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        validate_manifest(manifest)
        print('Existing fixed manifest retained:', manifest_path, flush=True)
        return manifest
    train_meta, test_meta = metadata('train_mini'), metadata('val')
    classes = select_species(train_meta['categories'])
    training, testing = image_records(train_meta, classes), image_records(test_meta, classes)
    splits = {'train': [], 'val': [], 'test': []}
    for c in classes:
        label = c['label']
        rows = sorted(training[label], key=lambda r: r['path'])
        if len(rows) != 50 or len(testing[label]) != 10:
            raise ValueError(f'Unexpected official split counts for {c["name"]}')
        random.Random(42 + label).shuffle(rows)
        splits['val'].extend(rows[:10])
        splits['train'].extend(rows[10:])
        splits['test'].extend(sorted(testing[label], key=lambda r: r['path']))
    manifest = {
        'dataset': 'iNaturalist 2021 mini 100-species subset', 'synthetic': False,
        'seed': 42, 'image_root': str(ROOT / 'data' / 'images'),
        'selection': '20 species each in Plants, Insects, Birds, Mammals, Fungi; 10 genus pairs per group',
        'protocol': '40 train and 10 internal validation images per class from train_mini; official val is held-out test',
        'source': 'https://github.com/visipedia/inat_comp/tree/master/2021',
        'licenses': train_meta.get('licenses', []), 'classes': classes, 'splits': splits,
    }
    validate_manifest(manifest)
    write_json(manifest_path, manifest)
    fields = ['label', 'id', 'name', 'common_name', 'study_group', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    with (ROOT / 'data' / 'species.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(classes)
    print('Fixed manifest saved:', {s: len(rows) for s, rows in splits.items()}, flush=True)
    return manifest


def fetch_images():
    manifest = prepare()
    wanted = {r['path'] for rows in manifest['splits'].values() for r in rows}
    image_root = Path(manifest['image_root'])
    image_root.mkdir(parents=True, exist_ok=True)
    # Official gzip archives require downloading whole archives even for a subset.
    for name in ('train_mini.tar.gz', 'val.tar.gz'):
        archive = download(name, ROOT / 'data' / 'archives')
        print(f'Extracting selected species from {name} ...', flush=True)
        count = 0
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                relative = safe_relative(member.name)
                if relative not in wanted:
                    continue
                dest = image_root / relative
                if dest.exists() and dest.stat().st_size == member.size:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.with_suffix('.tmp').open('wb') as out:
                    shutil.copyfileobj(src, out)
                dest.with_suffix('.tmp').replace(dest)
                count += 1
                if count % 500 == 0:
                    print(f'Extracted {count} selected images', flush=True)
        print(f'Extracted {count} images from {name}', flush=True)
    missing = [p for p in wanted if not (image_root / p).is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} selected images missing; first: {missing[0]}')
    print('All 6,000 selected images are ready.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'download'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else fetch_images()
