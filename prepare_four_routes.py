"""Fetch pinned research inputs. Weights are optional; no training is launched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/four_routes_v1'
WORK = ROOT / 'work/four_routes_review_20260929'


def verified_model_file(path, record):
    if not path.is_file() or path.stat().st_size != record['size']:
        return False
    if 'lfs' in record:
        h = hashlib.sha256()
        expected = record['lfs']['sha256']
    else:
        h = hashlib.sha1()
        h.update(f"blob {record['size']}\0".encode())
        expected = record['blobId']
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest() == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fgclip2', action='store_true', help='Also download ~1.5 GB of pinned model weights.')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    records = json.loads((REPORT / 'sources.json').read_text())
    for record in records:
        target = WORK / record['repo'].split('/')[-1] / record['file']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']:
            continue
        response = requests.get(record['url'], timeout=(20, 120))
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == record['sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
    (WORK / 'sources.json').write_bytes((REPORT / 'sources.json').read_bytes())
    (WORK / 'fgclip2_hf_metadata.json').write_bytes((REPORT / 'fgclip2_hf_metadata.json').read_bytes())
    if args.fgclip2:
        required = ['config.json', 'configuration_fgclip2.py', 'modeling_fgclip2.py',
                    'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                    'tokenizer.json', 'tokenizer.model', 'README.md', 'model.safetensors']
        metadata = json.loads((REPORT / 'fgclip2_hf_metadata.json').read_text())
        records = {r['rfilename']: r for r in metadata['siblings']}
        model_dir = ROOT / 'cache/four_routes_v1/models/fgclip2_base'
        if all(verified_model_file(model_dir / name, records[name]) for name in required):
            print('Pinned research inputs and model verified locally; no network needed.')
            return
        os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
        from huggingface_hub import snapshot_download
        snapshot_download('qihoo360/fg-clip2-base', revision='430fbc8a912c86fd4de601381b6245a0edab22f0',
                          local_dir=model_dir, max_workers=2, allow_patterns=required)
        assert all(verified_model_file(model_dir / name, records[name]) for name in required)
    print('Pinned research inputs ready.')


if __name__ == '__main__':
    main()
