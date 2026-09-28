"""Checks for class selection, data leakage, paths and classification metrics."""
import copy
import unittest
import hashlib
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import numpy as np
import data_tools
from data_tools import safe_relative, select_species, validate_manifest
from run import metrics_from_confusion


class ProtocolTests(unittest.TestCase):
    def test_metrics_include_missed_class(self):
        m = metrics_from_confusion(np.array([[2, 0], [1, 0]]))
        self.assertAlmostEqual(m['top1_accuracy'], 2/3)
        self.assertAlmostEqual(m['macro_f1'], .4)

    def test_species_selection_is_balanced_and_reproducible(self):
        categories = []
        for group, (kingdom, cl) in enumerate([
            ('Plantae', 'Magnoliopsida'), ('Animalia', 'Insecta'), ('Animalia', 'Aves'),
            ('Animalia', 'Mammalia'), ('Fungi', 'Agaricomycetes')]):
            for genus in range(12):
                for species in range(3):
                    categories.append({'id': len(categories), 'kingdom': kingdom, 'class': cl,
                                       'family': f'family_{group}', 'genus': f'g{group}_{genus}',
                                       'name': f'g{group}_{genus} s{species}'})
        a, b = select_species(categories), select_species(list(reversed(categories)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)
        self.assertEqual(len({c['id'] for c in a}), 100)
        for group in ['Plants', 'Insects', 'Birds', 'Mammals', 'Fungi']:
            self.assertEqual(sum(c['study_group'] == group for c in a), 20)

    def test_rejects_data_leakage(self):
        manifest = {'classes': [{'label': 0}], 'splits': {
            s: [{'path': f'{s}/x.jpg', 'label': 0}] for s in ['train', 'val', 'test']}}
        validate_manifest(manifest)
        bad = copy.deepcopy(manifest)
        bad['splits']['test'][0]['path'] = 'train/x.jpg'
        with self.assertRaisesRegex(ValueError, 'leaked'):
            validate_manifest(bad)

    def test_rejects_bad_label(self):
        manifest = {'classes': [{'label': 1}], 'splits': {}}
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            validate_manifest(manifest)

    def test_rejects_archive_path_escape(self):
        for path in ['../outside.jpg', '/absolute.jpg', 'C:/outside.jpg', '..\\outside.jpg']:
            with self.assertRaises(ValueError):
                safe_relative(path)
        self.assertEqual(safe_relative('./val/species/a.jpg'), 'val/species/a.jpg')

    def test_parallel_download_resumes_without_duplicating_bytes(self):
        payload = bytes(range(256)) * 13000
        requested = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()

            def do_GET(self):
                start, end = map(int, self.headers['Range'][6:].split('-'))
                requested.append((start, end))
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
                self.send_header('Content-Length', str(end - start + 1))
                self.end_headers()
                self.wfile.write(payload[start:end+1])

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            (data_tools.ROOT/'work').mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=data_tools.ROOT/'work') as folder:
                (Path(folder)/'fixture.tar.gz.part').write_bytes(payload[:12345])
                with patch.object(data_tools, 'BASE_URL', f'http://127.0.0.1:{server.server_port}/'), \
                     patch.dict(data_tools.ARCHIVES, {'fixture.tar.gz': hashlib.md5(payload).hexdigest()}):
                    result = data_tools.download('fixture.tar.gz', folder)
                    self.assertEqual(result.read_bytes(), payload)
                    self.assertEqual(min(s for s, _ in requested), 12345)
                    self.assertFalse((Path(folder)/'fixture.tar.gz.part').exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
