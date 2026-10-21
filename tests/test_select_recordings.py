"""Regression tests for clipping-aware quota redistribution."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from dataset import select_recordings as selector


class TestSelection(unittest.TestCase):
    def run_selection(self, root, rejected):
        rows = []
        for site in (1, 2, 3):
            for n in range(4):
                audio_id = site * 10 + n
                rows.append(dict(Id=audio_id, Url=str(root / str(audio_id)),
                                 CaptureSiteId=site, SiteName=str(site), station_id=site,
                                 audio_datetime_local='2026-01-01 00:00:00',
                                 hour_of_day=0, actual_duration_seconds=60))
        args = SimpleNamespace(seed=42, metadata=root / 'unused.csv', count=6,
                               audio_root=None, source_prefix=Path('/'),
                               full_run_limit=3, near_fraction_limit=0.01,
                               out=root / 'manifest.csv', audit_out=root / 'audit.csv',
                               overwrite=True)

        def measure(path, offset):
            return {'longest_full_scale_run': 3 if int(path.name) in rejected else 0,
                    'fraction_ge_98pct_fs': 0}

        with patch.object(selector, 'load_inventory', return_value=pd.DataFrame(rows)), \
                patch.object(Path, 'is_file', return_value=True), \
                patch.object(selector, 'measure_clipping', side_effect=measure) as screen, \
                contextlib.redirect_stdout(io.StringIO()):
            selector.select(args)
        return pd.read_csv(args.out), pd.read_csv(args.audit_out), screen.call_count

    def test_redistributes_exhausted_group_and_is_reproducible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, audit, calls = self.run_selection(root, {10, 11, 12, 13})
            self.assertEqual(len(manifest), 6)
            self.assertEqual(set(manifest.site_id), {2, 3})
            self.assertEqual(manifest.groupby('site_id').size().tolist(), [3, 3])
            self.assertTrue(audit.audio_id.is_unique)
            self.assertEqual(calls, len(audit))
            self.assertEqual(int(audit.clipping_flag.sum()), 4)
            repeated, repeated_audit, _ = self.run_selection(root, {10, 11, 12, 13})
            pd.testing.assert_frame_equal(manifest, repeated)
            pd.testing.assert_frame_equal(audit, repeated_audit)

    def test_reports_global_shortage(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'needed 6, found 4'):
                self.run_selection(Path(directory), set(range(10, 14)) | set(range(20, 24)))

    def test_balanced_when_no_rejections(self):
        with TemporaryDirectory() as directory:
            manifest, audit, calls = self.run_selection(Path(directory), set())
            self.assertEqual(manifest.groupby('site_id').size().tolist(), [2, 2, 2])
            self.assertEqual(calls, 6)
            self.assertFalse(audit.clipping_flag.any())


if __name__ == '__main__':
    unittest.main()
