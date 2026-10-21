"""Run from repository root: python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from model.features import AcousticFeatures, CONFIG, StreamingFeatures


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = AcousticFeatures()

    def assert_stream_equal(self, signal, chunk_size):
        expected_lm, expected_mfcc = self.extractor.extract(signal)
        stream = StreamingFeatures(self.extractor)
        lm_chunks, mfcc_chunks = [], []
        for start in range(0, len(signal), chunk_size):
            lm, mfcc = stream.push(signal[start:start + chunk_size])
            lm_chunks.append(lm)
            mfcc_chunks.append(mfcc)
        stream.finish()
        got_lm = np.concatenate(lm_chunks, axis=0)
        got_mfcc = np.concatenate(mfcc_chunks, axis=0)
        self.assertEqual(expected_lm.shape, (299, 40))
        self.assertEqual(expected_mfcc.shape, (299, 13))
        np.testing.assert_allclose(got_lm, expected_lm, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got_mfcc, expected_mfcc, rtol=1e-5, atol=1e-5)

    def test_streaming_20ms_dma(self):
        rng = np.random.default_rng(42)
        self.assert_stream_equal(rng.standard_normal(CONFIG.samples_per_segment).astype(np.float32) * 0.01,
                                 CONFIG.hop_length)

    def test_streaming_variable_blocks(self):
        t = np.arange(CONFIG.samples_per_segment) / CONFIG.sample_rate
        signal = (0.1 * np.sin(2 * np.pi * 3_000 * t)).astype(np.float32)
        self.assert_stream_equal(signal, 777)

    def test_silence(self):
        zeros = np.zeros(CONFIG.samples_per_segment, dtype=np.float32)
        logmel, mfcc = self.extractor.extract(zeros)
        self.assertTrue(np.isfinite(logmel).all())
        self.assertTrue(np.isfinite(mfcc).all())
        self.assert_stream_equal(zeros, CONFIG.hop_length)

    def test_frame_counts_and_boundary(self):
        stream = StreamingFeatures(self.extractor)
        with self.assertRaises(ValueError):
            stream.finish()
        with self.assertRaises(ValueError):
            self.extractor.extract(np.zeros(CONFIG.samples_per_segment - 1))
        with self.assertRaises(ValueError):
            stream.push(np.zeros(CONFIG.samples_per_segment + 1))


if __name__ == '__main__':
    unittest.main()
