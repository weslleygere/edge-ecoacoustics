"""Exercise CLAP WAV decoding without loading model weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from dataset.generate_labels import SoundFileCLAP


class TestCLAPAudioLoader(unittest.TestCase):
    def setUp(self):
        self.model = SoundFileCLAP.__new__(SoundFileCLAP)
        self.model.args = SimpleNamespace(sampling_rate=48000)

    def test_pcm16_mono_without_torchcodec(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'segment.wav'
            signal = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)
            sf.write(path, signal, 48000, subtype='PCM_16')
            expected, _ = sf.read(path, dtype='float32')
            with patch('torchaudio.load', side_effect=AssertionError('TorchCodec path used')):
                audio, rate = self.model.read_audio(path)
            self.assertEqual(rate, 48000)
            np.testing.assert_array_equal(audio.numpy(), expected[None, :])
            # Also exercise upstream repetition/trimming used for embeddings.
            processed = self.model.load_audio_into_tensor(path, 2)
            self.assertEqual(tuple(processed.shape), (96000,))
            np.testing.assert_array_equal(processed.numpy(), np.tile(expected, 2))

    def test_resampling_and_channel_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'stereo.wav'
            signal = np.column_stack([np.zeros(24000), np.full(24000, 0.25)])
            sf.write(path, signal, 24000, subtype='PCM_16')
            native, rate = self.model.read_audio(path, resample=False)
            self.assertEqual(rate, 24000)
            self.assertEqual(tuple(native.shape), (2, 24000))
            audio, rate = self.model.read_audio(path, resample=True)
            self.assertEqual(rate, 48000)
            self.assertEqual(tuple(audio.shape), (2, 48000))
            self.assertTrue((audio[0] == 0).all())
            self.assertAlmostEqual(float(audio[1, 100]), 0.25, places=3)


if __name__ == '__main__':
    unittest.main()
