"""Prepare 48-kHz MFCC/Log-Mel student-model inputs from the committed CEI labels.

Usage from the repository root:
    python model/prepare_data.py --limit 10 --out data/processed_smoke
    python model/prepare_data.py

Outputs are indexed identically by row in metadata.csv and both .npy arrays.
The unit of splitting is station + recording date, so even nearby recordings
from the same station/day cannot leak across train/val/test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from numpy.lib.format import open_memmap
from sklearn.model_selection import GroupShuffleSplit

if __package__:
    from .features import AcousticFeatures, CONFIG
else:
    from features import AcousticFeatures, CONFIG

LABELS = ('bio', 'geo', 'antro')


def make_splits(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign approximately 70/15/15 by station/day, deterministically."""
    if manifest[['station_id', 'recording_date_local']].isna().any().any():
        raise ValueError('Station and date metadata required for leakage-aware splits')
    groups = (manifest['station_id'].astype(str) + '|' +
              manifest['recording_date_local'].astype(str)).to_numpy()
    if len(np.unique(groups)) < 4:
        raise ValueError('Need at least 4 station/date groups for a 3-way split')
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, remainder_idx = next(outer.split(manifest, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_local, test_local = next(inner.split(
        manifest.iloc[remainder_idx], groups=groups[remainder_idx]
    ))
    split = np.full(len(manifest), '', dtype=object)
    split[train_idx] = 'train'
    split[remainder_idx[val_local]] = 'val'
    split[remainder_idx[test_local]] = 'test'
    out = manifest.copy()
    out['split'] = split
    return out


def wav_path(source_path: str, audio_root: Path | None, source_prefix: Path) -> Path:
    path = Path(source_path)
    if audio_root is None:
        return path
    try:
        rel = path.relative_to(source_prefix)
    except ValueError as exc:
        raise ValueError(f'Path {path} is not below {source_prefix}') from exc
    return audio_root / rel


def prepare(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    labels = pd.read_csv(args.labels)
    required_manifest = {'audio_id', 'source_path', 'start_second',
                         'station_id', 'recording_date_local', 'site_id'}
    required_labels = {'audio_id', 'source_path', 'segment_index',
                       'start_s', 'end_s', *LABELS}
    if required_manifest - set(manifest.columns):
        raise ValueError(f'Missing manifest columns: {required_manifest - set(manifest.columns)}')
    if required_labels - set(labels.columns):
        raise ValueError(f'Missing labels columns: {required_labels - set(labels.columns)}')
    if manifest.audio_id.duplicated().any():
        raise ValueError('Manifest has duplicate audio_id values')
    if labels.duplicated(['audio_id', 'segment_index']).any():
        raise ValueError('Labels contain duplicate (audio_id, segment_index) keys')
    if labels[list(LABELS)].isna().any().any() or not np.isfinite(labels[list(LABELS)].to_numpy()).all():
        raise ValueError('CEI labels must be finite')
    if ((labels[list(LABELS)] < 0) | (labels[list(LABELS)] > 1)).any().any():
        raise ValueError('CEI labels outside [0,1]')
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        manifest = manifest.head(args.limit).copy()
    manifest = make_splits(manifest, args.seed)

    selected = set(manifest.audio_id)
    labels = labels[labels.audio_id.isin(selected)].copy()
    if len(labels) != len(manifest) * 10:
        raise ValueError('Expected exactly 10 label rows for every selected recording')
    label_by_id = {audio_id: df.sort_values('segment_index')
                   for audio_id, df in labels.groupby('audio_id', sort=False)}

    dest = Path(args.out)
    expected = ('logmel.npy', 'mfcc.npy', 'metadata.csv', 'feature_config.json')
    if not args.overwrite and any((dest / p).exists() for p in expected):
        raise FileExistsError(f'{dest} contains prepared data; change --out or pass --overwrite')
    dest.mkdir(parents=True, exist_ok=True)
    extractor = AcousticFeatures()
    n = len(labels)
    nframes = CONFIG.frames_per_segment
    logmel_file = open_memmap(dest / 'logmel.npy', mode='w+', dtype=np.float32,
                              shape=(n, nframes, CONFIG.n_mels))
    mfcc_file = open_memmap(dest / 'mfcc.npy', mode='w+', dtype=np.float32,
                            shape=(n, nframes, CONFIG.n_mfcc))
    metadata_rows = []
    row_index = 0

    for number, rec in enumerate(manifest.itertuples(index=False), start=1):
        lab = label_by_id.get(rec.audio_id)
        if lab is None or lab.segment_index.tolist() != list(range(10)):
            raise ValueError(f'Missing/non-sequential 6-s labels for {rec.audio_id}')
        path = wav_path(str(rec.source_path), args.audio_root, args.source_prefix)
        with sf.SoundFile(str(path)) as wav:
            if wav.samplerate != CONFIG.sample_rate:
                raise ValueError(f'{path}: expected 48 kHz, got {wav.samplerate} Hz')
            if wav.channels not in (1, 2):
                raise ValueError(f'{path}: expected mono/stereo, got {wav.channels} channels')
            for target in lab.itertuples(index=False):
                expected_start = float(rec.start_second) + float(target.segment_index) * 6
                if abs(float(target.start_s) - expected_start) > 1e-5 or abs(
                    float(target.end_s) - (expected_start + 6)
                ) > 1e-5:
                    raise ValueError(f'Timestamp mismatch for audio_id={rec.audio_id}')
                if str(target.source_path) != str(rec.source_path):
                    raise ValueError(f'Path mismatch for audio_id={rec.audio_id}')
                start_sample = round(expected_start * CONFIG.sample_rate)
                if start_sample + CONFIG.samples_per_segment > wav.frames:
                    raise ValueError(f'{path}: shorter than selected 6-s interval')
                wav.seek(start_sample)
                signal = wav.read(CONFIG.samples_per_segment,
                                  dtype='float32', always_2d=True)
                if len(signal) != CONFIG.samples_per_segment:
                    raise ValueError(f'{path}: incomplete 6-s read')
                mono = signal.mean(axis=1, dtype=np.float32)
                lm, mf = extractor.extract(mono)
                logmel_file[row_index] = lm
                mfcc_file[row_index] = mf
                metadata_rows.append({
                    'row': row_index,
                    'audio_id': rec.audio_id,
                    'segment_index': target.segment_index,
                    'source_path': rec.source_path,
                    'start_s': target.start_s,
                    'end_s': target.end_s,
                    'site_id': rec.site_id,
                    'station_id': rec.station_id,
                    'recording_date_local': rec.recording_date_local,
                    'split': rec.split,
                    **{name: getattr(target, name) for name in LABELS},
                })
                row_index += 1
        if number % 50 == 0 or number == len(manifest):
            print(f'Processed {number}/{len(manifest)} recordings', flush=True)

    logmel_file.flush()
    mfcc_file.flush()
    pd.DataFrame(metadata_rows).to_csv(dest / 'metadata.csv', index=False)
    (dest / 'feature_config.json').write_text(
        json.dumps({**CONFIG.describe(), 'seed': args.seed,
                    'split_unit': 'station_id+recording_date_local',
                    'splits': '70/15/15 by group (approximate by recording)'},
                   indent=2) + '\n', encoding='utf-8'
    )
    print(f'Created {row_index} windows in {dest}')
    print('Log-Mel:', logmel_file.shape, 'MFCC:', mfcc_file.shape)
    print(pd.Series([r['split'] for r in metadata_rows]).value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=Path('data/recordings_1000.csv'))
    parser.add_argument('--labels', type=Path, default=Path('data/cei_6s.csv'))
    parser.add_argument('--out', type=Path, default=Path('data/processed'))
    parser.add_argument('--limit', type=int, default=None, help='First N recordings for a smoke test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audio-root', type=Path, default=None,
                        help='New root replacing --source-prefix in source_path')
    parser.add_argument('--source-prefix', type=Path, default=Path('/escutado-audios'))
    parser.add_argument('--overwrite', action='store_true')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
