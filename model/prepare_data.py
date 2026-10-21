"""Prepare 48-kHz MFCC/Log-Mel data from right-channel audio and labels.

From repository root:
    python model/prepare_data.py --limit 10 --out data/processed_smoke
    python model/prepare_data.py

No audio resampling. Keep all 10 windows from each recording in one split;
site + calendar date groups cannot cross train/validation/test boundaries.
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
    """Deterministic approximate 70/15/15 split grouped by site and date."""
    groups = (manifest['site_id'].astype(str) + '|' +
              manifest['recording_date_local'].astype(str)).to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError('At least three distinct site/date groups are needed')
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, remainder = next(outer.split(manifest, groups=groups))
    if len(np.unique(groups[remainder])) < 2:
        raise ValueError('Not enough site/date groups to split validation and test')
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_idx, test_idx = next(inner.split(manifest.iloc[remainder], groups=groups[remainder]))
    split = np.full(len(manifest), '', dtype=object)
    split[train] = 'train'
    split[remainder[val_idx]] = 'val'
    split[remainder[test_idx]] = 'test'
    return manifest.assign(split=split)


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
    if 'recording_date_local' not in manifest:
        date_col = next((c for c in ('recording_datetime_local', 'audio_datetime_local')
                         if c in manifest), None)
        if date_col is None:
            raise ValueError('Manifest needs recording_date_local or a local datetime column')
        manifest['recording_date_local'] = pd.to_datetime(
            manifest[date_col], errors='raise').dt.strftime('%Y-%m-%d')

    required_manifest = {'audio_id', 'source_path', 'start_second',
                         'site_id', 'recording_date_local'}
    required_labels = {'audio_id', 'source_path', 'segment_index',
                       'start_s', 'end_s', *LABELS}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f'Missing manifest columns: {sorted(missing)}')
    if missing := required_labels - set(labels.columns):
        raise ValueError(f'Missing labels columns: {sorted(missing)}')
    if manifest[['site_id', 'recording_date_local']].isna().any().any():
        raise ValueError('Site and recording date must be known for each recording')
    if manifest.audio_id.duplicated().any():
        raise ValueError('Duplicate audio_id in manifest')
    if labels.duplicated(['audio_id', 'segment_index']).any():
        raise ValueError('Duplicate audio_id/segment_index in labels')
    y = labels[list(LABELS)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Labels must be finite and in [0,1]')

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        manifest = manifest.head(args.limit).copy()
    manifest = make_splits(manifest, args.seed)
    labels = labels[labels.audio_id.isin(set(manifest.audio_id))].copy()
    if len(labels) != len(manifest) * 10:
        raise ValueError('Expected exactly 10 labels per selected recording')
    label_by_id = {k: d.sort_values('segment_index')
                   for k, d in labels.groupby('audio_id', sort=False)}

    dest = Path(args.out)
    expected = ('logmel.npy', 'mfcc.npy', 'metadata.csv', 'feature_config.json')
    if not args.overwrite and any((dest / f).exists() for f in expected):
        raise FileExistsError(f'Output already exists in {dest}; use --overwrite or another --out')
    dest.mkdir(parents=True, exist_ok=True)
    extractor = AcousticFeatures()
    nframes = CONFIG.frames_per_segment
    logmel = open_memmap(dest / 'logmel.npy', mode='w+', dtype=np.float32,
                         shape=(len(labels), nframes, CONFIG.n_mels))
    mfcc = open_memmap(dest / 'mfcc.npy', mode='w+', dtype=np.float32,
                       shape=(len(labels), nframes, CONFIG.n_mfcc))
    rows = []
    row_index = 0
    for number, rec in enumerate(manifest.itertuples(index=False), start=1):
        lab = label_by_id.get(rec.audio_id)
        if lab is None or lab.segment_index.tolist() != list(range(10)):
            raise ValueError(f'Expected segment indices 0..9 for audio_id={rec.audio_id}')
        path = wav_path(str(rec.source_path), args.audio_root, args.source_prefix)
        with sf.SoundFile(str(path)) as wav:
            if wav.samplerate != CONFIG.sample_rate or wav.channels != 2:
                raise ValueError(f'{path}: expected 48-kHz stereo WAV')
            for target in lab.itertuples(index=False):
                start = float(rec.start_second) + 6 * int(target.segment_index)
                if abs(float(target.start_s) - start) > 1e-5 or abs(
                    float(target.end_s) - (start + 6)) > 1e-5:
                    raise ValueError(f'Inconsistent segment timestamps: audio_id={rec.audio_id}')
                if str(target.source_path) != str(rec.source_path):
                    raise ValueError(f'Source path mismatch: audio_id={rec.audio_id}')
                start_sample = round(start * CONFIG.sample_rate)
                if start_sample + CONFIG.samples_per_segment > wav.frames:
                    raise ValueError(f'Audio too short: {path}')
                wav.seek(start_sample)
                samples = wav.read(CONFIG.samples_per_segment,
                                   dtype='float32', always_2d=True)
                if len(samples) != CONFIG.samples_per_segment:
                    raise ValueError(f'Incomplete audio read: {path}')
                # Stereo channel order: left=0, RIGHT=1. Never average channels.
                right = samples[:, 1]
                lm, mf = extractor.extract(right)
                logmel[row_index] = lm
                mfcc[row_index] = mf
                rows.append({
                    'row': row_index, 'audio_id': rec.audio_id,
                    'segment_index': target.segment_index,
                    'source_path': rec.source_path,
                    'start_s': target.start_s, 'end_s': target.end_s,
                    'site_id': rec.site_id,
                    'recording_date_local': rec.recording_date_local,
                    'split': rec.split,
                    **{k: getattr(target, k) for k in LABELS},
                })
                row_index += 1
        if number % 50 == 0 or number == len(manifest):
            print(f'Processed {number}/{len(manifest)} recordings', flush=True)

    logmel.flush()
    mfcc.flush()
    pd.DataFrame(rows).to_csv(dest / 'metadata.csv', index=False)
    (dest / 'feature_config.json').write_text(json.dumps({
        **CONFIG.describe(), 'seed': args.seed, 'audio_channel': 'right',
        'split_unit': 'site_id+recording_date_local',
        'splits': 'approximately 70/15/15 grouped by site/date',
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Created {row_index} windows in {dest}')
    print('Log-Mel:', logmel.shape, 'MFCC:', mfcc.shape)
    print(pd.Series([r['split'] for r in rows]).value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=Path('data/recordings.csv'))
    parser.add_argument('--labels', type=Path, default=Path('data/cei_6s.csv'))
    parser.add_argument('--out', type=Path, default=Path('data/processed'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audio-root', type=Path, default=None)
    parser.add_argument('--source-prefix', type=Path, default=Path('/escutado-audios'))
    parser.add_argument('--overwrite', action='store_true')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
