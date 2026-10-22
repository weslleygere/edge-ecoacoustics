"""Select a balanced 48-kHz RIGHT-channel dataset with clipping screening.
"""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import soundfile as sf

AUDIT_CSV = 'audio_quality_integrated.csv'
COLUMNS = [
    'Id', 'Url', 'CaptureSiteId', 'SiteName',
    'audio_datetime_local', 'hour_of_day', 'wav_status', 'file_exists',
    'actual_rate', 'actual_channels', 'actual_bits', 'actual_duration_seconds',
]
SAMPLE_RATE = 48_000
SEGMENT_SECONDS = 60
PCM_FULL_SCALE = 32768


def load_inventory(path):
    """Use the verified WAV-header fields rather than database assumptions."""
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            with archive.open(AUDIT_CSV) as source:
                df = pd.read_csv(source, usecols=COLUMNS, low_memory=False)
    else:
        df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    good = (
        df['wav_status'].eq('header_ok') &
        df['file_exists'].eq(True) &
        df['actual_rate'].eq(SAMPLE_RATE) &
        df['actual_channels'].eq(2) &
        df['actual_bits'].eq(16) &
        df['actual_duration_seconds'].ge(SEGMENT_SECONDS) &
        df['hour_of_day'].between(0, 23)
    )
    df = df.loc[good].dropna(subset=[
        'Id', 'Url', 'CaptureSiteId', 'SiteName',
        'audio_datetime_local', 'hour_of_day', 'actual_duration_seconds',
    ]).drop_duplicates('Id').reset_index(drop=True)
    if df.empty:
        raise ValueError('No suitable 48-kHz stereo PCM16 recordings in inventory')
    return df


def quotas_by_site_hour(df, count, rng):
    """Make near-uniform quotas simultaneously across site, hour and their cells."""
    cells = df.groupby(['CaptureSiteId', 'hour_of_day']).indices
    keys = sorted(cells)
    if count > len(df):
        raise ValueError('Requested more recordings than there are candidates')
    quotas = {key: 0 for key in keys}
    per_site = Counter()
    per_hour = Counter()
    # Random order only breaks exact ties; distribution remains reproducible.
    tie_order = list(rng.permutation(len(keys)))
    tie_rank = {keys[i]: n for n, i in enumerate(tie_order)}
    for _ in range(count):
        possible = [key for key in keys if quotas[key] < len(cells[key])]
        if not possible:
            raise ValueError('Not enough candidates across the available site/hour groups')
        key = min(possible, key=lambda k: (
            per_site[k[0]], per_hour[k[1]], quotas[k], tie_rank[k]
        ))
        quotas[key] += 1
        per_site[key[0]] += 1
        per_hour[key[1]] += 1
    return cells, quotas


def longest_true_run(mask, carry):
    """Return longest run and trailing run; support runs spanning read blocks."""
    if not mask.any():
        return 0, 0
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    starts, ends = changes[::2], changes[1::2]
    lengths = ends - starts
    max_run = int(max(lengths.max(), carry + lengths[0] if starts[0] == 0 else 0))
    trailing = int(lengths[-1]) if ends[-1] == len(mask) else 0
    if starts[0] == 0 and len(starts) == 1 and ends[0] == len(mask):
        trailing += carry
    return max_run, trailing


def measure_clipping(path, offset_s):
    """Inspect RIGHT channel (index 1) only; never mix channels."""
    with sf.SoundFile(str(path)) as wav:
        if wav.samplerate != SAMPLE_RATE or wav.channels != 2 or wav.subtype != 'PCM_16':
            raise ValueError(f'Unexpected audio format in {path}')
        start = round(offset_s * SAMPLE_RATE)
        n_samples = SAMPLE_RATE * SEGMENT_SECONDS
        if start + n_samples > wav.frames:
            raise ValueError(f'Selected interval exceeds WAV length: {path}')
        wav.seek(start)
        peak, near_count, full_count = 0, 0, 0
        max_near_run, near_carry = 0, 0
        max_full_run, full_carry = 0, 0
        remaining = n_samples
        while remaining:
            block = wav.read(min(5 * SAMPLE_RATE, remaining), dtype='int16', always_2d=True)
            if not len(block):
                raise ValueError(f'Unexpected end of WAV: {path}')
            right = block[:, 1].astype(np.int32)
            magnitude = np.abs(right)  # int32: abs(-32768) is safe
            peak = max(peak, int(magnitude.max()))
            near = magnitude >= int(np.ceil(0.98 * PCM_FULL_SCALE))
            full = magnitude >= 32767
            near_count += int(near.sum())
            full_count += int(full.sum())
            current, near_carry = longest_true_run(near, near_carry)
            max_near_run = max(max_near_run, current)
            current, full_carry = longest_true_run(full, full_carry)
            max_full_run = max(max_full_run, current)
            remaining -= len(block)
    return {
        'peak_fraction_fs': peak / PCM_FULL_SCALE,
        'fraction_ge_98pct_fs': near_count / n_samples,
        'fraction_at_full_scale': full_count / n_samples,
        'longest_near_full_run': max_near_run,
        'longest_full_scale_run': max_full_run,
    }


def resolve_path(original, audio_root, source_prefix):
    original = Path(original)
    if audio_root is None:
        return original
    try:
        return audio_root / original.relative_to(source_prefix)
    except ValueError as exc:
        raise ValueError(f'WAV path is not under {source_prefix}: {original}') from exc


def select(args):
    rng = np.random.default_rng(args.seed)
    df = load_inventory(args.metadata)
    cells, quotas = quotas_by_site_hour(df, args.count, rng)
    print(f'Inventory: {len(df):,} eligible WAVs, {len(cells)} site/hour groups.', flush=True)
    selected, audited = [], []
    # Keep each candidate's screening result final, including across refill passes.
    candidates = {key: iter(rng.permutation(cells[key])) for key in sorted(cells)}
    exhausted = set()
    per_site, per_hour, per_cell = Counter(), Counter(), Counter()
    while len(selected) < args.count:
        for key in sorted(cells):
            needed = quotas[key]
            if not needed:
                continue
            accepted = 0
            for idx in candidates[key]:
                r = df.iloc[int(idx)]
                max_offset = int(np.floor(float(r.actual_duration_seconds) - SEGMENT_SECONDS + 1e-7))
                offset = int(rng.integers(0, max_offset + 1))
                path = resolve_path(r.Url, args.audio_root, args.source_prefix)
                if not path.is_file():
                    raise FileNotFoundError(f'WAV not mounted: {path}. Use --audio-root if needed.')
                metrics = measure_clipping(path, offset)
                # Conservative, transparent screening. A single high peak is not enough.
                clipped = (metrics['longest_full_scale_run'] >= args.full_run_limit or
                           metrics['fraction_ge_98pct_fs'] >= args.near_fraction_limit)
                audited.append({
                    'audio_id': r.Id, 'site_id': r.CaptureSiteId,
                    'hour_of_day': int(r.hour_of_day), 'start_second': offset,
                    **metrics, 'clipping_flag': bool(clipped),
                })
                if not clipped:
                    selected.append({
                        'audio_id': r.Id, 'source_path': r.Url,
                        'site_id': r.CaptureSiteId, 'site_name': r.SiteName,
                        'recording_datetime_local': r.audio_datetime_local,
                        'recording_date_local': str(r.audio_datetime_local)[:10],
                        'hour_of_day': int(r.hour_of_day),
                        'start_second': offset, 'duration_seconds': SEGMENT_SECONDS,
                        'segment_seconds': 6, 'sampling_rate_hz': SAMPLE_RATE,
                    })
                    accepted += 1
                    per_site[key[0]] += 1
                    per_hour[key[1]] += 1
                    per_cell[key] += 1
                    if accepted == needed:
                        break
            if accepted < needed:
                exhausted.add(key)
                print(f'Site/hour {tuple(map(int, key))}: needed {needed}, found {accepted}; '
                      'redistributing shortfall.', flush=True)
            if len(selected) % 500 < needed:
                print(f'Selected {len(selected)}/{args.count} recordings...', flush=True)

        shortfall = args.count - len(selected)
        if not shortfall:
            break
        available = [key for key in sorted(cells) if key not in exhausted]
        if not available:
            raise RuntimeError(
                f'Insufficient unclipped recordings across all site/hour groups: '
                f'needed {args.count}, found {len(selected)}. '
                'Reduce --count or loosen thresholds.'
            )
        # Allocate replacement slots using existing accepted totals for balance.
        quotas = dict.fromkeys(cells, 0)
        site_totals, hour_totals, cell_totals = per_site.copy(), per_hour.copy(), per_cell.copy()
        for _ in range(shortfall):
            key = min(available, key=lambda k: (
                site_totals[k[0]], hour_totals[k[1]], cell_totals[k], k
            ))
            quotas[key] += 1
            site_totals[key[0]] += 1
            hour_totals[key[1]] += 1
            cell_totals[key] += 1

    manifest = pd.DataFrame(selected).sort_values(
        ['site_id', 'hour_of_day', 'audio_id']
    ).reset_index(drop=True)
    audit = pd.DataFrame(audited)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (args.out.exists() or args.audit_out.exists()):
        raise FileExistsError('Output already exists; use --overwrite or change output paths')
    manifest.to_csv(args.out, index=False)
    audit.to_csv(args.audit_out, index=False)
    print(f'Saved {len(manifest):,} recordings: {args.out}')
    print(f'Clipping audit: {args.audit_out} ({int(audit.clipping_flag.sum())} rejected)')
    print('Per-site range:', manifest.groupby('site_id').size().agg(['min', 'max']).tolist())
    print('Per-hour range:', manifest.groupby('hour_of_day').size().agg(['min', 'max']).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--metadata', type=Path, required=True, help='outputs.zip or extracted audio_quality_integrated.csv')
    p.add_argument('--out', type=Path, default=Path('data/recordings.csv'))
    p.add_argument('--audit-out', type=Path, default=Path('data/recordings_clipping.csv'))
    p.add_argument('--count', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--audio-root', type=Path, default=None,
                   help='Root where original /escutado-audios/... WAVs are mounted')
    p.add_argument('--source-prefix', type=Path, default=Path('/escutado-audios'))
    p.add_argument('--near-fraction-limit', type=float, default=0.01,
                   help='Flag if >= this fraction of samples is at >=98%% FS (default: 1%%)')
    p.add_argument('--full-run-limit', type=int, default=3,
                   help='Flag if this many full-scale samples occur consecutively (default: 3)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.count <= 0 or not (0 < args.near_fraction_limit <= 1) or args.full_run_limit <= 0:
        p.error('count and full-run-limit must be positive, near-fraction-limit in (0,1]')
    select(args)


if __name__ == '__main__':
    main()
