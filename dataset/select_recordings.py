"""Create a new stratified sample from the original curated terrestrial metadata.

The exact sample used in this project is already committed to data/recordings_1000.csv.
This command is for generating a new selection; it is NOT guaranteed to recreate
that historical draw without its original RNG state and filtering decisions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True, help='Curated source metadata CSV')
    parser.add_argument('--out', default='data/new_recordings_1000.csv')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = pd.read_csv(args.metadata)
    cols = {'Id', 'Url', 'CaptureSiteId', 'SiteName', 'station_id',
            'audio_datetime_local', 'hour_of_day', 'SamplingRate', 'duration_seconds'}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing metadata columns: {sorted(missing)}')
    df = df.dropna(subset=list(cols)).copy()
    df['hour_of_day'] = pd.to_numeric(df['hour_of_day'], errors='coerce')
    df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
    df = df[(df['duration_seconds'] >= 180) & df['hour_of_day'].between(0, 23)]
    df = df.drop_duplicates('Id').copy()
    # Equal allocation over available daypart/site strata via repeated random draws.
    df['daypart'] = df['hour_of_day'].astype(int) // 6
    rng = np.random.default_rng(args.seed)
    groups = {key: value.index.to_numpy().copy() for key, value in df.groupby(['CaptureSiteId', 'daypart'])}
    for ids in groups.values():
        rng.shuffle(ids)
    if sum(map(len, groups.values())) < args.count:
        raise ValueError('Not enough eligible recordings')
    chosen = []
    keys = list(groups)
    rng.shuffle(keys)
    while len(chosen) < args.count:
        progressed = False
        for key in keys:
            if len(chosen) == args.count:
                break
            if len(groups[key]):
                chosen.append(groups[key][-1]); groups[key] = groups[key][:-1]
                progressed = True
        if not progressed:
            break
    df = df.loc[chosen].reset_index(drop=True)
    offsets = rng.choice([0, 60, 120], size=len(df))
    out = pd.DataFrame({
        'audio_id': df['Id'], 'source_path': df['Url'],
        'site_id': df['CaptureSiteId'], 'site_name': df['SiteName'],
        'station_id': df['station_id'],
        'recording_datetime_local': df['audio_datetime_local'],
        'recording_date_local': pd.to_datetime(df['audio_datetime_local']).dt.strftime('%Y-%m-%d'),
        'hour_of_day': df['hour_of_day'].astype(int),
        'start_second': offsets, 'duration_seconds': 60,
        'segment_seconds': 6, 'sampling_rate_hz': df['SamplingRate'],
    })
    dest = Path(args.out); dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f'Saved {len(out)} selected recordings to {dest}')


if __name__ == '__main__':
    main()
