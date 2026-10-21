"""Generate 6-second ecoacoustic targets from a selected-recordings manifest.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from msclap import CLAP
from unidecode import unidecode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', default='data/recordings_1000.csv')
    p.add_argument('--classes', default='dataset/prompts.txt')
    p.add_argument('--out', default='data/cei_6s.csv')
    p.add_argument('--similarities-out', default=None,
                   help='Save raw CLAP similarities so future formulas do not need another inference')
    p.add_argument('--limit', type=int, default=None, help='Try 10 before all 1000')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cpu', action='store_true', help='GPU enabled by default')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')

    samples = pd.read_csv(args.sample)
    required = {'audio_id', 'source_path', 'start_second'}
    if not required.issubset(samples.columns):
        raise ValueError(f'Manifest missing columns: {sorted(required - set(samples.columns))}')
    if samples['audio_id'].duplicated().any():
        raise ValueError('audio_id must be unique in the selection manifest')
    if args.limit is not None:
        samples = samples.head(args.limit)
    original = [line.strip() for line in Path(args.classes).read_text().splitlines() if line.strip()]
    prompts = [unidecode(line.lower().split(' - ')[-1]) for line in original]
    labels = [unidecode(line.lower()) for line in original]
    groups = {
        'bio': np.array(['biophony' in name for name in labels]),
        'geo': np.array(['geophony' in name for name in labels]),
        'antro': np.array([('anthropophony' in name or 'technophony' in name) for name in labels]),
    }
    if not all(mask.sum() >= 2 for mask in groups.values()):
        raise ValueError('Continuous Top-2 needs at least two prompts per category')

    cuda = not args.cpu
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check PyTorch installation or pass --cpu.')
    model = CLAP(version='2023', use_cuda=cuda)
    text_embeddings = model.get_text_embeddings(prompts)
    scores = []
    records = []
    pending = []
    pending_records = []

    def process_batch():
        if not pending:
            return
        with tempfile.TemporaryDirectory(prefix='cei6_') as tmp:
            paths = []
            for i, (audio, sr) in enumerate(pending):
                filename = str(Path(tmp) / f'{i:05d}.wav')
                sf.write(filename, audio, sr, subtype='PCM_16')
                paths.append(filename)
            audio_embeddings = model.get_audio_embeddings(paths)
            if isinstance(audio_embeddings, torch.Tensor):
                audio_embeddings = audio_embeddings.detach().cpu()
            else:
                audio_embeddings = torch.from_numpy(np.asarray(audio_embeddings))
            result = model.compute_similarity(audio_embeddings, text_embeddings)
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            result = np.asarray(result)
            if result.shape != (len(pending_records), len(prompts)):
                raise ValueError(f'Unexpected similarity shape: {result.shape}')
            scores.append(result)
            records.extend(pending_records)
        pending.clear()
        pending_records.clear()

    for i, r in enumerate(samples.itertuples(index=False), 1):
        try:
            with sf.SoundFile(str(r.source_path)) as wav:
                sr = wav.samplerate
                start = int(round(float(r.start_second) * sr))
                length = int(round(60.0 * sr))
                if start + length > len(wav):
                    raise ValueError('recording shorter than selected 60-second interval')
                wav.seek(start)
                audio = wav.read(length, dtype='float32', always_2d=True)
                if len(audio) != length:
                    raise ValueError('incomplete read')
            mono = audio.mean(axis=1)
            n = int(round(6.0 * sr))
            for j in range(10):
                segment = mono[j*n:(j+1)*n]
                if len(segment) != n:
                    raise ValueError('incomplete 6-second segment')
                pending.append((segment, sr))
                pending_records.append({
                    'audio_id': r.audio_id,
                    'source_path': r.source_path,
                    'segment_index': j,
                    'start_s': float(r.start_second) + 6*j,
                    'end_s': float(r.start_second) + 6*(j+1),
                })
                if len(pending) >= args.batch_size:
                    process_batch()
        except Exception as exc:
            # Failure affects the current recording; incomplete pending segments from
            # this recording must not be silently labeled as a complete recording.
            print(f'SKIP audio_id={r.audio_id}: {exc}')
        if i % 50 == 0:
            print(f'Read {i}/{len(samples)} recordings')
    process_batch()
    if not scores:
        raise RuntimeError('No segments processed')

    sim = np.concatenate(scores, axis=0)
    q = np.clip((sim - 5.0) / 15.0, 0.0, 1.0)
    result = pd.DataFrame.from_records(records)
    for name, mask in groups.items():
        top2 = np.sort(q[:, mask], axis=1)[:, -2:]
        result[name] = 0.75 * top2[:, -1] + 0.25 * top2[:, -2]

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest, index=False)
    print(f'Saved {len(result)} 6-second segments -> {dest}')
    if args.similarities_out:
        sim_dest = Path(args.similarities_out)
        sim_dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sim_dest,
            similarities=sim,
            prompts=np.asarray(prompts),
            audio_id=result['audio_id'].astype(str).to_numpy(),
            segment_index=result['segment_index'].to_numpy(),
            start_s=result['start_s'].to_numpy(),
        )
        print(f'Saved raw CLAP similarities -> {sim_dest}')
    print('Method: q=clip((s-5)/15,0,1); index=0.75*top1 + 0.25*top2')


if __name__ == '__main__':
    main()
