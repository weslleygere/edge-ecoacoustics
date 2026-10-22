"""Compare Ridge, MLP, CNN and DS-CNN on 48-kHz MFCC and Log-Mel.

Run from the repository root after model/prepare_data.py:
    python model/train.py --model all --feature all
    python model/train.py --model ridge --feature mfcc

Selection should use VALIDATION metrics only; test results are for reporting.
No quantization or ESP32 benchmarking occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

from architectures import build_model

TARGETS = ('bio', 'geo', 'antro')
MODELS = ('ridge', 'mlp', 'cnn', 'dscnn')
FEATURES = ('mfcc', 'logmel')


def statistics(x):
    """Fixed temporal pooling for the linear reference; shape [N,3*F]."""
    x = np.asarray(x, dtype=np.float32)
    return np.concatenate((x.mean(axis=1), x.std(axis=1),
                           x.max(axis=1)), axis=1).astype(np.float32)


def evaluate(y_true, y_pred):
    """Scores for 3 continuous pseudo-labels; avoid accuracy terminology."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {}
    for j, name in enumerate(TARGETS):
        a, b = y_true[:, j], y_pred[:, j]
        rho = spearmanr(a, b).statistic if len(a) > 1 else float('nan')
        metrics[name] = {
            'mae': float(mean_absolute_error(a, b)),
            'rmse': float(np.sqrt(mean_squared_error(a, b))),
            'r2': float(r2_score(a, b)) if len(a) > 1 else float('nan'),
            'spearman': float(rho) if np.isfinite(rho) else None,
        }
    metrics['macro_mae'] = float(np.mean([metrics[k]['mae'] for k in TARGETS]))
    return metrics


def load_data(root):
    root = Path(root)
    meta = pd.read_csv(root / 'metadata.csv')
    if not set(TARGETS + ('split', 'row')).issubset(meta.columns):
        raise ValueError('metadata.csv lacks required labels, split or row')
    if meta['row'].tolist() != list(range(len(meta))):
        raise ValueError('metadata rows must match .npy row order')
    if meta['split'].isna().any() or not set(meta.split.unique()).issubset({'train', 'val', 'test'}):
        raise ValueError('Unexpected data split')
    index = {}
    for split in ('train', 'val', 'test'):
        index[split] = np.flatnonzero(meta.split.to_numpy() == split)
        if len(index[split]) < 2:
            raise ValueError(f'Missing or insufficient {split} examples')
    if 'site_id' in meta and 'recording_date_local' in meta:
        groups = meta.site_id.astype(str) + '|' + meta.recording_date_local.astype(str)
        splits_per_group = pd.DataFrame({'group': groups, 'split': meta.split}).groupby('group').split.nunique()
        if (splits_per_group > 1).any():
            raise ValueError('Site/day leakage detected in split assignments')
    return meta, index


def fit_feature_scaler(x, train_idx, batch=256):
    """Compute per-bin mean/variance over training frames only, in batches."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    for begin in range(0, len(train_idx), batch):
        chunk = np.asarray(x[train_idx[begin:begin + batch]], dtype=np.float32)
        scaler.partial_fit(chunk.reshape(-1, chunk.shape[-1]))
    return scaler


def neural_arrays(x, idx, scaler):
    a = np.array(x[idx], dtype=np.float32, copy=True)
    a -= scaler.mean_.astype(np.float32)[None, None, :]
    a /= np.maximum(scaler.scale_, 1e-6).astype(np.float32)[None, None, :]
    return a[..., None]


def train_one(args, feature, model_name, meta, indices, all_rows):
    run_dir = args.out / f'{model_name}_{feature}'
    run_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(args.data / f'{feature}.npy', mmap_mode='r')
    if len(x) != len(meta) or x.ndim != 3 or x.shape[1] != 299:
        raise ValueError(f'{feature} array and metadata disagree')
    expected_freq = 13 if feature == 'mfcc' else 40
    if x.shape[2] != expected_freq:
        raise ValueError(f'Unexpected {feature} shape: {x.shape}')
    y = meta.loc[:, TARGETS].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or not ((y >= 0) & (y <= 1)).all():
        raise ValueError('Invalid target labels')
    train, val, test = (indices[s] for s in ('train', 'val', 'test'))

    if model_name == 'ridge':
        xt = statistics(x[train])
        xv = statistics(x[val])
        xe = statistics(x[test])
        scaler = StandardScaler().fit(xt)
        reg = Ridge(alpha=args.ridge_alpha)
        reg.fit(scaler.transform(xt), y[train])
        pred_val = np.clip(reg.predict(scaler.transform(xv)), 0, 1)
        pred_test = np.clip(reg.predict(scaler.transform(xe)), 0, 1)
        joblib.dump({'model': reg, 'scaler': scaler, 'pooling': 'mean,std,max'}, run_dir / 'model.joblib')
        parameters = int(reg.coef_.size + reg.intercept_.size)
        model_file = run_dir / 'model.joblib'
    else:
        import tensorflow as tf
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
        scaler = fit_feature_scaler(x, train)
        np.savez(run_dir / 'normalization.npz', mean=scaler.mean_.astype('float32'),
                 scale=scaler.scale_.astype('float32'))
        xt = neural_arrays(x, train, scaler)
        xv = neural_arrays(x, val, scaler)
        model = build_model(model_name, (x.shape[1], x.shape[2]))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                      loss=tf.keras.losses.Huber(delta=0.1),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
        checkpoint = run_dir / 'model.keras'
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss',
                                               save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                             restore_best_weights=True),
        ]
        history = model.fit(xt, y[train], validation_data=(xv, y[val]),
                            epochs=args.epochs, batch_size=args.batch_size,
                            callbacks=callbacks, verbose=2)
        pd.DataFrame(history.history).to_csv(run_dir / 'history.csv', index=False)
        model = tf.keras.models.load_model(checkpoint)
        pred_val = model.predict(xv, batch_size=args.batch_size, verbose=0)
        xe = neural_arrays(x, test, scaler)
        pred_test = model.predict(xe, batch_size=args.batch_size, verbose=0)
        pred_val = np.clip(pred_val, 0, 1)
        pred_test = np.clip(pred_test, 0, 1)
        parameters = int(model.count_params())
        model_file = checkpoint
        del xt, xv, xe, model
        tf.keras.backend.clear_session()

    results = {'model': model_name, 'feature': feature, 'parameters': parameters,
               'saved_model_bytes': model_file.stat().st_size,
               'val': evaluate(y[val], pred_val), 'test': evaluate(y[test], pred_test)}
    (run_dir / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    predictions = meta.iloc[test][['row', 'audio_id', 'segment_index']].copy()
    for j, target in enumerate(TARGETS):
        predictions[f'true_{target}'] = y[test, j]
        predictions[f'pred_{target}'] = pred_test[:, j]
    predictions.to_csv(run_dir / 'test_predictions.csv', index=False)
    print(f'{model_name:5s} {feature:6s}: val MAE={results["val"]["macro_mae"]:.4f}, '
          f'test MAE={results["test"]["macro_mae"]:.4f}, parameters={parameters}')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--out', type=Path, default=Path('results/training'))
    p.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    p.add_argument('--feature', choices=(*FEATURES, 'all'), default='all')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ridge-alpha', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.ridge_alpha < 0:
        p.error('epochs, batch-size, patience must be positive; ridge-alpha nonnegative')
    meta, indices = load_data(args.data)
    chosen_models = MODELS if args.model == 'all' else (args.model,)
    chosen_features = FEATURES if args.feature == 'all' else (args.feature,)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature in chosen_features:
        for model in chosen_models:
            result_path = args.out / f'{model}_{feature}' / 'metrics.json'
            if result_path.exists() and not args.overwrite:
                print(f'SKIP {model}_{feature}: existing metrics; use --overwrite to rerun')
                rows.append(json.loads(result_path.read_text()))
                continue
            rows.append(train_one(args, feature, model, meta, indices, None))
    # Avoid picking a winner using test outcomes: sort by validation MAE.
    all_results = []
    for path in sorted(args.out.glob('*/metrics.json')):
        all_results.append(json.loads(path.read_text()))
    if all_results:
        table = []
        for r in all_results:
            row = {'model': r['model'], 'feature': r['feature'],
                   'parameters': r['parameters'], 'saved_model_bytes': r['saved_model_bytes']}
            for split in ('val', 'test'):
                row[f'{split}_macro_mae'] = r[split]['macro_mae']
                for name in TARGETS:
                    for key in ('mae', 'rmse', 'r2', 'spearman'):
                        row[f'{split}_{name}_{key}'] = r[split][name][key]
            table.append(row)
        pd.DataFrame(table).sort_values('val_macro_mae').to_csv(args.out / 'comparison.csv', index=False)
        print(f'Comparison saved to {args.out / "comparison.csv"} (ranked by validation MAE)')


if __name__ == '__main__':
    main()
