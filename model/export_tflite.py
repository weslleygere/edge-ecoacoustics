"""Export the selected CNN + Log-Mel model to fully integer TFLite.

Run from the repository root:
    python model/export_tflite.py

Calibrates with TRAIN data only; checks Keras/TFLite predictions on VALIDATION.
The feature normalization stays OUTSIDE the TFLite graph, for ESP32 parity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('results/training/cnn_logmel/model.keras'))
    parser.add_argument('--normalization', type=Path, default=Path('results/training/cnn_logmel/normalization.npz'))
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    parser.add_argument('--out', type=Path, default=Path('firmware/esp32s3/model'))
    parser.add_argument('--calibration-count', type=int, default=500)
    parser.add_argument('--validation-count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.calibration_count, args.validation_count) < 1:
        parser.error('Sample counts must be positive')

    import tensorflow as tf

    meta = pd.read_csv(args.data / 'metadata.csv')
    x = np.load(args.data / 'logmel.npy', mmap_mode='r')
    if len(meta) != len(x) or x.shape[1:] != (299, 40):
        raise ValueError('Expected metadata-aligned Log-Mel tensor [N, 299, 40]')
    if not {'split', 'row', 'bio', 'geo', 'antro'}.issubset(meta.columns):
        raise ValueError('Missing expected metadata columns')
    if not np.array_equal(meta['row'].to_numpy(), np.arange(len(meta))):
        raise ValueError('metadata.csv does not match feature array row order')
    train = np.flatnonzero(meta['split'].to_numpy() == 'train')
    val = np.flatnonzero(meta['split'].to_numpy() == 'val')
    if not len(train) or not len(val):
        raise ValueError('Train and validation splits are required')

    with np.load(args.normalization) as archive:
        mean = np.asarray(archive['mean'], dtype=np.float32)
        scale = np.asarray(archive['scale'], dtype=np.float32)
    if mean.shape != (40,) or scale.shape != (40,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Invalid Log-Mel normalization statistics')
    if np.any(scale <= 0):
        raise ValueError('Feature scaling must be positive')
    effective_scale = np.maximum(scale, 1e-6)

    net = tf.keras.models.load_model(args.model, compile=False)
    if tuple(net.input_shape[1:]) != (299, 40, 1) or tuple(net.output_shape[1:]) != (3,):
        raise ValueError(f'Unexpected model I/O: {net.input_shape} -> {net.output_shape}')

    rng = np.random.default_rng(args.seed)
    calib_rows = rng.choice(train, size=min(args.calibration_count, len(train)), replace=False)
    val_rows = rng.choice(val, size=min(args.validation_count, len(val)), replace=False)

    def normalized(row):
        sample = np.asarray(x[int(row)], dtype=np.float32).copy()
        sample -= mean[None, :]
        sample /= effective_scale[None, :]
        return sample[None, ..., None]

    def representative_dataset():
        for row in calib_rows:
            yield [normalized(row)]

    converter = tf.lite.TFLiteConverter.from_keras_model(net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if inp['dtype'] != np.int8 or out['dtype'] != np.int8:
        raise ValueError('Expected INT8 input and output, conversion not fully integer')
    if tuple(inp['shape']) != (1, 299, 40, 1) or tuple(out['shape']) != (1, 3):
        raise ValueError('Unexpected quantized model tensor shape')
    in_scale, in_zero = inp['quantization']
    out_scale, out_zero = out['quantization']
    if in_scale <= 0 or out_scale <= 0:
        raise ValueError('Missing valid quantization parameters')

    truth = np.empty((len(val_rows), 3), dtype=np.float32)
    pred_keras = np.empty_like(truth)
    pred_int8 = np.empty_like(truth)
    for i, row in enumerate(val_rows):
        sample = normalized(row)
        truth[i] = meta.iloc[int(row)][['bio', 'geo', 'antro']].to_numpy(dtype=np.float32)
        pred_keras[i] = net(sample, training=False).numpy()[0]
        quantized = np.clip(np.round(sample / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp['index'], quantized)
        interpreter.invoke()
        pred_int8[i] = (interpreter.get_tensor(out['index']).astype(np.float32)[0] - out_zero) * out_scale

    keras_clip = np.clip(pred_keras, 0, 1)
    int8_clip = np.clip(pred_int8, 0, 1)
    diff = np.abs(pred_int8 - pred_keras)
    result = {
        'model': 'cnn_logmel',
        'validation_examples': int(len(val_rows)),
        'calibration_examples': int(len(calib_rows)),
        'calibration_split': 'train',
        'validation_split': 'val',
        'keras_macro_mae': float(np.mean(np.abs(keras_clip - truth))),
        'tflite_macro_mae': float(np.mean(np.abs(int8_clip - truth))),
        'mean_abs_prediction_difference': float(diff.mean()),
        'max_abs_prediction_difference': float(diff.max()),
        'input': {'shape': [1, 299, 40, 1], 'dtype': 'int8', 'scale': float(in_scale), 'zero_point': int(in_zero)},
        'output': {'shape': [1, 3], 'dtype': 'int8', 'scale': float(out_scale), 'zero_point': int(out_zero), 'order': ['bio', 'geo', 'antro']},
        'tflite_bytes': len(model_bytes),
        'normalization': 'Normalize Log-Mel per frequency: (value - mean[f]) / max(scale[f], 1e-6), then quantize using input scale/zero_point.',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cnn_logmel_int8.tflite').write_bytes(model_bytes)
    np.savez(args.out / 'normalization.npz', mean=mean, scale=effective_scale)
    (args.out / 'quantization.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('Export successful. Next: verify operators and memory on ESP32-S3.')


if __name__ == '__main__':
    main()
