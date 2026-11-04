
# Student-model training and evaluation

This document describes the training and evaluation of ecoacoustic student models using CLAP-derived pseudo-labels.

## Objective

Predict three independent continuous indices:

- Bio: biophony.
- Geo: geophony.
- Antro: anthropophony and technophony.

The targets are values in [0, 1]. Neural models use three independent sigmoid outputs rather than Softmax.

The indices measure agreement with CLAP-derived pseudo-labels and should not be interpreted as independently validated ecological ground truth.

## Dependencies

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

The full neural training workflow also requires TensorFlow, SciPy and joblib:

```bash
python -m pip install tensorflow scipy joblib
```

## Dataset and features

Feature preparation:

```bash
python model/prepare_data.py
```

Expected outputs:

```text
data/processed/
    mfcc.npy
    logmel.npy
    metadata.csv
    feature_config.json
```

The dataset contains 50,000 six-second examples.

Data splits are assigned by site and calendar date to reduce leakage between nearby recordings.

Training-only statistics are used for feature normalization.

## Model comparison

The experiment compares four architectures:

- Ridge.
- MLP.
- CNN.
- DS-CNN.

Each architecture is evaluated with:

- MFCC features.
- Log-Mel features.

Eight model-feature combinations are compared.

## Running experiments

First run tests:

```bash
python -m unittest discover -s tests -v
```

Run an individual baseline:

```bash
python model/train.py --model ridge --feature mfcc
```

Run all experiments:

```bash
python model/train.py --model all --feature all
```

Use `--overwrite` to rerun completed experiments.

A small neural smoke test can be performed with:

```bash
python model/train.py --model cnn --feature mfcc --epochs 2
```

Neural training uses early stopping to limit unnecessary epochs.

## Evaluation

Models are evaluated separately for Bio, Geo and Antro using:

- MAE.
- RMSE.
- R².
- Spearman correlation.

Macro MAE is the unweighted mean of the three category MAEs.

Model selection uses validation metrics. The test set is reserved for final reporting.

The comparison concerns complete feature-model pipelines rather than isolated neural architectures.

## Experiment artifacts

Training outputs are stored under:

```text
results/training/
```

A completed experiment may include:

- `metrics.json`.
- `test_predictions.csv`.
- `model.keras` or `model.joblib`.
- `history.csv`.
- `normalization.npz`.

The comparison table is saved as:

```text
results/training/comparison.csv
```

Large training artifacts are excluded from Git.

## Selected deployment model

The selected model is `cnn_logmel`.

| Property | Value |
| --- | --- |
| Parameters | 5,403 |
| Input | `[1, 299, 40, 1]` |
| Output | `[1, 3]` |
| Precision | float32 |
| Validation macro MAE | 0.068134 |
| TFLite model size | 25,540 bytes |

The float32 TFLite model was checked against the Keras model.

Observed validation metrics:

- Keras macro MAE: 0.068134.
- TFLite macro MAE: 0.068134.
- Maximum absolute prediction difference: approximately `4.17e-7`.

The model is deployed on the ESP32-S3 using TensorFlow Lite Micro.

## On-device evaluation

The ESP32-S3 uses:

- ESP-IDF 6.0.2.
- Streaming Log-Mel DSP.
- CNN float32 inference.
- 256 KiB PSRAM tensor arena.
- Approximately 146 KiB actual arena usage.

Observed inference latency is approximately 490 ms per six-second segment.

The firmware publishes Bio, Geo and Antro over MQTT/TLS.

Initial hardware tests confirmed continuous acquisition and inference without I2S errors or ring-buffer overflows.

A 24-hour unattended stability test remains pending.

## Interpretation and limitations

The student learns to approximate the CLAP-derived indices.

Its outputs should be interpreted as continuous measures of acoustic-category evidence, not calibrated event probabilities.

The training dataset emphasizes natural environments. Generalization to substantially different acoustic environments has not been established.

Future evaluations can compare student predictions against CLAP on newly collected in-domain audio.
