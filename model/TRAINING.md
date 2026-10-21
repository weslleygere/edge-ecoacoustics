# Student-model training and comparison

Place `model/architectures.py`, `model/train.py` and `tests/test_training.py` at the corresponding paths in the existing repository. Keep the previously validated feature extractor and data-preparation scripts unchanged.

## Dependencies

The existing `requirements.txt` already installs NumPy, pandas and scikit-learn. For training, also install TensorFlow, SciPy and joblib in a compatible Python environment:

```bash
python -m pip install tensorflow scipy joblib
```

For CUDA support, follow the TensorFlow installation procedure that matches your GPU drivers and operating system. The linear Ridge baseline runs without TensorFlow, although this implementation imports the architecture module without initializing TensorFlow.

## Data

Prepare the original **48-kHz** audio features first:

```bash
python model/prepare_data.py
```

Expect `data/processed/{mfcc.npy,logmel.npy,metadata.csv,feature_config.json}`. Features and labels are already aligned and split by station/day. No teacher rerun is needed.

## Training

Smoke test the Ridge baseline first:

```bash
python -m unittest discover -s tests -v
python model/train.py --model ridge --feature mfcc
```

Then run all **8 pipelines** (Ridge/MLP/CNN/DS-CNN × MFCC/Log-Mel):

```bash
python model/train.py --model all --feature all
```

To rerun a finished experiment, use `--overwrite`. To try a small neural run, use e.g. `--model cnn --feature mfcc --epochs 2` before the full run. By default training allows up to 100 epochs with early stopping after 12 epochs without validation-loss improvement.

## Outputs

Each `results/training/<architecture>_<feature>/` directory includes `metrics.json`, `test_predictions.csv`, a saved model (`model.keras` or `model.joblib`), and neural models also include `history.csv` and `normalization.npz`. `results/training/comparison.csv` combines all completed experiments, **ranked by validation MAE**, not by test MAE.

Each CEI pseudo-label is evaluated independently with MAE, RMSE, R² and Spearman correlation. The macro MAE is an unweighted average across Bio, Geo and Antro. Metrics represent fidelity to CLAP-derived pseudo-labels, not an independent ecological ground truth.

The Ridge baseline pools feature time series with mean, standard deviation and maximum, standardizes these pooled statistics using training data only, and fits three Ridge outputs. The neural MLP uses temporal mean and maximum inside the model, while CNN and DS-CNN receive full frame matrices. Comparisons therefore concern entire pipelines, not architecture-only ablations.

The neural models share identical training hyperparameters and split assignments. Neural normalization uses only training frames and is serialized for eventual on-device reproduction. Model outputs use three independent sigmoids; no Softmax.

**Important:** choose deployment candidates using validation metrics and estimated complexity. The test set should only support final reporting. The `saved_model_bytes` column is the size of the serialized artifact, **not** INT8 size, peak SRAM or Flash usage. Quantization, TFLite Micro compatibility and actual ESP32-S3 benchmarks are intentionally deferred.

## Validation limits

The script is syntax-checked and its CPU-only metrics, pooling and Ridge path can be tested locally with synthetic features. Running the complete experiment requires your large `data/processed` cache and a TensorFlow-capable environment; the project audio is not part of Git.
