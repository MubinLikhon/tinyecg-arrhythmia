# TinyECG: Arrhythmia Classification with a ~1.5k-Parameter 1D CNN

A deliberately small convolutional network that classifies single ECG heartbeats into five arrhythmia classes using only about **1,500 trainable parameters**, reaching **97.8% accuracy** and **0.89 macro F2** on a held-out validation split.

## Why tiny?

Wearables and embedded monitors have tight memory and power budgets, so a heartbeat classifier has to be small as well as accurate. This project treats model size as a first-class objective. Candidate models are compared with a composite score that rewards accuracy and recall-weighted F2 and penalizes parameter count:

```
score = 0.7 × accuracy + 0.3 × macro-F2 − 0.02 × ln(parameters)
```

## Classes

| Label | Class |
| --- | --- |
| 0 | Fusion (F) |
| 1 | Normal (N) |
| 2 | Unknown (Q) |
| 3 | Supraventricular ectopic (S) |
| 4 | Ventricular ectopic (V) |

## Highlights

- **1,487 parameters** for the selected model (width 6), with no large dense layers.
- **Depthwise-separable 1D convolutions** with BatchNorm and global average pooling.
- **Slope channel:** the first difference of each beat is fed as a second input channel, which costs only `7 × width` extra parameters.
- **Class imbalance handling:** Normal beats are about 83% of the data and Fusion about 0.7%, so training uses mild class weights, and per-class logit offsets are tuned afterwards (no extra parameters).
- **EMA weight averaging** to smooth epoch-to-epoch noise in training.
- **Size-aware model selection:** a width sweep picks the model with the best trade-off, not just the best accuracy.

## Data

The notebook expects `data/train.csv` with 187 amplitude columns (`ecg_0` … `ecg_186`) and a `type` column holding the label, as in the table above. A `data/test.csv` of unlabeled beats is optional. The format follows the widely used preprocessed MIT-BIH Arrhythmia heartbeat dataset (one beat per row, five classes). The data is not included in this repository.

<!-- TODO: state your exact data source and license here, and cite it (see References). -->

The training set has 87,554 beats: Normal 72,471, Unknown 6,431, V 5,788, S 2,223, Fusion 641.

## Method

1. **Preprocessing:** per-beat z-score normalization, plus a z-scored first-difference channel.
2. **Model:** a convolutional stem, then three depthwise-separable blocks (depthwise conv, pointwise conv, BatchNorm, ReLU, max-pool), global average pooling and a linear classifier.
3. **Training:** AdamW with a OneCycle learning-rate schedule, light augmentation (amplitude scaling, Gaussian noise, small circular shifts) and class-weighted cross-entropy, `weight = (N / (K·count))^0.1`.
4. **Selection:** the width, and the checkpoint within each run, are chosen by the composite score on a stratified 20% validation split (17,511 beats).
5. **Bias tuning:** per-class logit offsets are fit on the validation split by coordinate ascent.

## Results (validation split)

| Width | Parameters | Accuracy | Macro F2 | Score |
| --- | --- | --- | --- | --- |
| 4 | 785 | 0.9686 | 0.8492 | 0.7995 |
| **6** | **1,487** | **0.9780** | **0.8876** | **0.8048** |
| 8 | 2,397 | 0.9813 | 0.9030 | 0.8021 |

Per-class results for the selected model (width 6):

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Fusion | 0.808 | 0.789 | 0.798 |
| Normal | 0.985 | 0.991 | 0.988 |
| Unknown | 0.984 | 0.981 | 0.982 |
| S | 0.830 | 0.721 | 0.772 |
| V | 0.947 | 0.930 | 0.939 |

Wider models gain a little accuracy but lose on the size penalty, which is why the 1.5k-parameter model comes out on top.

## Limitations

- **Beat-wise split.** The dataset has no patient identifiers, so the validation split is by beat, not by patient. Beats from the same patient can appear in both splits, which tends to inflate results compared to real patient-independent evaluation.
- **Optimistic validation numbers.** The best epoch and the bias offsets are chosen on the same validation split. The Fusion class has only 128 validation beats, so its metrics are noisy.
- **Research code, not a medical device.** This model is not validated for clinical use.

## Usage

```bash
pip install -r requirements.txt
# put train.csv (and optionally test.csv) in ./data
jupyter notebook tinyecg.ipynb
```

Run all cells in order; a GPU speeds up the sweep considerably. Settings such as `WIDTHS`, `SWEEP_EPOCHS`, `CLASS_POWER`, `USE_EMA`, `USE_DIFF` and `USE_BIAS` are at the top of the notebook. The notebook saves the model weights (`model_*.pt`), the bias offsets (`bias_*.npy`) and optional predictions (`predictions_*.csv`) to `outputs/`.

## Repository layout

```
tinyecg.ipynb      full pipeline: data, model, width sweep, evaluation, export
requirements.txt   Python dependencies
README.md
LICENSE
```

## Ideas for future work

- Patient-wise cross-validation for a more realistic estimate.
- Quantization and int8 export for microcontroller deployment.
- Residual or dilated blocks, and a neural architecture search under a fixed parameter budget.
- Multi-seed evaluation to put error bars on the width comparison.

## References

- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, 2001.
- M. Kachuee, S. Fazeli, M. Sarrafzadeh, "ECG Heartbeat Classification: A Deep Transferable Representation," *IEEE ICHI*, 2018.

## License

MIT, see [LICENSE](LICENSE).
