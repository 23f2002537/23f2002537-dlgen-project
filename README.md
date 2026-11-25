# 🎵 Messy Mashup: Music Genre Classification

Deep learning pipeline for classifying **noisy, cross-song music mashups** into 10 genres, built for the *Messy Mashup* challenge in the IITM BS **Deep Learning & Generative AI** course (T1 2026).

**Author:** Sahil Kumar · **Roll No:** 23f2002537 · **W&B project:** `23f2002537-t12026`

---

## Overview

The test set consists of 30-second clips made by blending **stems** (drums, bass, vocals, other) taken from *different songs of the same genre*, with environmental noise added on top. The training data, however, is a library of clean, separated stems. The core challenge is this **domain shift**: train on clean stems, predict on noisy mashups.

The solution has three stages:

1. **Synthetic data generation:** build 12,000 training mashups that mimic the test distribution.
2. **Model training:** train three architectures on log-mel spectrograms and compare them by macro F1.
3. **Ensemble inference:** sliding-window, weighted-logit ensemble of GenreCNN and EfficientNet-B0.

**Task:** 10-class audio classification (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
**Metric:** macro-averaged F1

---

## Pipeline

```
 Clean stems (10 genres × 100 songs × 4 stems)        ESC-50 noise (2,000 clips)
            │                                                  │
            └──────────────► Cross-song mashup synthesis ◄─────┘
                              (4 stems from 4 songs, ±10% time-stretch,
                               noise at 0.02–0.10 amplitude, peak-normalised)
                                            │
                              128-bin log-mel spectrogram (128 × 431)
                                            │
                                  SpecAugment (baked in)
                                            │
                              12,000 precomputed .pt tensors
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
          GenreCNN                    EfficientNet-B0                    CRNN
        (from scratch)                 (pretrained)                (CNN + BiGRU)
              └──────────────┬──────────────┘
                             ▼
          0.6 × GenreCNN + 0.4 × EfficientNet logits
          averaged over sliding 10 s windows → argmax → genre
```

### 1. Data augmentation

| Step | Detail |
|------|--------|
| Cross-song stem mixing | Drums, bass, vocals and other stems drawn from **four different songs of the same genre** |
| Time-stretch | 50% chance per stem, rate in [0.9, 1.1]. ±10% was chosen from the tempo EDA; ±15% would distort high-BPM genres such as disco and pop |
| Noise injection | Random [ESC-50](https://github.com/karolpiczak/ESC-50) clip at amplitude 0.02–0.10 (higher values drowned the genre signal) |
| Features | 10 s segments at 22,050 Hz → log-mel spectrogram (`n_fft=2048`, `hop=512`, `n_mels=128`) → per-sample standardisation |
| SpecAugment | 2 frequency masks (≤20 bins) and 2 time masks (≤40 frames) at generation time, plus runtime masking (≤50 frames) during training |
| Output | 12,000 samples (~1,200 per genre), each a `{"mel": (128, 431), "label": int}` tensor |

The EDA covers seven analyses (stem spectrograms, tempo, test-mashup diversity, clip duration, drum rhythm, spectral centroid, MFCCs), and each one drove a specific augmentation choice.

### 2. Models

| Model | Type | Params | Description |
|-------|------|--------|-------------|
| **GenreCNN** | From scratch | 1.70 M | 5 × (Conv → BN → ReLU → MaxPool) blocks with 32→512 channels, global average pooling, 2-layer head |
| **EfficientNet-B0** | Pretrained (ImageNet) | 4.02 M | First conv adapted to 1 channel by averaging RGB weights, input resized to 224×224, new `Dropout(0.4) + Linear` head |
| **CRNN** | From scratch | 4.95 M | 3-block CNN front-end → 2-layer bidirectional GRU → mean pooling → classifier |

**Shared training setup:** label-smoothing cross-entropy (0.1), AdamW (weight decay 1e-4), cosine annealing, gradient clipping (1.0), mixed precision (AMP), batch size 32, 25 epochs, 85/15 train/val split (seed 42 → 10,200 train / 1,800 val), best checkpoint chosen by validation macro F1.
**EfficientNet** uses two phases: 3 epochs training only the head (backbone frozen), then full fine-tuning at a lower LR (3e-4).
All runs are logged to Weights & Biases.

### 3. Inference and ensemble

- Each 30 s test clip is scanned with a **10 s sliding window, 50% overlap** (capped at 8 windows; 5 windows for a 30 s clip).
- Each window gets the same mel processing as in training.
- GenreCNN and EfficientNet-B0 produce logits per window, combined as **0.6 × GenreCNN + 0.4 × EfficientNet-B0**.
- Combined logits are averaged across windows, then `argmax` gives the genre.
- The CRNN was trained and evaluated but **left out of the final ensemble** because its validation F1 was lower and less stable.

---

## Results

Validation performance (best checkpoint, 1,800 held-out samples from the synthetic dataset):

| Model | Macro F1 | Accuracy |
|-------|:--------:|:--------:|
| GenreCNN | 0.9577 | 0.9594 |
| EfficientNet-B0 | **0.9628** | **0.9644** |
| CRNN | 0.9174 | 0.9206 |

> **Note:** these scores are on a validation split of the *synthetic* mashups generated by the pipeline, not on the hidden test set. The final submission (3,020 test clips) is a weighted ensemble of GenreCNN and EfficientNet-B0.

Earlier baselines from the course milestones: a Decision Tree on hand-crafted features (tempo, spectral centroid, ZCR, rolloff, RMS) reached only ~0.18 macro F1, and a first-pass CRNN in Milestone 4 reached 0.67 after 10 epochs.

---

## Repository Structure

```
23f2002537-dlgen-project/
├── notebooks/
│   ├── EDA-and-Data-Augmentation.ipynb   # 7-part EDA + synthetic dataset generation (12k samples)
│   ├── training-models.ipynb             # Trains GenreCNN, EfficientNet-B0, CRNN (W&B logged)
│   ├── DL-23f2002537-notebook-t12026.ipynb  # Final ensemble inference → submission.csv
│   ├── milestone_1.ipynb                 # Dataset validation, corrupt-file & silence diagnostics
│   ├── milestone-2.ipynb                 # EDA + Decision Tree baseline
│   ├── milestone-4.ipynb                 # Synthetic mashups, mel features, first CRNN
│   └── milestone-5.ipynb                 # AST (Audio Spectrogram Transformer) concepts
├── models/
│   ├── GenreCNN_best.pth                 # Best GenreCNN weights (state_dict)
│   ├── EfficientNet_best.pth             # Best EfficientNet-B0 weights (state_dict)
│   └── CRNN_best.pth                     # Best CRNN weights (state_dict)
├── reports/                              # Milestone reports (1, 2, 4, 5) and final project report
└── README.md
```

---

## Getting Started

The notebooks were developed and run on **Kaggle (GPU)**, so dataset and model paths point to `/kaggle/input/...`. To reproduce, update those paths for your environment.

### Dependencies

```bash
pip install torch torchvision librosa numpy pandas scikit-learn matplotlib seaborn tqdm wandb transformers
```

### Run order

1. **`EDA-and-Data-Augmentation.ipynb`**: needs the Messy Mashup stems and ESC-50 noise data. Generates the 12,000 `.pt` training tensors.
2. **`training-models.ipynb`**: trains all three models. Requires a `WANDB_API_KEY` secret (and Kaggle credentials if you upload models).
3. **`DL-23f2002537-notebook-t12026.ipynb`**: loads the trained checkpoints, runs ensemble inference on the test mashups, and writes `submission.csv` (`id`, `genre`).

### Loading a checkpoint

The `.pth` files are `state_dict`s. Copy the model class definitions (`ConvBlock`, `GenreCNN`, `EfficientNetGenre`, `CRNN`) from the notebooks, then:

```python
import torch

model = GenreCNN(num_classes=10)
model.load_state_dict(torch.load("models/GenreCNN_best.pth", map_location="cpu"))
model.eval()

# input: log-mel spectrogram of shape (batch, 1, 128, 431)
logits = model(mel_batch)
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Input features | 128-bin log-mel spectrogram | Keeps rhythm, timbre and brightness as 2D structure CNNs can use; subsumes MFCC information |
| Noise ceiling | 0.10 | Early experiments with amplitudes up to 1.0 overwhelmed the genre signal |
| Time-stretch | ±10% | Stays within genre-coherent BPM ranges for all 10 genres |
| Segment length | 10 s | 3× more samples per clip while still covering several rhythmic cycles |
| EfficientNet training | Freeze → unfreeze | Stops the random head from corrupting pretrained features early on |
| Ensemble | Logit-level soft voting, 0.6 / 0.4 | Keeps confidence information; two-model ensemble was more stable than including the CRNN |

---

## Tech Stack

Python · PyTorch · torchvision · librosa · scikit-learn · Weights & Biases · Hugging Face Transformers (AST, Milestone 5) · Kaggle GPU

---

## Acknowledgements

- IIT Madras BS Programme, Deep Learning & Generative AI course
- [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound dataset
- EfficientNet-B0 ImageNet weights via torchvision
