# SKANN-SSL V2.1.0 Demo

**Underwater Acoustic Vessel Classification System**

Self-supervised learning approach for classifying vessel types from hydrophone audio signatures.

© 2026 Oravont Systems LLP. All rights reserved.

---

## Features

- **4-Class Vessel Classification**: Cargo Ship, Tanker, Fishing Vessel, Small Craft
- **Real-time Audio Playback**: Listen to acoustic signatures while classifying
- **Radar Plot Visualization**: Interactive probability display
- **Physics-Aware Architecture**: Selective Kernel convolutions tuned for underwater acoustics (15–500+ Hz)

---

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo
python skann_ssl_demo.py
```

### Requirements

- Python 3.10+
- Windows / macOS / Linux
- ~500 MB disk space (model + sample data)

---

## Folder Structure

```
SKANN-SSL-Demo/
├── skann_ssl_demo.py          # Main demo application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── model/
│   ├── SKANN_SSL_Production_Bundle.joblib    # Trained encoder
│   └── vessel_territories.joblib              # Classification centroids
└── data/
    ├── manifest.csv           # Clip metadata
    └── tensors/               # Audio tensors (16 kHz, 1 second)
        ├── tensor_000000.npy
        ├── tensor_000001.npy
        └── ...
```

---

## Usage

1. **Launch**: Run `python skann_ssl_demo.py`
2. **Select Clip**: Use dropdown or click "🎲 Random"
3. **Classify**: Click "🎯 CLASSIFY"
4. **Listen**: Audio plays in loop (click "🔇 Stop" to mute)
5. **Review**: Radar plot shows class probabilities

---

## Model Performance

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.8299 |
| Embedding Dimension | 128 |
| Vessel Classes | 4 |
| Sample Rate | 16 kHz |
| Clip Duration | 1 second |

### Frequency Coverage (SK Kernels)

| Kernel Size | Frequency | Acoustic Phenomenon |
|-------------|-----------|---------------------|
| 1023 | 15+ Hz | Shaft rate |
| 511 | 31+ Hz | Generator (25 Hz) |
| 255 | 62+ Hz | Generator (50 Hz) |
| 127 | 125+ Hz | Blade pass |
| 63 | 250+ Hz | Hull resonance |
| 31 | 500+ Hz | Cavitation |

---

## Troubleshooting

**"sounddevice not installed"**
```bash
pip install sounddevice
```

**No audio output**
- Check system audio settings
- Demo works without audio (visual classification still functional)

**CUDA errors on CPU machine**
- The demo automatically handles CPU-only environments

---

## Contact

For licensing, integration, or technical support:

**Oravont Systems LLP**

---

*SKANN-SSL: Selective Kernel Audio Neural Networks with Self-Supervised Learning*
