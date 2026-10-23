# DetectAI – AI-Based Deepfake Detection System

<div align="center">
  <img src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='80' height='80' fill='%238b5cf6'><path d='M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 6c1.66 0 3 1.34 3 3 0 1.09-.59 2.04-1.46 2.56L15 15H9l1.46-2.44C9.59 12.04 9 11.09 9 10c0-1.66 1.34-3 3-3z'/></svg>" alt="DetectAI Logo" />
  <h3><em>"See through the fake."</em></h3>
  <p><strong>Enterprise-Grade Hybrid AI Detection Engine (EfficientNetB0 CNN + 38-Dim Random Forest)</strong></p>
</div>

---

## 📌 Overview
**DetectAI** is a deepfake detection platform engineered to identify facial tampering, deepfakes, and synthetic manipulations in images and videos. 

### Why a Hybrid Architecture?
Most deepfake detectors rely solely on Convolutional Neural Networks (CNNs). While CNNs capture semantic facial inconsistencies (asymmetries, distorted teeth, warped eyes), GANs and diffusion algorithms also leave mathematical fingerprints invisible to humans. 

DetectAI combines:
1. **Deep Learning Branch (60% weight)**: Transfer learning with **EfficientNetB0** (pre-trained on ImageNet) to evaluate morphological facial features.
2. **Forensic Random Forest Branch (40% weight)**: A 200-tree ensemble analyzing **38 handcrafted forensic signals**:
   - **Error Level Analysis (ELA)**: Compression artifact discrepancies across pixel matrices.
   - **2D Fast Fourier Transform (FFT)**: High-frequency spectral energy depletion and concentric GAN halos.
   - **Laplacian Variance**: Boundary blur vs. synthetic over-smoothing.
   - **Local Binary Patterns (LBP)**: Micro-texture histograms.
   - **Color-Channel Moments**: Mean, variance, and skewness across RGB and HSV.
   - **Edge Density & Noise Residuals**: Sensor PRNU noise preservation.
3. **Temporal Video Pipeline**: Frame sampling, face localization, and majority-vote aggregation with an interactive timeline.
4. **Explainability Studio**: Real-time **Grad-CAM Saliency Maps** with opacity slider, **ELA difference maps**, and certified **PDF Forensic Inspection Reports**.

---

## 📂 Project Structure

```
detectai/
├── app.py                     # FastAPI backend server with full REST endpoints
├── config.py                  # Central configuration, paths, weights, and constants
├── features.py                # 38-dimensional forensic feature extraction engine
├── train_model.py             # Two-phase transfer learning trainer for CNN
├── train_rf.py                # Random forest forensic classifier trainer
├── test_app.py                # Full integration test suite
├── requirements.txt           # Pinned, compatible Python dependencies
├── history.db                 # SQLite persistent audit storage (auto-created)
├── sample_face.jpg            # Included sample portrait for instant testing
├── model/                     # Trained models and calibration curves
│   ├── model.h5               # Trained EfficientNetB0 CNN model weights
│   ├── rf_model.pkl           # Trained 200-tree Random Forest classifier
│   └── haarcascade_...xml     # Haar frontal face detector
├── static/
│   ├── css/
│   │   └── style.css          # Design system (Purple/Lavender/Lilac + Glassmorphism)
│   └── js/
│       ├── charts.js          # Chart.js visualization manager
│       └── main.js            # UI interactions, drag-and-drop, theme, video seek
├── templates/
│   └── index.html             # Accessible, semantic single-page application
├── uploads/                   # Secured temporary inference storage (auto-cleaned)
└── reports/                   # Generated PDF forensic audit reports
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager

### 2. Clone / Setup Project
```bash
cd detectai
```

### 3. Create and Activate a Virtual Environment
#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Web Application

Launch the server with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 🧪 Testing in Demo Mode vs. Production

DetectAI includes **automatic fallback detection**:
- If `model/model.h5` or `model/rf_model.pkl` are missing, the system gracefully activates **Demo Mode**.
- In Demo Mode, it prints a clear console notification and generates deterministic, plausible cryptographic results based on the file's SHA-256 hash.
- This allows you to test drag-and-drop, sliders, charts, Grad-CAM overlays, video timeline seeking, and PDF reports immediately.

---

## 🏋️ Training the Models

### 1. Training the Random Forest (`train_rf.py`)
To train using your own dataset, place images in `dataset/real/*.jpg` and `dataset/fake/*.jpg`:
```bash
python train_rf.py
```
To rapidly train a working model using synthetic baseline distributions:
```bash
python train_rf.py --synthetic
```
Outputs saved to `model/rf_model.pkl`, `model/rf_confusion_matrix.png`, and `model/rf_roc_curve.png`.

### 2. Training the CNN Branch (`train_model.py`)
To train the EfficientNetB0 CNN with data augmentation and two-phase fine-tuning:
```bash
python train_model.py
```
To compile and test the architecture rapidly without downloading large datasets:
```bash
python train_model.py --synthetic --epochs1 2 --epochs2 2
```
Outputs saved to `model/model.h5`, `model/training_curves.png`, and `model/confusion_matrix.png`.

---

## 🛠️ Automated Verification Suite

Run the full end-to-end integration test:
```bash
python test_app.py
```

This tests:
1. `GET /health` API status
2. `GET /` UI rendering
3. `POST /predict` face detection and error handling
4. `POST /predict` end-to-end inference on `sample_face.jpg`
5. `GET /report/{id}` ReportLab PDF generation
6. `GET /stats` and `GET /history` SQLite audit logging

---

## 🎨 Theme Tokens & Customization
- **Theme Palette**:
  - `--purple-900: #2e1065`
  - `--purple-700: #6d28d9`
  - `--purple-500: #8b5cf6`
  - `--lavender: #c4b5fd`
  - `--lilac: #e9d5ff`
  - `--bg-dark: #140a2b`
  - `--card: #1f1240`
  - `--verdict-real: #10b981` (Accessible emerald green)
  - `--verdict-fake: #ef4444` (Accessible ruby red)
- **Theme Toggle**: Switch between Dark and Light mode via the navbar button; the preference is saved in `localStorage`.

---

## ⚖️ Legal Disclaimer
*Analysis verdicts and confidence metrics produced by DetectAI are probabilistic indicators calculated by machine learning algorithms. They are designed for investigative triage and educational analysis, not judicial or legal proof.*
