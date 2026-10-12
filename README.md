# GenAI Fingerprint

GenAI Fingerprint (GenAIFP) is a local research prototype that estimates whether an image resembles one of the generator classes in its installed model or the real-camera class. It combines a pretrained DINOv2 visual embedding, spatial-frequency measurements, and a calibrated XGBoost classifier. The project includes a command-line interface, a local browser interface, and a Kaggle training workflow.

> **Disclaimer:** **The output is experimental. It can be wrong and is not evidence of an image's origin, not a forensic conclusion, and not a benchmark result.** The included model is a bounded research model with significant class imbalance in performance and weak rejection of unsupported inputs. Read the evaluation and limitations below before interpreting a prediction.

---

## Contents

- [Local Setup & Quick Start](#local-setup--quick-start)
  - [Prerequisites](#prerequisites)
  - [Option A: Python Virtual Environment (Recommended)](#option-a-python-virtual-environment-recommended)
  - [Option B: Docker / Container Setup](#option-b-docker--container-setup)
  - [Verify Your Installation](#verify-your-installation)
- [Using the System](#using-the-system)
  - [Local Web Interface](#local-web-interface)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [All CLI Commands](#all-cli-commands)
- [How the System Works](#how-the-system-works)
  - [Prediction Pipeline](#prediction-pipeline)
  - [Supported Output Classes](#supported-output-classes)
- [Project & Evaluation Metrics](#project--evaluation-metrics)
  - [System & Architecture Metrics](#system--architecture-metrics)
  - [Dataset & Partitioning Metrics](#dataset--partitioning-metrics)
  - [Benchmark & Attribution Metrics](#benchmark--attribution-metrics)
  - [Per-Class Held-Out Performance](#per-class-held-out-performance)
- [Advanced Workflows](#advanced-workflows)
  - [Retrain in Kaggle & Download Models](#retrain-in-kaggle--download-models)
  - [Synthetic Demo Mode](#synthetic-demo-mode)
  - [Running Automated Tests](#running-automated-tests)
- [Project Layout](#project-layout)
- [Dataset, Privacy, and Limitations](#dataset-privacy-and-limitations)

---

## Local Setup & Quick Start

You can run predictions immediately upon cloning the repository—the calibrated model bundle (`models/attribution_hybrid`) is tracked in Git, so you do not need to download training datasets or train a model locally.

### Prerequisites

- **Python**: Version 3.10 or newer (tested on 3.10–3.12).
- **System Libraries (Linux)**: OpenCV and Pillow require standard graphic runtime libraries:
  ```bash
  # Debian / Ubuntu / Mint
  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
  ```
- **Internet Access on First Run**: Required during initial setup to fetch packages, and on the first image analysis to download pretrained DINOv2 weights via `torch.hub`.
- **Hardware**: CPU inference is fully supported; a CUDA-capable GPU improves inference speed.

---

### Option A: Python Virtual Environment (Recommended)

Run all commands from the repository root so `main.py`, `src/genaifp/`, and `models/` resolve correctly:

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repository-url>
   cd genaifp
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # On Linux / macOS
   # .venv\Scripts\activate       # On Windows PowerShell
   ```

3. **Install core project dependencies**:
   `requirements.txt` installs all core libraries and Web UI server packages (`fastapi`, `uvicorn`, `python-multipart`):
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Install PyTorch and vision models**:
   PyTorch is kept separate so you can install the build optimal for your hardware:
   - **For CPU only (lightweight, ~200 MB)**:
     ```bash
     python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     python -m pip install timm
     ```
   - **For CUDA GPU (NVIDIA)**:
     ```bash
     python -m pip install torch torchvision timm
     ```

*(Alternative: You can also install in editable mode via optional extras: `pip install -e '.[web,dev]'` after installing your preferred PyTorch build).*

---

### Option B: Docker / Container Setup

If you prefer not to configure a local Python environment, you can run the full web application via Docker:

```bash
# Using Docker Compose
docker compose up --build

# Or using Docker directly
docker build -t genaifp .
docker run -p 8000:8000 genaifp
```

The web service will start and bind to `http://127.0.0.1:8000` with automated health checking.

---

### Verify Your Installation

Once installed, verify that everything is working properly:

#### 1. Analyze an image via CLI
Pass any PNG, JPEG, or WebP image to `main.py predict`:
```bash
python main.py predict --image_path path/to/your/image.jpg
```
*(On the first invocation, PyTorch will download the DINOv2 weights from `torch.hub` to your local cache).*

#### 2. Start the local Web UI
```bash
python main.py web
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. You should see a green "Evaluated model ready" status indicator.

#### 3. Run the automated tests
```bash
python -m pytest
```

---

## Using the System

### Local Web Interface

Launch the local web server:
```bash
python main.py web [--host <addr>] [--port <n>] [--model <path>]
```

- **URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000) (loopback only by default for safety).
- **Project guide**: [http://127.0.0.1:8000/about](http://127.0.0.1:8000/about).
- **Upload constraints**: Supports PNG, JPEG, and WebP up to 20 MB and 40 megapixels.
- **Privacy & Security**: Uploads are processed entirely in memory and never saved to disk. Cross-origin requests are rejected by default. For containerized or reverse-proxy deployments (e.g. Docker/Coolify), set `GENAIFP_ALLOW_REMOTE=1` and `--host 0.0.0.0`.

---

### Command-Line Interface (CLI)

#### `predict`
Attributing a single image:
```bash
python main.py predict --image_path <path> [--model_path <path>] [--crop_size <n>] [--output_json <path>]
```

| Option | Default | Description |
|---|---|---|
| `--image_path` | *required* | Path to the image file to analyze (PNG, JPEG, WebP) |
| `--model_path` | `models/attribution_hybrid` | Model bundle basename (without `.json` or `.pkl`) |
| `--crop_size` | `512` | Center-crop size in pixels; smaller images are reflection-padded |
| `--output_json` | *none* | Path to write full attribution scores, provenance, and OOD metrics as JSON |

*Shortcut: You can also pass a bare image path directly: `python main.py path/to/image.png`.*

---

### All CLI Commands

Run `python main.py [command] --help` for specific options:

| Command | Aliases | Description |
|---|---|---|
| `predict` | *image path* | Attribute an image to a generator class or `real_camera` |
| `web` | — | Start the local FastAPI web app |
| `build` | `build_dataset` | Build feature tables; requires `--demo` locally |
| `train` | `training` | Train an attribution model; requires `--demo-only` locally |
| `download` | `download_curated` | Display instructions for downloading Kaggle run artifacts |
| `download_shards` | `download_shard` | Display instructions for Kaggle source-data workflows |
| `help` | `-h`, `--help` | Display CLI command summary |

---

## How the System Works

### Prediction Pipeline

The system processes a 512 × 512 RGB crop through two complementary feature pipelines:

```
Input Image (RGB)
  │
  ├── Center crop & SRM 3x3 high-pass filter ──> 2D FFT Spectrum ──> PCA (128d) + 12 hand-crafted spectral metrics
  │
  └── Resized RGB crop ────────────────────────> DINOv2 ViT-S/14  ──> Visual embedding (384d)
                                                                         │
                                                                         ▼
                                                     Combined 524d Hybrid Feature Vector
                                                                         │
                                                                         ▼
                                                     Calibrated XGBoost Classifier + OOD Detector
                                                                         │
                                                                         ▼
                                                     Class Probabilities & Attribution Verdict
```

1. **Preprocessing**: Images are center-cropped to 512 × 512 RGB (smaller inputs are reflection-padded). The spectral branch converts the crop to luminance and applies an SRM high-pass filter to extract subtle residual noise.
2. **Visual features**: Pretrained DINOv2 (`vit_small_patch14_dinov2.lvd142m`) extracts a normalized 384-dimensional semantic embedding.
3. **Spatial-frequency features**: The residual image is transformed via 2D FFT into a log-power spectrum, projected into 128 dimensions via training-fitted PCA.
4. **Hand-crafted spectral metrics**: 12 scalar features capture peak-to-average power ratio, high-frequency spectral slope/intercept, directional variance, and energy across 8 radial frequency bands.
5. **Calibrated classification & OOD detection**: The combined 524-dimensional feature vector is passed to an XGBoost classifier with Platt/temperature probability calibration and an out-of-distribution (OOD) distance threshold.

### Supported Output Classes

The installed model classifies images across **12 target categories**:
- **Generators (11)**: `dalle2`, `dalle3`, `midjourney_v5_2`, `midjourney_v6_1`, `firefly_image2`, `firefly_image3`, `flux_dev`, `flux_schnell`, `ideogram_v1`, `ideogram_v2`, `imagen3`.
- **Real (1)**: `real_camera`.

Other generators or inputs outside these distributions are flagged as uncertain/OOD when detected.

---

## Project & Evaluation Metrics

The project tracks detailed metrics across system architecture, dataset partitioning, benchmark classification, and out-of-distribution (OOD) rejection.

### System & Architecture Metrics

| Metric | Specification / Measurement | Notes |
|---|---|---|
| **Input Crop Resolution** | 512 × 512 px (RGB) | Center crop; smaller images use reflection padding |
| **Combined Feature Vector** | **524 dimensions** | 384d visual + 128d spectral PCA + 12d frequency scalars |
| **Visual Embedding Backbone** | DINOv2 ViT-S/14 (`384d`) | Pretrained `vit_small_patch14_dinov2.lvd142m` via `torch.hub` |
| **Spectral Transform** | SRM 3×3 + 2D FFT + PCA (`128d`) | Captures high-frequency noise residuals |
| **Hand-Crafted Spectral Metrics** | 12 scalar measurements | Radial energy (8 bands), azimuthal spread, slope, peak-to-avg |
| **Attribution Classifier** | Calibrated XGBoost | Multi-class tree booster with Platt/temperature probability scaling |
| **Model Footprint on Disk** | ~72 MB total | XGBoost (`3.5 MB`), feature extractor PCA (`68 MB`), metadata (`~25 KB`) |
| **Inference Latency** | ~150–250 ms (GPU) / ~1–2 s (CPU) | Tested on NVIDIA Tesla T4 vs. multi-core x86_64 CPU |
| **Upload Safety Constraints** | Max 20 MB / 40 megapixels | Processed in-memory; zero disk writes for uploaded user images |
| **Test Suite Pass Rate** | **100% (85 / 85 passed)** | Unit, CLI, feature extraction, and API integration coverage |

### Dataset & Partitioning Metrics

Data provenance and splits from the latest evaluated Kaggle training run (*OwensLab CommunityForensics*):

| Metric | Value | Description |
|---|---:|---|
| **Configured Source Shards** | 3 shards | `HFCF_Commercial_0`, `HFCF_Commercial_1`, `HFCF_small_174` |
| **Target Engine Classes** | 12 classes | 11 generative AI models + `real_camera` |
| **Sample Cap per Class** | 100 images | Prevents class imbalance during training and calibration |
| **Total Clean Examples** | 1,200 images | Exactly 100 per target category |
| **Total Degraded OOD Examples** | 1,200 images | Paired degraded copies for out-of-domain rejection tuning |
| **Training Split** | 780 clean (735 prompt groups) | Disjoint prompt groups for fitting PCA and classifier |
| **Calibration Split** | 188 clean + 188 degraded OOD | Disjoint prompt groups for probability & OOD threshold fitting |
| **Held-Out Test Split** | 232 clean + 232 degraded OOD | Disjoint prompt groups (224 groups) for unbiased evaluation |
| **Held-Out Test Image Archive** | 120 curated images | 10 distinct holdouts per class exported to `samples/` |

### Benchmark & Attribution Metrics

Evaluated on the internal held-out test split (232 known-class images + 232 degraded OOD images):

| Benchmark Metric | Result | Target / Context |
|---|---:|---|
| **Known-Class Held-Out Accuracy** | **58.2%** (135 / 232) | Multi-class attribution across all 12 classes |
| **Known-Class Macro-F1** | **0.556** | Unweighted mean of per-class F1 scores |
| **Calibration Temperature ($T$)** | `0.935` | Platt probability scaling parameter |
| **OOD Rejection Threshold** | `0.677` | Score boundary for flagging unsupported inputs |
| **Calibration OOD AUROC** | **0.723** | Area Under ROC Curve on calibration holdouts |
| **Calibration False Reject Rate (FRR)**| **4.8%** | Fraction of valid target images incorrectly flagged as OOD |
| **Held-Out Degraded OOD Recall** | **15.5%** | Detection rate of degraded variants on test holdout |

### Per-Class Held-Out Performance

Performance breakdown on the 232 held-out test images across all 12 target classes:

| Class | Test Samples | Accuracy | Macro F1 | Status |
|---|---:|---:|---:|---|
| **Real camera** | 18 | 94.4% | **0.850** | Strong separation |
| **Firefly Image 3** | 21 | 85.7% | **0.837** | Strong separation |
| **Midjourney V6.1** | 17 | 76.5% | **0.765** | Moderate-strong separation |
| **Firefly Image 2** | 20 | 80.0% | **0.744** | Moderate-strong separation |
| **DALL·E 2** | 18 | 94.4% | **0.708** | Distinct spectral fingerprint |
| **Midjourney V5.2** | 21 | 76.2% | **0.615** | Moderate separation |
| **FLUX dev** | 22 | 59.1% | **0.578** | Moderate separation |
| **DALL·E 3** | 20 | 30.0% | **0.387** | High confusion with modern diffusion |
| **Imagen 3** | 22 | 22.7% | **0.357** | Challenging attribution |
| **FLUX schnell** | 14 | 42.9% | **0.343** | Confused with FLUX dev |
| **Ideogram V1** | 21 | 23.8% | **0.270** | Challenging attribution |
| **Ideogram V2** | 18 | 16.7% | **0.214** | High variance / low sample F1 |

*Note: These metrics reflect internal prompt-grouped research evaluation on CommunityForensics shards, not an official benchmark score. See [Dataset, Privacy, and Limitations](#dataset-privacy-and-limitations).*

---

## Advanced Workflows

### Retrain in Kaggle & Download Models

Because extracting visual embeddings and FFT features from hundreds of gigabytes of Parquet shards requires significant GPU and memory resources, real-data model training runs on **Kaggle** rather than locally.

#### 1. Configure Kaggle CLI Credentials
Install the CLI and configure your Kaggle API key:
```bash
python -m pip install kaggle
# Place your kaggle.json in ~/.kaggle/ (Linux/macOS) or %USERPROFILE%\.kaggle\ (Windows)
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. Assemble and Push the Training Notebook
The notebook is built from modular source scripts into `kaggle/communityforensics_train.ipynb`:
```bash
# Rebuild the standalone training notebook
python kaggle/build_notebook.py

# Push and trigger remote execution
kaggle kernels push -p kaggle
```
*(Kernel parameters such as GPU acceleration and internet access are configured in [`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json)).*

#### 3. Monitor Kernel Status and Logs
Monitor the remote job while it executes on a GPU instance:
```bash
# Check running status
kaggle kernels status nahilr/genaifp-communityforensics-train

# Stream live execution logs
kaggle kernels logs nahilr/genaifp-communityforensics-train
```

#### 4. Download Exported Artifacts
Once execution completes, download the three packaged ZIP archives (`genaifp-model-bundle.zip`, `genaifp-test-images.zip`, `genaifp-reports.zip`):
```bash
# Download using the project helper (with retry & validation):
python kaggle/download_outputs.py

# Or download directly using the Kaggle CLI:
kaggle kernels output nahilr/genaifp-communityforensics-train -p kaggle-output/downloads/
```

#### 5. Install Artifacts into the Local Repository
Install the unpacked files into their target directories:
```bash
python kaggle/install_outputs.py
```
This automatically updates:
- [`models/`](models/): Installs the XGBoost model weights, calibration metadata, and feature extractor transform.
- [`reports/`](reports/): Installs evaluation summaries, feature importances, and provenance reports.
- `samples/communityforensics-test/`: Installs 120 held-out test images (10 per class) with their provenance CSV.

### Synthetic Demo Mode

You can test dataset building and training locally using procedural class-coded patterns without downloading real data:

```bash
# Build synthetic feature tables
python main.py build --demo \
  --output data/demo/features_demo.parquet \
  --demo_samples_per_class 100

# Train a synthetic demo model (marked as demo-only)
python main.py train \
  --input data/demo/features_demo.parquet \
  --model_out models/demo_attribution \
  --report_dir reports/demo \
  --demo-only
```

*Note: Synthetic models are procedurally generated and are intentionally rejected by production prediction commands.*

### Running Automated Tests

Run the test suite from the repository root:
```bash
python -m pytest
```
Tests cover preprocessing, spectral FFT extraction, visual embeddings, calibration logic, CLI routing, upload safety, and web endpoints.

---

## Project Layout

```text
.
├── main.py                     # Local CLI entrypoint (predict, web, build, train)
├── Dockerfile                  # Production container definition (CPU-optimized)
├── docker-compose.yml          # Container orchestration configuration
├── requirements.txt            # Core and web Python dependencies
├── pyproject.toml              # Packaging metadata and optional dependency groups
├── src/genaifp/                # Core Python package
│   ├── classifier.py           # XGBoost attribution model, calibration & OOD scoring
│   ├── feature_extractor.py    # Hybrid PCA & scalar feature orchestration
│   ├── spectral_engine.py      # SRM filtering, 2D FFT, radial/azimuthal metrics
│   ├── visual_engine.py        # DINOv2 visual embedding extractor
│   ├── preprocessor.py         # Center cropping, padding, luminance conversion
│   ├── web.py                  # Local FastAPI web application backend
│   └── web/                    # Static browser UI assets (HTML, CSS, JS)
├── models/                     # Installed model bundle (tracked in git)
│   ├── attribution_hybrid.json # Calibrated XGBoost model weights
│   ├── attribution_hybrid.pkl  # Calibration temperature & OOD detector
│   ├── attribution_hybrid.manifest.json # Provenance, class labels, preprocessing spec
│   └── feature_extractor.pkl   # Fitted spectral PCA & scaling pipeline
├── reports/                    # Kaggle evaluation and provenance reports
├── kaggle/                     # Standalone Kaggle training and packaging scripts
├── samples/                    # Held-out test samples and demo reference images
├── tests/                      # Automated test suite (pytest)
├── AGENTS.md                   # Repository guidelines for AI coding agents
└── AGENT_LOG.md                # Dated chronological development history
```

---

## Dataset, Privacy, and Limitations

- **Dataset & Attribution**: Real-data training utilizes OwensLab's *CommunityForensics* and *CommunityForensics-Small* datasets. Please review upstream dataset licenses and citation guidelines.
- **Privacy**: The local web server never writes uploaded images to disk; all analysis is performed in memory.
- **Experimental Scope**: The model is trained on specific versions of 11 generator classes and real photography. Image compression, social-media re-encoding, resizing, or newer model versions can degrade accuracy.
- **Weak OOD Rejection**: Current out-of-distribution calibration catches approximately 15.5% of degraded out-of-domain images. Unknown generator models may receive high-confidence false attribution.
- **Uneven Performance**: F1 performance varies significantly across classes (from 0.21 on Ideogram V2 to 0.85 on real photos).
- **Not Forensic Evidence**: Predictions are research estimates and must **never** be interpreted as legal or conclusive proof of image provenance.
