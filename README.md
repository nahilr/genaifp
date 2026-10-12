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
  - [High-Level Architecture (Simplified)](#high-level-architecture-simplified)
  - [Detailed Engineering Pipeline (Full Technical Specification)](#detailed-engineering-pipeline-full-technical-specification)
  - [The 524-Dimensional Hybrid Feature Vector](#the-524-dimensional-hybrid-feature-vector)
  - [Machine Learning Models Employed](#machine-learning-models-employed)
  - [Step-by-Step Prediction Pipeline](#step-by-step-prediction-pipeline)
  - [Why a Hybrid Architecture?](#why-a-hybrid-architecture)
  - [Design Rationale: Why Center Crop?](#design-rationale-why-center-crop)
  - [Deep Dive: The 12 Hand-Crafted Frequency Metrics](#deep-dive-the-12-hand-crafted-frequency-metrics)
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
  - [Model Bundle & Artifact Specification (models/)](#model-bundle--artifact-specification-models)
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
# Production / Reverse Proxy (Traefik / Coolify)
# Uses exposed container port 8000 without binding to a host port:
docker compose up --build -d

# Direct container run with host port mapping (e.g. for local testing):
docker build -t genaifp .
docker run -p 8000:8000 genaifp
```

The web service exposes port `8000` internally with automated health checking at `/api/health`. For reverse-proxy deployments (Coolify, Traefik, Nginx), route your domain or subdomain directly to internal port `8000`.

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

### High-Level Architecture (Simplified)

This high-level conceptual diagram illustrates the two-track design at a glance:

```text
                         ┌─────────────────────────────┐
                         │      Input Image (RGB)      │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │  Track A: Look & Feel │                     │  Track B: Pixel Math  │
     │  (Macro Visual Style) │                     │  (Microscopic Noise)  │
     ├───────────────────────┤                     ├───────────────────────┤
     │ • DINOv2 Transformer  │                     │ • 3×3 High-Pass SRM   │
     │ • Lighting & Textures │                     │ • 2D Fourier (FFT)    │
     │ • Compression-proof   │                     │ • Periodic Grid Spikes│
     │   (384 dimensions)    │                     │   (140 dimensions)    │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  524-Feature Hybrid Vector  │
                         │     (Fused Fingerprint)     │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   XGBoost Decision Engine   │
                         │    & Temperature Scaling    │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │    Safety / OOD Detector    │
                         │ (Flags Unsupported Models)  │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │     Attribution Verdict     │
                         │ (e.g. Midjourney V6.1: 94%) │
                         └─────────────────────────────┘
```

---

### Detailed Engineering Pipeline (Full Technical Specification)

This comprehensive engineering schematic maps every mathematical transformation, matrix operation, tensor normalization, and decision stage in the system:

```text
                                              Input Image (RGB)
                                                      │
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
                     [Visual Branch]                                 [Spectral Branch]
                              │                                               │
                   Center crop to 512×512                          Center crop to 512×512
                              │                                               │
                  Bicubic resize to 224×224                     Convert to Luminance (Y)
                              │                                  Y = 0.299R + 0.587G + 0.114B
                              ▼                                               │
                    ImageNet Normalization                                    ▼
                   μ=[0.485,0.456,0.406]                           SRM 3×3 High-Pass Filter
                   σ=[0.229,0.224,0.225]                           [-1  2 -1; 2 -4  2; -1  2 -1]
                              │                                    (Extracts Noise Residual)
                              ▼                                               │
                    DINOv2 ViT-S/14 Model                                     ▼
              (vit_small_patch14_dinov2.lvd142m)                      2D FFT Fast Fourier
                              │                                      fftshift(fft2(Residual))
                              ▼                                               │
                    L2 Normalization                                          ▼
                      v / ||v||_2                                     Log-Power Spectrum
                              │                                      LogPower = ln(1 + |F|^2)
                              ▼                                               │
                     384 Visual Features                                      ├─── Flatten Upper Triangle (131,328 values)
                                                                              │         │
                                                                              │         ▼
                                                                              │    PCA Projection (StandardScaler + PCA)
                                                                              │    (Fitted on 128 Components)
                                                                              │         │
                                                                              │         ▼
                                                                              │    128 Spectral PCA Features
                                                                              │
                                                                              └─── Physical Scalar Analysis
                                                                                        │
                                                                                        ├─── RAPSD (Radial Mean Power)
                                                                                        │    PASR = max(P) / mean(P)
                                                                                        │    Decay Slope (α) & Intercept (β)
                                                                                        │    8 Sub-Band Energies (E_1 to E_8)
                                                                                        │
                                                                                        └─── Azimuthal Grid (4 Quadrants)
                                                                                             Directional Variance across Sectors
                                                                                                │
                                                                                                ▼
                                                                                           12 Handcrafted Frequency Metrics
                              │                                               │
                              └───────────────────────┬───────────────────────┘
                                                      ▼
                                         Concatenate All Features
                                     [524-Dimensional Hybrid Vector]
                                      x = [v_DINO(384) || p_PCA(128) || s_scalar(12)]
                                                      │
                                                      ▼
                                           Calibrated XGBoost Tree
                                         (max_depth=5, n_est=300-500)
                                                      │
                                                      ▼
                                         Raw Logits Margins z_c
                                                      │
                                                      ▼
                                         Platt Temperature Scaling
                                         P(y=c) = softmax(z_c / T)
                                                (T = 0.935)
                                                      │
                                                      ▼
                                         Multi-Signal OOD Gating
                                         (Logistic Regression Detector)
                                         ├── Top Confidence: max_c P(y=c)
                                         ├── Shannon Entropy: H = -sum(p ln p) / ln(K)
                                         ├── Visual Centroid Cosine Similarity: max_c (v · μ_c)
                                         └── Free Energy: E(x) = -T ln sum(exp(z_c/T))
                                                      │
                                                      ▼
                                         Threshold Check: Score > 0.677?
                                         ├── True  ──> Status: UNCERTAIN (OOD)
                                         └── False ──> Status: CONFIDENT
                                                      │
                                                      ▼
                                     Final Forensic Verdict & JSON Payload
                                     (Class Verdict, Probability Spread, Provenance)
```

---

### The 524-Dimensional Hybrid Feature Vector

The fused feature vector represents an image across three distinct, complementary forensic layers:

1. **Features 0 to 383 (384 dimensions): DINOv2 Deep Visual Embeddings**
   - Extracted by [`VisualEngine`](src/genaifp/visual_engine.py).
   - Generated by a frozen Vision Transformer (`vit_small_patch14_dinov2.lvd142m`) pretrained via self-supervised learning on 142M images.
   - Captures high-level macro-rendering habits, color-grading biases, lighting distribution, micro-contrast, and diffusion textures that remain invariant under JPEG compression and spatial downscaling.
2. **Features 384 to 511 (128 dimensions): Spectral PCA Projection**
   - Extracted by [`FeatureExtractor`](src/genaifp/feature_extractor.py).
   - Compresses the $512 \times 512$ 2D Fast Fourier Transform (FFT) log-power spectrum's upper triangle (131,328 unique frequency bins) down to 128 principal components using a training-fitted `StandardScaler` and `PCA`.
   - Captures holistic, high-frequency spatial noise patterns and residual grid geometry left behind by neural network image upsamplers.
3. **Features 512 to 523 (12 dimensions): Handcrafted Physical Metrics**
   - Extracted by [`FeatureExtractor.extract_12_scalar()`](src/genaifp/feature_extractor.py).
   - Domain-engineered signal measurements grounded in digital image processing and natural image statistics:
     - **PASR (`pasr`)**: Peak-to-Average Spectral Ratio detecting sharp periodic frequency spikes.
     - **Spectral Slope ($\alpha$) & Intercept ($\beta$)**: Log-log linear regression assessing compliance with the natural scene $1/f^\gamma$ power law.
     - **Sub-Band Energies ($E_1$ to $E_8$)**: 8 concentric radial frequency bands acting as an octave-band graphic equalizer for spatial noise.
     - **Directional Variance (`directional_var`)**: 4-quadrant azimuthal variance detecting rectangular grid biases vs. isotropic optical lens noise.

---

### Machine Learning Models Employed

GenAIFP combines an ensemble of five specialized machine learning and statistical models operating in an inference chain:

| Model Component | Framework & Architecture | Parameters / Spec | Exact Operational Role |
|---|---|---|---|
| **1. Visual Feature Extractor** | **DINOv2 ViT-S/14** (`timm` / `torch.hub`) | 21M parameters, frozen weights | Extracts dense 384d visual embeddings capturing macro-rendering style, lighting, and diffusion micro-contrast textures. Invariant down to JPEG QF 65 and spatial resizing. |
| **2. Spectral Compressor** | **StandardScaler + PCA(128)** (`scikit-learn`) | Matrix shape $(128 \times 131,328)$ | Compresses the $131,328$ frequency bins of the 2D FFT log-power upper triangle into 128 orthogonal components capturing high-frequency VAE upsampler lattice noise. |
| **3. Attribution Classifier** | **XGBoost Tree Booster** (`xgboost.XGBClassifier`) | 300–500 trees, depth 5, histogram method | Multi-class decision engine evaluating the fused 524d hybrid vector. Emits raw margin scores (logits) across all 12 target classes. |
| **4. Probability Calibrator** | **Platt Temperature Scaling** (`scipy.optimize`) | Empirical scalar ($T \approx 0.831$) | Divides raw XGBoost logits by $T$ before Softmax to eliminate overconfidence and calibrate output probabilities to true empirical accuracy. |
| **5. Safety Guardrail** | **Logistic Regression Pipeline** (`scikit-learn`) | Balanced class weights, 4 input features | Jointly monitors top confidence, Shannon entropy, DINOv2 centroid cosine similarity, and free energy score to flag unsupported out-of-distribution inputs. |

---

### Step-by-Step Prediction Pipeline

Here is the complete end-to-end lifecycle of an image through the prediction pipeline:

#### Step 1: Ingestion & Safety Verification
- **Module**: [`src/genaifp/web.py`](src/genaifp/web.py) / [`main.py`](main.py)
- **Validation**: Uploads are restricted to PNG, JPEG, and WebP formats up to 20 MB and 40 megapixels (guarding against decompression bomb attacks).
- **Privacy & Security**: Images are streamed strictly into in-memory byte buffers (`io.BytesIO`) and converted directly into NumPy pixel arrays. Uploaded files are **never written to disk**.

#### Step 2: Preprocessing & Luminance Extraction
- **Module**: [`src/genaifp/preprocessor.py`](src/genaifp/preprocessor.py)
- **Spatial Alignment**: The image is cropped to an uninterpolated $512 \times 512$ center crop. Images smaller than 512 pixels in either dimension are reflection-padded along the borders.
- **Luminance Conversion**: The RGB crop is converted to grayscale luminance ($Y$) using standard ITU-R BT.601 coefficients:
  $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

#### Step 3: High-Pass Noise Residual Extraction
- **Module**: [`src/genaifp/preprocessor.py:apply_srm_highpass()`](src/genaifp/preprocessor.py)
- **Filter**: Convolves the luminance array with a $3 \times 3$ Spatial Rich Model (SRM) high-pass kernel:
  $$K_{\text{SRM}} = \begin{bmatrix} -1 & 2 & -1 \\ 2 & -4 & 2 \\ -1 & 2 & -1 \end{bmatrix}$$
- **Forensic Function**: Because the filter elements sum to zero ($\sum K = 0$), macroscopic image content (faces, skies, objects) is subtracted out, isolating subtle residual sensor noise and generative synthesis artifacts.

#### Step 4: 2D Spectral Decomposition & PCA
- **Module**: [`src/genaifp/spectral_engine.py`](src/genaifp/spectral_engine.py)
- **Frequency Transformation**: Computes the 2D Fast Fourier Transform (`scipy.fft.fft2`), shifts the DC zero-frequency component to the center (`fftshift`), and computes the log-power spectrum:
  $$\text{LogPower}(u, v) = \ln\left(1 + |F(u, v)|^2\right)$$
- **Compression & Scalars**: Extracts the Hermitian-symmetric upper triangle (131,328 values) and projects it into 128 dimensions via training-fitted PCA. Simultaneously computes the Radially Averaged Power Spectral Density (RAPSD) and 4-quadrant azimuthal profiles to extract the 12 physical scalar metrics.

#### Step 5: Visual Foundation Feature Extraction
- **Module**: [`src/genaifp/visual_engine.py`](src/genaifp/visual_engine.py)
- **Model**: Passes the $512 \times 512$ RGB crop through a bicubic resize to $224 \times 224$, standard ImageNet normalization, and a forward pass through frozen `vit_small_patch14_dinov2.lvd142m`.
- **Normalization**: Applies $L_2$ vector normalization ($\|v\|_2 = 1$) to generate the final 384-dimensional visual embedding.

#### Step 6: Feature Fusion & Tabular Tree Attribution
- **Module**: [`src/genaifp/feature_extractor.py:extract_hybrid()`](src/genaifp/feature_extractor.py) & [`src/genaifp/classifier.py`](src/genaifp/classifier.py)
- **Concatenation**: Fuses the vectors into a single 524-dimensional array:
  $$\mathbf{x}_{\text{hybrid}} = \begin{bmatrix} \mathbf{v}_{\text{DINO}} (384) & \mathbf{p}_{\text{PCA}} (128) & \mathbf{s}_{\text{scalar}} (12) \end{bmatrix}$$
- **Classification**: An XGBoost gradient-boosted decision tree evaluates the tabular vector, outputting raw class margin scores (logits).

#### Step 7: Probability Calibration (Temperature Scaling)
- **Module**: [`src/genaifp/classifier.py:predict_proba()`](src/genaifp/classifier.py)
- **Platt Scaling**: To correct for raw model overconfidence, margin scores $z_c$ are divided by an empirical temperature parameter ($T = 0.935$, fitted via Negative Log-Likelihood minimization on disjoint calibration holdouts) before softmax:
  $$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

#### Step 8: Multi-Signal Out-of-Distribution (OOD) Gating
- **Module**: [`src/genaifp/classifier.py:calibrate_ood()`](src/genaifp/classifier.py)
- **Uncertainty Features**: An auxiliary Logistic Regression detector evaluates four joint signals:
  1. *Top Class Confidence*: $\max_c P(y = c \mid \mathbf{x})$.
  2. *Normalized Shannon Entropy*: $H_{\text{norm}} = -\sum p_i \ln(p_i) / \ln(K)$, measuring probability dispersion across classes.
  3. *Visual Centroid Similarity*: Maximum cosine similarity against training class centroids in DINOv2 visual space ($\max_c \mathbf{v} \cdot \boldsymbol{\mu}_c$).
  4. *Free Energy Score*: $E(\mathbf{x}) = -T \ln \sum_c \exp(z_c / T)$.
- **Gating**: If the predicted OOD probability exceeds the calibrated threshold (**$0.677$**), the verdict is flagged as **`status: uncertain (OOD)`**.

#### Step 9: Final Verdict Output & Provenance Tracking
- **Output**: Returns a complete forensic payload including the winning engine class, confidence, calibrated probability distribution across all 12 classes, OOD warning flags, physical spectral metrics, and dataset provenance metadata (source shard, prompt grouping hash, and dataset digest).

### Why a Hybrid Architecture?

Early generative image attribution research focused exclusively on spatial-frequency analysis (2D Fourier analysis of residual noise) to identify periodic lattice artifacts left behind by neural upsampling layers (such as transposed convolutions in Variational Autoencoders / VAEs). While effective on pristine, uncompressed PNG outputs, **pure Fourier detection degrades significantly on real-world web images**:
- **JPEG Compression Quantization**: Standard JPEG encoding compresses images via $8 \times 8$ Discrete Cosine Transform (DCT) quantization grids. This creates artificial periodic boundaries that Fourier analysis can confound with generative VAE upsampling artifacts.
- **Resampling and Downscaling**: Web uploads, social media platforms, and thumbnails are routinely resized. Downscaling filters act as aggressive low-pass smoothing operations that attenuate high-frequency pixel lattices.

To resolve this limitation, GenAIFP combines two complementary representations:
1. **Track A — DINOv2 Foundation Vision Features (384d)**: A frozen Vision Transformer (`vit_small_patch14_dinov2.lvd142m`) trained with self-supervision on 142M images. It extracts high-level semantic, lighting, and rendering habits, color-grading biases, and diffusion micro-contrast textures that remain invariant under JPEG compression (down to QF 65) and spatial downscaling.
2. **Track B — High-Frequency Fourier Spectral Features (140d)**: A 3×3 SRM high-pass filter isolates residual noise, followed by 2D Fast Fourier Transformation. This captures fine-grained physical lattice spikes, energy distributions across frequency bands, and spatial-frequency decay rates (128d PCA + 12 physical scalars).
3. **Calibrated Tabular Fusion (524d)**: An XGBoost gradient-boosted decision tree fuses both feature sets. The hybrid architecture achieves robust, multi-class attribution on clean and compressed inputs with low latency (~85 ms on CPU).

### Design Rationale: Why Center Crop?

Rather than resizing the entire image down to $512 \times 512$, the pipeline extracts a direct **$512 \times 512$ center crop** (with reflection padding only when an input dimension is smaller than 512):

1. **Preserving Native Nyquist Frequencies (No Interpolation Confound)**:
   - Resizing an image (e.g., $1024 \times 1024 \to 512 \times 512$) blends neighboring pixels using bilinear or bicubic interpolation.
   - This averaging operation attenuates high-frequency noise and alters spectral energy ratios by up to 74%. Worse, interpolation introduces its own mathematical sinc/ringing artifacts. A model trained on resized images risks learning the signature of the resizing algorithm rather than the generative engine.
   - Center cropping samples pixels at a raw 1:1 scale without interpolation, preserving native Nyquist-limit frequency characteristics and microscopic decoder lattices.
2. **Avoiding Watermarks, UI Badges, and Lens Vignetting**:
   - Web downloads and synthetic engines often place watermarks, signatures, or metadata badges in the outer corners or borders (e.g., DALL·E corner marks or letterboxing).
   - In real-camera photographs, physical lenses suffer from optical aberrations, light falloff (vignetting), and chromatic distortion near the extreme image margins.
   - The center crop samples the cleanest optical and synthetic region, away from peripheral artifacts.
3. **Consistent Geometric Grid for 2D FFT and PCA**:
   - The 2D Fourier Transform and PCA projection require a fixed, square coordinate matrix. Center cropping guarantees that every radial ring and azimuthal sector represents the exact same physical spatial frequency across all inputs, regardless of original image aspect ratio or resolution.
4. **Deterministic and Reproducible Forensics**:
   - Unlike random cropping used in data augmentation, center cropping is 100% deterministic: analyzing the same image repeatedly produces identical feature vectors and calibration scores.

### Deep Dive: The 12 Hand-Crafted Frequency Metrics

While PCA projects the 2D log-power spectrum into 128 general components, PCA is an unsupervised statistical compression technique that does not explicitly enforce physical laws of digital signal processing. To complement PCA, GenAIFP computes **12 domain-engineered scalar metrics** derived from Radially Averaged Power Spectral Density (RAPSD) and azimuthal quadrant profiles:

```
                                  Center of Spectrum (Low Frequencies)
                                             │
                                     [ Ring 1: E_1 ]   (Coarse shapes & structures)
                                     [ Ring 2: E_2 ]
                                     [ Ring 3: E_3 ] ◄── #1 Most Important Feature in XGBoost
                                     [ Ring 4: E_4 ]
                                     [ Ring 5: E_5 ]
                                     [ Ring 6: E_6 ]
                                     [ Ring 7: E_7 ]
                                     [ Ring 8: E_8 ]   (Microscopic pixel-level noise)
                                             │
                                    Outer Edge (Nyquist Frequency)
```

1. **PASR — Peak-to-Average Spectral Ratio (`pasr`)**:
   - **Formula**: $\text{PASR} = \frac{\max(P_{\text{norm}})}{\text{mean}(P_{\text{norm}})}$ where $P_{\text{norm}}$ is the normalized radial power spectrum.
   - **Forensic Role**: Natural camera noise spreads power continuously across high frequencies. Neural upsampling layers (e.g., transposed convolutions or pixel-shufflers) induce sharp periodic harmonics, manifesting as distinct spectral peaks that elevate PASR.
2. **Spectral Decay Slope ($\alpha$) (`alpha`)**:
   - **Formula**: Fitted exponent from the log-log linear regression $\ln P(r) = -\alpha \ln r + \beta$ over the upper 50% of radial frequencies.
   - **Forensic Role**: Natural scenes adhere to the classical $1/f^\gamma$ power law of natural image statistics. Neural network decoders alter this decay rate, either artificially flattening the high-frequency slope or over-smoothing the fine textures.
3. **Spectral Intercept ($\beta$) (`beta`)**:
   - **Formula**: The intercept term from the high-frequency log-log linear fit.
   - **Forensic Role**: Quantifies the baseline energy offset in the decay spectrum, establishing the absolute noise floor of the sensor or generative decoder.
4. **Sub-Band Energies ($E_1$ to $E_8$) (`E_1` – `E_8`)**:
   - **Formula**: Integrals of normalized spectral power across 8 concentric radial bands of equal width, from the lowest non-DC frequency to the inscribed radius ($r_{\max} = 255$).
   - **Forensic Role**: Acts as an 8-band graphic equalizer for spatial noise. Different generative architectures deposit artifacts in characteristic frequency octaves depending on their latent-to-pixel upsampling factor (e.g., $8\times$ vs. $16\times$).
5. **Directional Variance (`directional_var`)**:
   - **Formula**: The variance across 4 equal-angle azimuthal quadrants ($90^\circ$ sectors: $0^\circ\text{--}90^\circ$, $90^\circ\text{--}180^\circ$, $180^\circ\text{--}270^\circ$, $270^\circ\text{--}360^\circ$) averaged over all radial rings.
   - **Forensic Role**: Real optical lenses are circular, producing roughly isotropic (directionally uniform) noise. Neural network architectures operate over rectangular Cartesian coordinate grids ($x$ and $y$ kernel strides), frequently introducing directional bias along the horizontal or vertical axes.

#### Feature Importance in the Trained Model
Across the full 524-dimensional feature vector, these 12 physical measurements account for **8.0% of total tree gain**, yet individual metrics serve as the primary decision anchors in the trained model (`reports/feature_importance.csv`):
- **Rank 1**: `SubBand_E3` (Importance: `0.0153`) — Highest across all 524 features.
- **Rank 2**: `PASR` (Importance: `0.0125`) — Second highest across all features.
- **Rank 3**: `Slope_Alpha` (Importance: `0.0095`) — Third highest across all features.
- **Rank 4**: `Intercept_Beta` (Importance: `0.0088`) — Fourth highest across all features.

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
| **Sample Cap per Class** | 500 images | Prevents class imbalance during training and calibration |
| **Total Clean Examples** | 6,000 images | 500 per target category |
| **Total Degraded OOD Examples** | 6,000 images | Paired degraded copies for out-of-domain rejection tuning |
| **Training Split** | 3,870 clean (3,173 prompt groups) | Disjoint prompt groups for fitting PCA and classifier |
| **Calibration Split** | 935 clean + 935 degraded OOD | Disjoint prompt groups (780 groups) for probability & OOD threshold fitting |
| **Held-Out Test Split** | 1,195 clean + 1,195 degraded OOD | Disjoint prompt groups (974 groups) for unbiased evaluation |
| **Held-Out Test Image Archive** | 120 curated images | 10 distinct holdouts per class exported to `samples/` |

### Benchmark & Attribution Metrics

Evaluated on the internal held-out test split (1,195 known-class images + 1,195 degraded OOD images):

| Benchmark Metric | Result | Target / Context |
|---|---:|---|
| **Known-Class Held-Out Accuracy** | **72.4%** (865 / 1,195) | Multi-class attribution across all 12 classes |
| **Known-Class Macro-F1** | **0.708** | Unweighted mean of per-class F1 scores |
| **Calibration Temperature ($T$)** | `0.831` | Platt probability scaling parameter |
| **OOD Rejection Threshold** | `0.751` | Score boundary for flagging unsupported inputs |
| **Calibration OOD AUROC** | **0.751** | Area Under ROC Curve on calibration holdouts |
| **Calibration False Reject Rate (FRR)**| **4.9%** | Fraction of valid target images incorrectly flagged as OOD |
| **Held-Out Degraded OOD Recall** | **24.1%** | Detection rate of degraded variants on test holdout |

### Per-Class Held-Out Performance

Performance breakdown on the 1,195 held-out test images across all 12 target classes:

| Class | Test Samples | Accuracy | Macro F1 | Status |
|---|---:|---:|---:|---|
| **Real camera** | 95 | 93.7% | **0.942** | Strong separation |
| **Midjourney V6.1** | 135 | 86.7% | **0.897** | Strong separation |
| **Firefly Image 3** | 96 | 90.6% | **0.853** | Strong separation |
| **Firefly Image 2** | 95 | 84.2% | **0.829** | Strong separation |
| **Midjourney V5.2** | 113 | 86.7% | **0.824** | Strong separation |
| **DALL·E 2** | 91 | 93.4% | **0.780** | Distinct spectral fingerprint |
| **DALL·E 3** | 99 | 70.7% | **0.741** | Moderate-strong separation |
| **Imagen 3** | 98 | 60.2% | **0.638** | Moderate separation |
| **FLUX schnell** | 111 | 55.0% | **0.587** | Moderate separation |
| **FLUX dev** | 98 | 52.0% | **0.537** | Moderate separation |
| **Ideogram V2** | 95 | 35.8% | **0.450** | Challenging attribution |
| **Ideogram V1** | 69 | 49.3% | **0.415** | Challenging attribution |

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
kaggle kernels status nahilr/genaifp

# Stream live execution logs
kaggle kernels logs nahilr/genaifp
```

#### 4. Download Exported Artifacts
Once execution completes, download the three packaged ZIP archives (`genaifp-model-bundle.zip`, `genaifp-test-images.zip`, `genaifp-reports.zip`):
```bash
# Download using the project helper (with retry & validation):
python kaggle/download_outputs.py

# Or download directly using the Kaggle CLI:
kaggle kernels output nahilr/genaifp -p kaggle-output/downloads/
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
genaifp/
├── main.py                     # Unified CLI entrypoint (predict, web, build, train)
├── Dockerfile                  # Production container definition (CPU-optimized)
├── requirements.txt            # Core dependencies (PyTorch, timm, xgboost, fastapi, etc.)
├── pyproject.toml              # Packaging metadata and optional dependency groups
│
├── src/genaifp/                # Core Python package
│   ├── preprocessor.py         # 512×512 center-cropping, luminance conversion, SRM filter
│   ├── spectral_engine.py      # 2D FFT, power spectrum, RAPSD, azimuthal grids
│   ├── visual_engine.py        # DINOv2 ViT-S/14 feature extractor
│   ├── feature_extractor.py    # Orchestrates PCA fitting and 524d hybrid feature fusion
│   ├── classifier.py           # Calibrated XGBoost classifier + Logistic Regression OOD detector
│   ├── web.py                  # Local FastAPI web server
│   └── web/                    # Frontend UI: index.html, about.html, app.js, style.css
│
├── kaggle/                     # Standalone cloud training workflow
│   ├── build_notebook.py       # Assembles modular scripts into communityforensics_train.ipynb
│   ├── data_access.py          # Hugging Face Parquet streaming downloader
│   ├── dataset_build.py        # Remote feature extractor runner
│   ├── training.py             # Prompt-grouped training, calibration, and metrics exporter
│   ├── package_outputs.py      # Packs model, test images, and reports into ZIPs
│   ├── download_outputs.py     # Authenticated downloader via Kaggle API
│   └── install_outputs.py      # Local installer that unpacks artifacts into models/ & reports/
│
├── models/                     # Git-tracked production model bundle
│   ├── attribution_hybrid.json # XGBoost model weights (3.5 MB)
│   ├── attribution_hybrid.pkl  # Calibration metadata, temperature, OOD detector
│   ├── attribution_hybrid.manifest.json # Complete data provenance and audit trail
│   └── feature_extractor.pkl   # Fitted PCA and StandardScaler transforms (68 MB)
│
├── reports/                    # Evaluation summaries and feature importances
├── samples/                    # Curated held-out test images (10 per class) with manifest
├── tests/                      # Automated test suite (pytest)
├── AGENTS.md                   # Repository guidelines for AI coding agents
└── AGENT_LOG.md                # Dated chronological development history
```

### Model Bundle & Artifact Specification (`models/`)

The installed model bundle consists of four versioned files operating as an atomic unit:

| File | Size | Format | Contents & Operational Role |
|---|---:|---|---|
| [`models/attribution_hybrid.json`](models/attribution_hybrid.json) | 7.9 MB | JSON | **Trained XGBoost Decision Trees**. Contains native tree topologies, feature split indices, numerical thresholds, and class leaf weights. Being pure JSON, it is platform-independent and immune to Python serialization breaking changes. |
| [`models/attribution_hybrid.pkl`](models/attribution_hybrid.pkl) | 22 KB | Python Pickle (`joblib`) | **Calibration & Safety Metadata**. Bundles the `LabelEncoder` (mapping 0–11 to class names), empirical temperature parameter ($T \approx 0.831$), the 12 average DINOv2 visual centroids ($\boldsymbol{\mu}_c$), the fitted OOD Logistic Regression pipeline, and the OOD threshold ($0.751$). |
| [`models/attribution_hybrid.manifest.json`](models/attribution_hybrid.manifest.json) | 3.2 KB | JSON | **Provenance Audit Trail & Health Contract**. Cryptographic record storing source shard names, dataset SHA-256 digest (`8ad0080e...`), prompt-grouping method, preprocessing spec, and partition counts. Verified at server startup to ensure model integrity. |
| [`models/feature_extractor.pkl`](models/feature_extractor.pkl) | 70.9 MB | Python Pickle (`joblib`) | **Fitted Spectral PCA Pipeline**. Contains the `StandardScaler` (131,328 means and standard deviations) and the Scikit-learn `PCA(128)` projection matrix of shape $(128 \times 131,328)$. Storing nearly 17 million 32-bit floats accounts for its ~70 MB size; tracked via **Git LFS**. |

---

## Dataset, Privacy, and Limitations

- **Dataset & Attribution**: Real-data training utilizes OwensLab's *CommunityForensics* and *CommunityForensics-Small* datasets. Please review upstream dataset licenses and citation guidelines.
- **Privacy**: The local web server never writes uploaded images to disk; all analysis is performed in memory.
- **Experimental Scope**: The model is trained on specific versions of 11 generator classes and real photography. Image compression, social-media re-encoding, resizing, or newer model versions can degrade accuracy.
- **Weak OOD Rejection**: Current out-of-distribution calibration catches approximately 15.5% of degraded out-of-domain images. Unknown generator models may receive high-confidence false attribution.
- **Uneven Performance**: F1 performance varies significantly across classes (from 0.21 on Ideogram V2 to 0.85 on real photos).
- **Not Forensic Evidence**: Predictions are research estimates and must **never** be interpreted as legal or conclusive proof of image provenance.
