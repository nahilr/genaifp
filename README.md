# GenAI Fingerprint

GenAI Fingerprint (GenAIFP) is a local research prototype that estimates whether an image resembles one of the generator classes in its installed model or the real-camera class. It combines a pretrained DINOv2 visual embedding, spatial-frequency measurements, and a calibrated XGBoost classifier. The project includes a command-line interface, a local browser interface, and a Kaggle training workflow.

**The output is experimental. It can be wrong and is not evidence of an image's origin, not a forensic conclusion, and not a benchmark result.** The included model is a bounded research model with significant class imbalance in performance and weak rejection of unsupported inputs. Read the evaluation and limitations below before interpreting a prediction.

## Contents

- [Current state](#current-state)
- [What the system does](#what-the-system-does)
- [Quick start: use the included model](#quick-start-use-the-included-model)
- [Install from a clean checkout](#install-from-a-clean-checkout)
- [Analyze an image from the command line](#analyze-an-image-from-the-command-line)
- [Use the local web app](#use-the-local-web-app)
- [Retrain in Kaggle, then download the model](#retrain-in-kaggle-then-download-the-model)
- [Run training in Kaggle](#retrain-in-kaggle-then-download-the-model)
- [Synthetic demo mode](#synthetic-demo-mode)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Dataset, privacy, and limitations](#dataset-privacy-and-limitations)

## Current state

The workspace contains the successful Kaggle model bundle in `models/`, 120 labeled held-out images in `samples/communityforensics-test/`, and three downloaded ZIP files in the ignored `kaggle-output/downloads/` folder. Raw source shards and generated feature tables remain in Kaggle and are not downloaded locally. The reports archive from the new workflow is installed directly in `reports/`.

| Internal grouped evaluation | Kaggle version 6 result |
|---|---:|
| Known-class held-out accuracy | 58.2% (232 examples) |
| Known-class macro-F1 | 0.556 |
| Held-out degraded-image OOD recall | 15.5% (232 examples) |
| Calibration degraded-image OOD AUROC | 0.723 |
| Calibration false reject rate | 4.8% |

Per-class F1 on the held-out known-class rows:

| Class | Test examples | F1 |
|---|---:|---:|
| DALL·E 2 | 18 | 0.708 |
| DALL·E 3 | 20 | 0.387 |
| Firefly Image 2 | 20 | 0.744 |
| Firefly Image 3 | 21 | 0.837 |
| FLUX dev | 22 | 0.578 |
| FLUX schnell | 14 | 0.343 |
| Ideogram V1 | 21 | 0.270 |
| Ideogram V2 | 18 | 0.214 |
| Imagen 3 | 22 | 0.357 |
| Midjourney V5.2 | 21 | 0.615 |
| Midjourney V6.1 | 17 | 0.765 |
| Real camera | 18 | 0.850 |

These are internal prompt-grouped results, not an official benchmark score. The 11 generator classes come from source rows marked `test`; the real-camera rows came from source rows marked `train`. The configured shards contained no unsupported generator examples, so OOD calibration and recall use degraded copies of target images. This 15.5% OOD recall does not measure rejection of genuinely unseen generators. Prompt grouping also cannot guarantee generation-batch or source-photo separation.

## What the system does

### Prediction pipeline

The ready-to-use model expects a 512 × 512 RGB crop (smaller inputs are padded by the local analyzer). Processing proceeds as follows:

1. **Preprocess the image.** Decode it as RGB and take a center crop. The spectral branch converts the crop to luminance and applies a 3 × 3 SRM high-pass filter to emphasize local residual patterns.
2. **Extract visual features.** DINOv2 (`vit_small_patch14_dinov2.lvd142m`) processes a resized view of the crop and produces a normalized 384-number embedding. This is a general pretrained visual representation; it is not itself an image-generator detector.
3. **Extract spatial-frequency features.** The residual is transformed with a 2D FFT. A log-power spectrum is summarized and projected through a training-fitted scaler and PCA into 128 numbers.
4. **Add hand-designed measurements.** Twelve values describe spectral peak-to-average ratio, high-frequency slope and intercept, energy across eight radial frequency bands, and directional variation.
5. **Classify and assess uncertainty.** The 384 + 128 + 12 values form a 524-number vector for the XGBoost classifier. A separate calibration step adjusts class probabilities, and a separate OOD model/threshold estimates whether an input may be outside the supported classes.
6. **Return a research result.** The CLI and web UI show the predicted class, scores, uncertainty/OOD fields, and model provenance. An OOD flag is not guaranteed to catch unknown images.

### Supported output classes

The installed Kaggle model has 12 labels: `dalle2`, `dalle3`, `midjourney_v5_2`, `midjourney_v6_1`, `firefly_image2`, `firefly_image3`, `flux_dev`, `flux_schnell`, `ideogram_v1`, `ideogram_v2`, `imagen3`, and `real_camera`. Other inputs do not receive a named generator class. Class labels are dataset categories, not proof of model lineage.

### Training pipeline at a glance

In Kaggle, the CommunityForensics workflow reads image bytes and source metadata from Parquet shards, selects capped samples, removes exact duplicate encoded images, creates prompt-based grouping IDs, computes features, records source provenance, and adds one degraded copy for each retained known-class image. It uses prompt-group-disjoint train, calibration, and test partitions, fits scaling/PCA on training groups, calibrates probabilities and OOD rejection on calibration groups, and reports internal held-out results. The dataset has no generation-batch ID or unique real source-image ID, so those relationships cannot be kept disjoint.

For the completed version 6 run, the 12 target classes supplied 1,200 original examples in total, capped at 100 per class. Those examples were divided into 780 training, 188 calibration, and 232 test examples. Their degraded copies supplied 188 calibration OOD examples and 232 test OOD examples. The prompt-group counts were 735 training, 178 calibration, and 224 test. No unsupported-generator examples were found in the configured shards.

The source split is kept in the manifest for provenance: 1,100 selected generator rows were marked `test` upstream and 100 real-camera rows were marked `train`. The internal partitions are built by prompt groups across those source rows; they do not reproduce the dataset's official benchmark split. The Kaggle output includes the model bundle and a separate archive of 120 held-out images, 10 for each target class.

## Quick start: use the included model

### Install from a clean checkout

Use Python 3.10 or newer. These steps do not install this repository as a package. Run commands from the project root so `main.py`, `src/genaifp/`, and `models/` resolve correctly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision timm
```

`requirements.txt` supplies the project's general Python dependencies. PyTorch, torchvision, and timm are separate because PyTorch's appropriate build depends on your operating system and CPU/GPU setup. If the generic pip command does not select a suitable PyTorch build for your machine, install the matching PyTorch/torchvision build first, then install `timm`.

DINOv2 pretrained weights are downloaded the first time an image is analyzed. The machine needs internet access for that first download. CPU inference is possible but can be slow; a compatible GPU can improve speed. The included model files are already in the checkout, so you do not need to download or train a model just to try prediction.

## Analyze an image from the command line

```bash
source .venv/bin/activate
python main.py predict --image_path samples/midjourney_v6_sample.png
```

The checked-in `samples/*_sample.png` images are procedural, class-coded examples made by the demo generator. They are useful for checking that the command runs, but predictions on them are not a meaningful accuracy test.

Save a result as JSON:

```bash
python main.py predict \
  --image_path /path/to/image.jpg \
  --model_path models/attribution_hybrid \
  --output_json reports/prediction.json
```

`--model_path` is the model basename without `.json` or `.pkl`. By default, the CLI uses `models/attribution_hybrid`. Its `.manifest.json` must identify a completed calibrated CommunityForensics research evaluation; legacy/demo models are rejected. The manifest also sets the expected crop size, feature transform, class list, and provenance.

Useful CLI commands:

```bash
python main.py --help
python main.py build --help
python main.py train --help
python main.py download_shards --help
```

## Use the local web app

Install the web server and upload support in the active environment:

```bash
python -m pip install fastapi 'uvicorn[standard]' python-multipart
```

Start the app from the repository root:

```bash
python main.py web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The project guide is at [http://127.0.0.1:8000/about](http://127.0.0.1:8000/about). To use another evaluated model bundle, set its basename with `GENAIFP_MODEL` before starting the app, or pass a model path with `--model`:

```bash
GENAIFP_MODEL=models/attribution_hybrid python main.py web
python main.py web --model models/attribution_hybrid --port 8000
```

The service binds to loopback (`127.0.0.1`). It accepts PNG, JPEG, and WebP uploads up to 20 MB and 40 megapixels. It rejects cross-origin prediction uploads, does not expose the FastAPI API documentation pages, and processes uploads in memory; the application does not save uploaded images. The model and DINOv2 weights are loaded for the running process. Stop the server with `Ctrl+C`.

## Retrain in Kaggle, then download the model

Dataset downloads, image feature extraction, training, calibration, and evaluation run in Kaggle. You do not need to download the source dataset or train on your computer. The current Kaggle caps are 100 examples per target class, 25 per unsupported generator, and 200 unsupported examples total when such rows are available. The configured shards currently contain only the target generators, so degraded copies provide the OOD examples; unsupported-source rows will be retained if present. The workflow exports 10 held-out images for each class with a CSV source manifest.

The target classes are DALL·E 2, DALL·E 3, Midjourney V5.2, Midjourney V6.1, Firefly Image 2, Firefly Image 3, FLUX dev, FLUX schnell, Ideogram V1, Ideogram V2, Imagen 3, and `real_camera`. These are the model versions found in the current configured dataset shards. Other generated images remain unknown/OOD.

Training and artifact handling are separate stages. The Kaggle notebook has a training stage and a packaging stage; packaging creates three archives: the model bundle, held-out images, and evaluation/provenance reports. To download those archives on a machine with Kaggle CLI credentials:

```bash
python kaggle/download_outputs.py
```

Install them separately on the machine where you want to use the model. This step needs no Kaggle login; you can copy the ZIPs from `kaggle-output/downloads/` to another machine and point `--downloads` at that folder:

```bash
python kaggle/install_outputs.py
# Or on another machine:
python kaggle/install_outputs.py --downloads /path/to/downloads
```

The installer puts model files in `models/`, current generated reports directly in `reports/`, and the extracted samples in `samples/communityforensics-test/`. The ZIPs contain no source shards or feature tables. See [the Kaggle workflow guide](kaggle/README.md) for the separate training, packaging, download, and install stages.

The train, calibration, and test groups are separated by normalized prompt. The original source split is retained for provenance. Commercial dataset rows are marked `test`, so results are an internal grouped evaluation rather than the dataset's official benchmark score. Prompt grouping cannot guarantee that generation batches or source photographs are disjoint.

The local app only performs prediction. It may download DINOv2 weights the first time you analyze an image; it does not download training data, extract dataset features, or train a model.

## Synthetic demo mode

Demo mode creates procedural, class-coded images for exercising feature extraction. Those generated patterns deliberately differ by label, so a high demo score says nothing about real generator attribution. Demo models are tagged synthetic and are rejected by prediction and the web app.

```bash
python main.py build --demo \
  --output data/demo/features_demo.parquet \
  --demo_samples_per_class 100

python main.py train \
  --input data/demo/features_demo.parquet \
  --model_out models/demo_attribution \
  --report_dir reports/demo \
  --demo-only
```

Demo builds also create a `features_hybrid.parquet` and `feature_extractor.pkl` beside the requested output, save another feature extractor at `models/feature_extractor.pkl`, and save reference images under `samples/`. Run demo commands only when you want to create those generated files; they are not needed for normal prediction.

## Project layout

```text
.
├── main.py                         # Local CLI: prediction, web app, synthetic demos
├── src/genaifp/                        # Local preprocessing, features, training, and web code
│   ├── web.py                      # Local FastAPI app backend
│   └── web/                        # Browser UI and project guide
├── kaggle/                         # Standalone Kaggle sources and generated notebook
├── models/               # Included evaluated model bundle and manifest
├── reports/                        # Current Kaggle reports and evaluation summary
├── samples/                        # Demo images and downloaded held-out test set
├── tests/                           # Automated test modules
├── requirements.txt                # General dependencies (no project install)
├── pyproject.toml                  # Packaging metadata and optional dependency groups
├── AGENTS.md                       # Repository instructions for coding agents
└── AGENT_LOG.md                    # Dated development history
```

Feature tables and downloaded shards may be created locally under `data/` by older research commands; the documented real-data workflow keeps that processing in Kaggle. Kaggle ZIPs are downloaded under ignored `kaggle-output/`. The current evaluation and provenance reports are installed directly in `reports/`; the model bundle also carries its manifest at `models/attribution_hybrid.manifest.json`. `pyproject.toml` contains package metadata, but the documented workflow runs `python main.py` directly and does not use an editable install.

## Tests

Install the requirements and the optional vision/web dependencies first, then run the test suite from the repository root:

```bash
python -m pip install torch torchvision timm fastapi 'uvicorn[standard]' python-multipart
python -m pytest
```

The tests cover CLI routing, preprocessing, spectral/visual feature code, model behavior, dataset import, download behavior, the research trainer, robustness utilities, and the web API. The current suite completed with 85 passing tests and one scikit-learn warning.

## Dataset, privacy, and limitations

- **Dataset and attribution:** the real-data workflow uses OwensLab's CommunityForensics and CommunityForensics-Small Parquet data. Review the current dataset card, license, and attribution requirements before downloading, using, or sharing source data or derived artifacts.
- **Input handling:** the local web service does not save uploads; image bytes are processed in memory. The feature importer can retain original image bytes by default, so use `--drop_image_bytes` when that is not needed and protect raw shards/features as appropriate.
- **Model scope:** the installed model has 11 generator labels plus real camera. Prompts, datasets, post-processing, and distributions can still differ from training data.
- **Weak OOD rejection:** the threshold caught 15.5% of held-out degraded images. Rejection of unsupported generators was not measured because none were present in the configured shards.
- **Uneven class performance:** per-class F1 ranges from 0.214 (Ideogram V2) to 0.850 (real camera) in this small evaluation.
- **Robustness limits:** severe JPEG and downscale variants are often misclassified without uncertainty. Severe blur triggers uncertainty more often but also has poor top-class accuracy.
- **Partial grouping:** prompts are disjoint across splits, but batches and individual real source images cannot be grouped with fields present in this dataset.
- **Not a forensic tool:** never use the model score as proof of authorship, authenticity, or generator identity.
