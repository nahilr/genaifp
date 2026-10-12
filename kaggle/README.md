# Kaggle training workflow

The Kaggle workflow does the expensive work: downloads the CommunityForensics shards, extracts image features, creates prompt-grouped partitions, trains and calibrates the model, evaluates it, and selects held-out test images. The notebook is standalone and uses Kaggle's installed libraries; it does not import `src/genaifp/` or install this repository.

The configured sample caps are 500 clean examples per target class, 25 unsupported examples per generator, and 200 unsupported examples total when such rows occur. Each clean target image also produces a degraded OOD example, assigned to a prompt-grouped partition. The configured shards contain only the 11 target generators, so degraded copies supply the OOD examples in this run; unsupported-source rows are retained when available. The original dataset split is preserved in the provenance manifest. Since the commercial rows are marked `test` in the source, the model's split is an internal prompt-grouped evaluation, not an official benchmark result.

The 11 generator targets are DALL·E 2/3, Midjourney V5.2/V6.1, Firefly Image 2/3, FLUX dev/schnell, Ideogram V1/V2, and Imagen 3. V5.2 is used because that is the Midjourney version present in the configured training shards; it is not relabeled as V5.1.

The completed run used 6,000 clean examples (500 per class): 3,870 for training, 935 for calibration, and 1,195 for testing. Their degraded copies supplied 935 calibration and 1,195 test OOD examples. Prompt groups are disjoint across these partitions. The source shards contained no unsupported-generator rows.

The packaging stage should leave exactly three downloadable files in `/kaggle/working`:

- `genaifp-model-bundle.zip` — the classifier JSON/metadata, fitted feature transform, and provenance manifest required by the local app.
- `genaifp-test-images.zip` — 10 distinct held-out original images per class, a CSV with labels/source details, and a contact sheet.
- `genaifp-reports.zip` — `evaluation_summary.json`, `feature_importance.csv`, `import_summary.json`, `environment.json`, and the model provenance manifest.

No raw shards or feature table are included in the downloads. The image ZIP contains 120 images when training has 11 generator labels plus `real_camera`.

## Build and run

Training and packaging are separate notebook stages. Running the full notebook executes both in order; you can rerun the packaging stage independently while training artifacts remain in the Kaggle session. Training cleans raw shards and the feature table but leaves the model, reports, and selected images ready for packaging.

After changing Kaggle source files, regenerate the notebook and push/run it:

```bash
python kaggle/build_notebook.py
kaggle kernels push -p kaggle
kaggle kernels status nahilr/genaifp
kaggle kernels logs nahilr/genaifp
```

Kaggle needs internet access and a GPU accelerator. DINOv2 weights are fetched by the notebook through `torch.hub`. Check that the run completed before downloading its output.

## Download the archives

From the project root, run:

```bash
python kaggle/download_outputs.py
```

This uses the Kaggle CLI to download all three ZIP files into `kaggle-output/downloads/`. Kaggle credentials must be configured on the downloading machine. Use `--kernel` or `--output` to change the source kernel or download location.

## Install on the model-use machine

Run the installer wherever the project checkout and ZIP files are available:

```bash
python kaggle/install_outputs.py
```

Installation does not require Kaggle credentials or a Kaggle session. Copy the ZIPs to another machine and use `--downloads /path/to/downloads`, or provide `--model-archive`, `--image-archive`, and `--report-archive` individually. By default, files go directly into `models/`, `reports/`, and `samples/communityforensics-test/`; installation replaces the image folder on every run and removes a legacy `reports/real_data/` folder if one exists.

The local app does no dataset processing or training. Prediction uses the installed model and may download DINOv2 weights on its first run. Start the analyzer with `python main.py web`.

## Current remote status

The latest Kaggle run completed successfully on a Tesla T4 and exported all three archives: the model bundle, the 120 held-out images, and the generated reports. Those ZIPs are downloaded in `kaggle-output/downloads/` and installed in `models/`, `reports/`, and `samples/communityforensics-test/`.

Earlier revisions are kept for context: versions 3–5 were attempts that exposed Midjourney label, OOD sample, and output-folder check issues, and version 6 fixed those while exporting only the model and image ZIPs. The current workflow adds the reports archive and separates training, packaging, downloading, and installation. Re-run `kaggle kernels push -p kaggle` after changing Kaggle sources, then re-run the download and install stages.
