# Agent guidance

## Project layout

- `main.py` is the local CLI entrypoint; `src/genaifp/` contains the local application and research modules. Keep the import package name `genaifp`.
- `kaggle/main.py` is the Kaggle orchestration entrypoint. Keep Kaggle workflow code split into adjacent files by responsibility, and use `kaggle/build_notebook.py` to assemble `kaggle/communityforensics_train.ipynb` configured by `kaggle/kernel-metadata.json`.
- `kaggle-output/` contains downloaded run artifacts and should not be committed.
- `AGENT_LOG.md` is the root activity log and preserves historical development notes.

## Kaggle workflow constraints

- Keep the Kaggle script standalone. Do not import `src/genaifp/`, embed a second project copy, or run `pip install -e`.
- Use Kaggle's available third-party libraries directly. DINOv2 weights may be fetched through `torch.hub` when the kernel has internet access.
- When the workflow or output paths change, update the generated notebook, `kaggle/README.md`, and root `README.md`.
- Keep training in `kaggle/main.py`, archive creation in `kaggle/package_outputs.py`, authenticated downloads in `kaggle/download_outputs.py`, and local installation in `kaggle/install_outputs.py` so each stage can run independently.
- Preserve the source dataset, shard names, sampling caps, and source provenance in the output manifest. Commercial-source rows marked `test` are an internal grouped holdout, not an official benchmark score.
- Do not commit raw dataset shards, generated feature tables, or extracted image collections. Keep the final model bundle in `models/` and current evaluation reports in `reports/` versionable.

## Reporting

- Distinguish current behavior from historical plans and results.
- Report whether a Kaggle revision was actually pushed and run. Syntax compilation alone does not validate the remote workflow.
- After each meaningful code change, append a dated entry to the root `AGENT_LOG.md` describing what changed and what validation was performed. Include Kaggle push/run status when relevant. Skip trivial edits that do not change behavior or project structure.
