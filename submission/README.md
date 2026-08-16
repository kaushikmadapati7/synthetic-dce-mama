# MAMA-SYNTH Grand Challenge submission container

Packages our pre-only synthesizer into the GC algorithm container. `inference.py`
replicates the training pipeline exactly — center-crop/pad → (optional) per-image
normalization → residual U-Net → invert both → write `output.mha` at native size with
metadata preserved — so what we validated locally is what runs on GC.

I/O contract (fixed by the challenge):
```
IN : /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
OUT: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
```

## Steps
**1. Stage the chosen model** into `submission/model/`:
```bash
cp <best-run>/checkpoints/<unet|multitask>_last.pt  submission/model/model.pt
cp <best-run>/config.json                           submission/model/config.json
```
`config.json` supplies `base_ch / n_upsamples / spatial_size / per_image_norm`.
**Add `"out_channels": 2` if the model is `multitask`** (the run config doesn't store it;
default is 1 for `unet`). See `model/config.example.json`.

**2. Test locally** (needs Docker; HPC compute nodes usually have Singularity, not
Docker — build on a machine/node with Docker, or your laptop):
```bash
cp <preprocessed>/mha/input/DUKE_001.mha submission/test/input/images/pre-contrast-dce-mri-slice-breast/
bash submission/do_test_run.sh          # builds + runs; output in submission/test/output/
```

**3. Export + upload:**
```bash
bash submission/do_save.sh              # -> mama-synth-submit-v0.1.0.tar.gz
# upload at grand-challenge.org -> your Algorithm -> Container Management
```

## Notes
- The container imports the real `mama_synth` package (no vendored copy → no drift); the
  build context is the package dir so `model.pt`/`config.json` under `submission/model/`
  are bundled in.
- CPU torch (sub-second per slice); GC GPU is used automatically if present.
- The interface slugs are pre-set; override via `MAMA_INPUT_SLUG`/`MAMA_PREDICTION_SLUG`
  if the challenge phase ever changes them.
- `model.pt` is git-ignored (`*.pt`); stage it at build time.
