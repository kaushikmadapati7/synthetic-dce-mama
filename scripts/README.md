# mama_synth HPC scripts

Run order on the cluster. **Key constraint:** HPC compute nodes (`gpu`/`cpu`
partitions) have **no outbound internet** — the http proxy is unreachable from the
compute fabric. Only the **login node** can reach Synapse / download torchvision
weights, so anything needing the network happens there.

| # | Where | Command | Purpose |
|---|-------|---------|---------|
| 1 | login (tmux) | `synapse get -r syn60868042 ...` (see `run_download.sh` on scratch) | download raw 3D dataset |
| 2 | login | `git clone https://github.com/mama-research/mama-synth.git /path/to/mama_mia/mama-synth` | get official preprocess/eval scripts (challenge repo) |
| 3 | login | `bash mama_synth/scripts/precache_weights.sh` | cache VGG16 (+Inception) into `$TORCH_HOME` on scratch |
| 4 | `sbatch` (cpu) | `sbatch mama_synth/scripts/preprocess.slurm` | raw 3D → 2D `.mha` tree (+ `training_pre_stats.json`) |
| 5 | `sbatch` (gpu) | `sbatch mama_synth/scripts/train.slurm` | train + eval, writes `runs/<model>_<jobid>/` |

All `sbatch` jobs read defaults from env vars — override inline, e.g.:

```bash
MODEL=gan EPOCHS=200 BATCH_SIZE=24 sbatch mama_synth/scripts/train.slurm
```

Submit from the **project root** (the dir containing `mama_synth/`). `TORCH_HOME`
must match between `precache_weights.sh` and `train.slurm` (default
`/path/to/torch_cache`) so the offline node finds the weights.

Notes:
- `train.slurm` defaults to `--no-fid`; set `COMPUTE_FID=1` only after step 3 cached
  the Inception weights.
- `preprocess.slurm` has `IMAGE_DIR` / `SEG_DIR` placeholders — confirm them against
  the real downloaded tree and the official `preprocess.py --help` before running.
- Smoke-test the code path anytime without data: `EXTRA_ARGS=--dummy` won't work
  through the slurm guard, so for a dummy run call `python -m mama_synth.main --dummy ...`
  directly (see `mama_synth/main.py` docstring).
