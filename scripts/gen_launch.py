#!/usr/bin/env python
"""Launch the whole paper experiment program from paper/experiments.yaml.

This is the runnable counterpart of the manifest: it merges each run's overrides onto
the shared `base`, builds a standalone env-prefixed `sbatch` command, and either prints
the commands (default, so you can review/pipe) or submits them (`--submit`).

  # review every command:
  python -m mama_synth.scripts.gen_launch
  # actually submit all jobs (run from the code root on the cluster):
  python -m mama_synth.scripts.gen_launch --submit
  # one group only:
  python -m mama_synth.scripts.gen_launch --groups E5,E6 --submit

Dedup runs (fam_multitask, enh_000) are not retrained — they are emitted as relative
symlinks to their shared source run (loss_l1mselpips); the symlink resolves once that
run's dir exists. Generative (LDM) runs get a longer wall-clock.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # mama_synth/
MANIFEST = ROOT / "paper" / "experiments.yaml"
SLURM = "mama_synth/scripts/train.slurm"
LDM_MODELS = {"ldm_ddpm", "ldm_flow", "ldm_bridge"}
# runs that are identical to an already-trained run -> symlink instead of retrain
DEDUP = {"fam_multitask": "loss_l1mselpips", "enh_000": "loss_l1mselpips"}


def env_str(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        if k == "EXPORT_NOTE":
            continue
        parts.append(f"{k}={shlex.quote(str(v))}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="run sbatch (else just print)")
    ap.add_argument("--groups", default="", help="comma-separated subset, e.g. E5,E6")
    ap.add_argument("--gres", default="gpu:nvidia_h100_nvl:1")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--qos", default="gpu")
    ap.add_argument("--time", default="06:00:00", help="wall-clock for non-LDM runs")
    ap.add_argument("--time-ldm", default="16:00:00", help="wall-clock for LDM runs")
    ap.add_argument("--include-optional", action="store_true",
                    help="also launch the optional E1p family-x-OOD group")
    args = ap.parse_args()

    spec = yaml.safe_load(MANIFEST.read_text())
    base = {k: v for k, v in spec["base"].items() if k != "EXPORT_NOTE"}
    base.setdefault("EXPORT_PREDS", 1)   # always export predictions for the official scorer
    wanted = set(filter(None, args.groups.split(",")))

    cmds, links = [], []
    for run in spec["runs"]:
        grp, name = run["group"], run["name"]
        if wanted and grp not in wanted:
            continue
        if grp == "E1p" and not args.include_optional and not wanted:
            continue
        if name in DEDUP:
            links.append(f"ln -sfn {DEDUP[name]} runs/{name}")
            continue
        cfg = dict(base)
        cfg.update(run.get("overrides", {}))
        cfg["OUTPUT_DIR"] = f"runs/{name}"
        is_ldm = str(cfg.get("MODEL", "")) in LDM_MODELS
        gpu = (f"--partition={args.partition} --qos={args.qos} "
               f"--time={args.time_ldm if is_ldm else args.time} --gres={args.gres}")
        cmds.append((name, f"{env_str(cfg)} sbatch {gpu} {SLURM}"))

    print(f"# {len(cmds)} training jobs + {len(links)} dedup symlinks "
          f"(manifest: {MANIFEST.name})\n")
    for name, c in cmds:
        print(f"# {name}\n{c}\n")
    if links:
        print("# dedup symlinks (run once their source dir exists; dangling is fine):")
        for l in links:
            print(l)

    if args.submit:
        print("\n# --- submitting ---")
        for name, c in cmds:
            print(f"submitting {name} ...")
            subprocess.run(c, shell=True, check=False)
        for l in links:
            subprocess.run(l, shell=True, check=False)


if __name__ == "__main__":
    main()
