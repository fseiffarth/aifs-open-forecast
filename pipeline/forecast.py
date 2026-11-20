#!/usr/bin/env python3
"""Run an AIFS Single v2.0 forecast from the most recent ECMWF open-data analysis.

Each run:
  1. asks ECMWF open data for the newest cycle (00/06/12/18 UTC) whose atmosphere AND wave
     data are published (the model also needs the cycle 6 h earlier, which is always there),
  2. skips if that cycle was already forecast (unless --force),
  3. writes a run config derived from models/aifs-single-2.0/inference.yaml,
  4. runs `anemoi-inference run` and writes GRIB output to forecasts/<YYYYMMDDHH>/.

Safe to call repeatedly (e.g. from cron): a lock prevents overlapping runs and finished
cycles are skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPAT_DIR = Path(__file__).resolve().parent / "compat"

# Each model has its own package pins (models/<name>/pyproject.toml), hence its own venv.
MODELS = {
    "single": dict(dir=ROOT / "models" / "aifs-single-2.0", checkpoint="aifs-single-mse-2.0.ckpt",
                   sha256="4e23ef40157b9e99b5228e5dcffb0d0d5a789f019ffcc89033ee9caae5d7ff6b",
                   hf_repo="ecmwf/aifs-single-2.0", size="~1 GB", venv=ROOT / ".venv"),
    "ens": dict(dir=ROOT / "models" / "aifs-ens-2.0", checkpoint="aifs-ens-crps-2.0.ckpt",
                sha256="e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb",
                hf_repo="ecmwf/aifs-ens-2.0", size="~2.6 GB", venv=ROOT / ".venv-ens"),
}
MODEL_DIR = MODELS["single"]["dir"]
CHECKPOINT = MODEL_DIR / MODELS["single"]["checkpoint"]
BASE_CONFIG = MODEL_DIR / "inference.yaml"
VENV_BIN = ROOT / ".venv" / "bin"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}", flush=True)


def latest_cycle(source: str) -> dt.datetime:
    """Newest cycle for which both the oper (atmosphere) and wave streams are available."""
    from ecmwf.opendata import Client

    client = Client(source)
    oper = client.latest(type="fc", step=0, param="2t")
    wave = client.latest(type="fc", step=0, param="swh", stream="wave")
    return min(oper, wave)


def native_flash_attn_usable(device: str) -> bool:
    if device != "cuda" or importlib.util.find_spec("flash_attn") is None:
        return False
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(0)
        if free >= 20 * 1024**3:  # AIFS at N320 needs well above 8 GB
            return "cuda"
    return "cpu"


def ensure_model(model: str = "single") -> None:
    """Download the checkpoint and its config from Hugging Face on first use."""
    m = MODELS[model]
    model_dir, hf_repo = m["dir"], m["hf_repo"]
    ckpt = model_dir / m["checkpoint"]
    needed = [ckpt, model_dir / "inference.yaml", model_dir / "lsm.grib"]
    if all(f.exists() for f in needed):
        return
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))  # keep out of $HOME
    from huggingface_hub import snapshot_download

    log(f"model not found in {model_dir}; downloading {hf_repo} from Hugging Face ({m['size']}, once)")
    snapshot_download(hf_repo, local_dir=model_dir)
    missing = [f.name for f in needed if not f.exists()]
    if missing:
        raise SystemExit(f"download of {hf_repo} is missing {', '.join(missing)}")
    log("verifying checkpoint checksum")
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != m["sha256"]:
        ckpt.rename(ckpt.with_suffix(".ckpt.bad"))
        raise SystemExit(f"checkpoint checksum mismatch (got {h.hexdigest()}); moved to {ckpt.name}.bad")
    log(f"model ready in {model_dir}")


def build_config(date: dt.datetime, lead_time: str, device: str, out_dir: Path, model: str = "single") -> dict:
    m = MODELS[model]
    cfg = yaml.safe_load((m["dir"] / "inference.yaml").read_text())
    cfg["checkpoint"] = str(m["dir"] / m["checkpoint"])
    cfg["date"] = date.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["lead_time"] = lead_time
    cfg["device"] = device
    cfg.pop("env", None)  # runtime_env() sets ANEMOI_INFERENCE_NUM_CHUNKS for every model
    for proc in cfg.get("pre_processors", []):
        spec = proc.get("forward-transform-filter")
        if isinstance(spec, dict) and "path" in spec:
            spec["path"] = str(m["dir"] / Path(spec["path"]).name)
    cfg["output"]["grib"]["path"] = str(out_dir / f"{m['dir'].name}_{date:%Y%m%d%H}.grib")
    return cfg


def runtime_env(device: str, threads: int | None = None) -> dict:
    """Environment for running anemoi-inference (used by forecast.py and ensemble.py)."""
    env = os.environ.copy()
    cache = ROOT / ".cache"  # keep all caches on the project disk (home has a small quota)
    env.setdefault("XDG_CACHE_HOME", str(cache))
    env.setdefault("HF_HOME", str(cache / "huggingface"))
    env.setdefault("TMPDIR", str(cache / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # mir (open-data regridding) preloads atlas' Fortran binding, which needs libgfortran.so.5;
    # the machine has none, so a copy from conda-forge lives in .sysdeps/runtime.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(ROOT / ".sysdeps" / "runtime"), env.get("LD_LIBRARY_PATH")]))
    env.setdefault("OMP_NUM_THREADS", str(threads or os.cpu_count()))
    env.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")  # chunk the graph mappers to cap memory
    if not native_flash_attn_usable(device):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(COMPAT_DIR), env.get("PYTHONPATH")]))
        log("using PyTorch flash_attn shim (no sm_80+ GPU / flash-attn)")
    return env


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lead-time", default="240h", help="forecast length, e.g. 48h, 5d, 240h (default 240h)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="default: auto")
    p.add_argument("--date", help="initial time YYYYMMDDHH instead of the latest cycle")
    p.add_argument("--source", default="ecmwf", choices=["ecmwf", "aws", "azure", "google"],
                   help="open-data mirror (default ecmwf)")
    p.add_argument("--output-root", default=str(ROOT / "forecasts"))
    p.add_argument("--force", action="store_true", help="re-run even if this cycle is already done")
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="torch CPU threads")
    args = p.parse_args()

    out_root = Path(args.output_root).resolve()  # subprocess runs with cwd=out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    lock = open(out_root / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another forecast run is in progress; exiting")
        return 0

    date = dt.datetime.strptime(args.date, "%Y%m%d%H") if args.date else latest_cycle(args.source)
    out_dir = out_root / f"{date:%Y%m%d%H}"
    done = out_dir / "DONE"
    if done.exists() and not args.force:
        log(f"cycle {date:%Y-%m-%d %HZ} already forecast ({out_dir}); nothing to do")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    done.unlink(missing_ok=True)

    ensure_model()
    device = resolve_device(args.device)
    cfg = build_config(date, args.lead_time, device, out_dir)
    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = runtime_env(device, args.threads)
    log(f"forecast from {date:%Y-%m-%d %HZ}, lead time {args.lead_time}, device {device}")
    log_file = out_dir / "run.log"
    with open(log_file, "w") as lf:
        rc = subprocess.call(
            [str(VENV_BIN / "python"), str(COMPAT_DIR / "run_anemoi_inference.py"), "run", str(cfg_path)],
            cwd=out_dir, env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    if rc != 0:
        log(f"FAILED (exit {rc}); see {log_file}")
        return rc
    done.write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
    (out_root / "latest").unlink(missing_ok=True)
    (out_root / "latest").symlink_to(out_dir.name)
    log(f"done: {cfg['output']['grib']['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
