"""SageMaker training-job entry point: runs scripts/run_all.sh inside the
AWS PyTorch training container.

* results/, runs/ and logs/ live in /opt/ml/checkpoints, which SageMaker syncs
  to the job's checkpoint S3 URI while it runs. With managed spot training,
  an interrupted job restarts with them restored, and the .done markers make
  run_all.sh skip finished stages.
* Hyperparameters named env_<VAR> become environment variables for
  run_all.sh (e.g. env_SLLM_STEPS=30 for a smoke test).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
CKPT = Path("/opt/ml/checkpoints")


def log(*a):
    print("[sm_entry]", *a, flush=True)


def pick_data_dir() -> Path:
    best, free = None, -1
    for c in ("/opt/ml/input", "/tmp", "/opt/ml/code"):
        p = Path(c)
        if p.exists() and os.access(p, os.W_OK):
            f = shutil.disk_usage(p).free
            if f > free:
                best, free = p, f
    log(f"data dir {best} ({free / 2**30:.0f} GB free)")
    return best / "sslab_data"


def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    for d in ("results", "runs", "logs"):
        (CKPT / d).mkdir(exist_ok=True)
        t = CODE / d
        if t.is_symlink():
            continue
        if t.exists():
            for f in t.iterdir():  # keep anything shipped with the code (e.g. .gitkeep)
                dst = CKPT / d / f.name
                if not dst.exists():
                    shutil.move(str(f), dst)
            shutil.rmtree(t)
        t.symlink_to(CKPT / d)
    done = sorted(p.name for p in (CKPT / "logs").glob("*.done"))
    log("restored finished stages:", done or "none")

    env = dict(os.environ)
    hp_file = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_file.exists():
        for k, v in json.load(open(hp_file)).items():
            if k.startswith("env_"):
                v = v.strip('"') if isinstance(v, str) else str(v)
                env[k[4:]] = v
                log(f"env {k[4:]}={v}")
    data = pick_data_dir()
    data.mkdir(parents=True, exist_ok=True)
    env.update(DATA=str(data), HF_HOME=str(data / "hf_cache"), TORCH_HOME=str(data / "torch_cache"),
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE / "requirements.txt")],
                   check=True, cwd=CODE)
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run([sys.executable, "-c", "import torch, transformers, peft, speechbrain; "
                    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), "
                    "'transformers', transformers.__version__, 'peft', peft.__version__, 'speechbrain', speechbrain.__version__)"],
                   check=True, env=env)
    rc = subprocess.run(["bash", "scripts/run_all.sh"], cwd=CODE, env=env).returncode
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    for f in ("results/SUMMARY.md", "logs/summary.txt"):
        if (CODE / f).exists():
            shutil.copy(CODE / f, model_dir / Path(f).name)
    failed = sorted(p.stem for p in (CKPT / "logs").glob("*.log")
                    if not (CKPT / "logs" / f"{p.stem}.done").exists() and p.stem != "summary")
    log("stages without .done:", failed or "none")
    sys.exit(rc)


if __name__ == "__main__":
    main()
