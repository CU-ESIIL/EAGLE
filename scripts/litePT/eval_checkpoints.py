"""Run the validation tasks (config `eval_tasks`, eagle_als.evaluation) on a run's saved checkpoints.

Useful for runs that predate a task, or to re-score checkpoints after adding one. The model and view
settings come from the run's config.json; the tasks come from --config (default: the current
pre-training config). Needs a GPU (sparse convolutions).

    python scripts/litePT/eval_checkpoints.py $EAGLE_SCRATCH/runs/ssl_s_2gpu --init --tb

Writes <run>/eval_checkpoints.csv (one row per checkpoint and metric). With --tb, also adds the
scores to the run's TensorBoard (eval/*, eval_<task>/*) at each checkpoint's step; --init adds a
random-initialisation baseline at step 0.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

from eagle_als.evaluation import build_tasks, run_tasks, tb_tag
from eagle_als.ssl import SonataLitePT
from eagle_als.train_utils import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--config", default=str(Path(__file__).with_name("configs") / "ssl_litept_s.py"),
                   help="config whose eval_tasks / eval_knn_k are used")
    p.add_argument("--steps", type=int, nargs="*", help="only these checkpoint steps (default: all)")
    p.add_argument("--init", action="store_true", help="also score a random-init model (step 0)")
    p.add_argument("--tb", action="store_true", help="write the scores to <run>/tb")
    a = p.parse_args()

    run = Path(a.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    task_cfg = load_config(a.config)
    cfg.update(eval_tasks=task_cfg["eval_tasks"], eval_knn_k=task_cfg.get("eval_knn_k", 20))
    device = torch.device("cuda")

    ckpts = {int(re.search(r"step_(\d+)", c.name).group(1)): c for c in (run / "checkpoints").glob("step_*.pth")}
    if a.steps:
        ckpts = {s: c for s, c in ckpts.items() if s in a.steps}
    jobs = ([(0, None)] if a.init else []) + sorted(ckpts.items())
    if not jobs:
        sys.exit(f"no checkpoints in {run / 'checkpoints'}")

    tasks = build_tasks(cfg["eval_tasks"], cfg)
    writer = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run / "tb", filename_suffix=".eval_checkpoints")
    rows = []
    for step, path in jobs:
        torch.manual_seed(cfg.get("seed", 0))
        model = SonataLitePT(cfg["backbone"], **cfg["ssl"]).to(device)
        if path is not None:
            state = torch.load(path, map_location="cpu", weights_only=False)["model"]
            missing, _ = model.load_state_dict(state, strict=False)
            missing = [k for k in missing if k.startswith("teacher.backbone")]
            if missing:
                raise RuntimeError(f"{path}: teacher backbone weights missing, e.g. {missing[:3]}")
        logs = run_tasks(model, tasks, device)
        print(f"[eval] step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in logs.items()
                                               if tb_tag(k).startswith("eval/")), flush=True)
        rows += [dict(step=step, metric=k, value=v) for k, v in logs.items()]
        if writer:
            for k, v in logs.items():
                writer.add_scalar(tb_tag(k), v, step)
        del model
        torch.cuda.empty_cache()
    if writer:
        writer.close()
    out = run / "eval_checkpoints.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
