"""Plot SSL training curves from <run>/log.txt (written by train_ssl.py).

    python scripts/litePT/plot_ssl_log.py $EAGLE_SCRATCH/runs/<run_name>
Writes <run>/training_curves.png and prints first/last-window means of each logged metric.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("total loss", ["loss"]),
    ("component losses", ["mask_loss", "roll_mask_loss", "unmask_loss"]),
    ("KL(teacher || student), mask loss", ["mask_kl"]),
    ("entropy (nats)", ["target_entropy", "pred_entropy"]),
    ("teacher prototypes used (argmax)", ["protos_used"]),
    ("teacher/student argmax agreement", ["argmax_agree"]),
    ("matched fraction", ["mask_match", "roll_match", "unmask_match"]),
    ("grad norm", ["grad_norm"]),
]


def main(run):
    run = Path(run)
    df = pd.DataFrame([json.loads(line) for line in open(run / "log.txt")])
    df = df.drop_duplicates("step", keep="last").sort_values("step")
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, (title, keys) in zip(axes.flat, PANELS):
        for k in keys:
            if k in df:
                ax.plot(df.step, df[k], label=k)
        ax.set_title(title)
        ax.set_xlabel("step")
        if len(keys) > 1:
            ax.legend(fontsize=8)
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(run / "training_curves.png", dpi=110)
    n = max(1, len(df) // 10)
    summary = pd.DataFrame({"first": df.head(n).mean(numeric_only=True), "last": df.tail(n).mean(numeric_only=True)})
    print(summary.loc[[k for _, ks in PANELS for k in ks if k in summary.index]].round(4).to_string())
    print(f"-> {run / 'training_curves.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
