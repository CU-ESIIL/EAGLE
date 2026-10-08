"""Evaluation tasks run during pre-training on the frozen teacher encoder (a validation step).

Tasks are listed in the config as `eval_tasks`, one spec per task with a `type` and a `name`:

    eval_tasks = [dict(type="classification", name="nlcd", table=..., ...)]   # see eagle_als.probe

`train_ssl.py` builds every task once on rank 0 when the run starts, then runs them every `eval_every`
steps (and at step 0 of a fresh run, the random-initialisation baseline) while the other ranks wait.
The model is in eval mode with gradients off, and is put back in train mode afterwards.

A task's `run(model, device)` returns {metric: value}. In TensorBoard (see `tb_tag`):
    metric without "/"  (headline, e.g. linear_f1)        -> eval/<name>/<metric>   (all tasks in one section)
    metric with "/"     (detail, e.g. linear_f1/<class>)  -> eval_<name>/<metric>   (one section per task)

Adding a task type: subclass `EvalTask` in a module, decorate it with `@register("<type>")`, and add
the module to `TASK_MODULES`. Keep a task's run time well under the NCCL timeout (10 min), since the
other GPUs wait for it.
"""

import importlib
import time

import torch

TASK_MODULES = ("eagle_als.probe",)  # imported by build_tasks so their task types register
TASKS = {}


def register(type_name):
    """Class decorator: make an EvalTask subclass available as `type=<type_name>` in eval_tasks."""
    def deco(cls):
        TASKS[type_name] = cls
        return cls
    return deco


class EvalTask:
    """Base class. __init__ does all one-off preparation (load data, fixed inputs); run() must be repeatable."""

    def __init__(self, spec, cfg):
        self.name = spec["name"]

    def run(self, model, device):
        """{metric: float} for the current (frozen) model."""
        raise NotImplementedError


def build_tasks(specs, cfg):
    for module in TASK_MODULES:
        importlib.import_module(module)
    tasks = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type", "classification")
        if kind not in TASKS:
            raise ValueError(f"eval task {spec.get('name')}: unknown type {kind!r} (known: {sorted(TASKS)})")
        tasks.append(TASKS[kind](spec, cfg))
    names = [t.name for t in tasks]
    if len(set(names)) != len(names):
        raise ValueError(f"eval task names must be unique: {names}")
    return tasks


def run_tasks(model, tasks, device):
    """{"<task>/<metric>": value} for every task (plus <task>/seconds), model frozen and restored afterwards."""
    was_training = model.training
    model.eval()
    try:
        logs = {}
        with torch.no_grad():
            for task in tasks:
                t0 = time.time()
                for k, v in task.run(model, device).items():
                    logs[f"{task.name}/{k}"] = float(v)
                logs[f"{task.name}/seconds"] = time.time() - t0
        return logs
    finally:
        model.train(was_training)


def tb_tag(key):
    """TensorBoard tag for a run_tasks key: headline metrics under eval/, detail metrics in eval_<task>/."""
    task, metric = key.split("/", 1)
    return f"eval_{task}/{metric}" if "/" in metric else f"eval/{key}"
