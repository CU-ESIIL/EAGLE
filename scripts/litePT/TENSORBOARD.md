# Reading the SSL pre-training curves

This guide is written into each run's TensorBoard **Text** tab at startup (`train_ssl.py`), and the
repository README links to it. All `train/*` values come from the last micro-batch of the logged
step, averaged over GPUs. They are noisy, so turn on smoothing (about 0.9).

## The one identity to remember

Each loss is a cross-entropy between the teacher's Sinkhorn targets and the student's prediction
over 4,096 prototypes:

    loss  =  target_entropy  +  mask_kl
             (how spread the      (how far the student is
              teacher targets are)  from those targets)

- The loss can never fall below `target_entropy`. The ceiling, ln 4096 = 8.32, means a uniform target.
- `target_entropy` depends on the teacher temperature: a higher `teacher_temp` gives flatter targets
  and a higher entropy. While the temperature ramps from 0.04 to 0.07 (the first 5% of `total_steps`),
  the loss is pushed **up** for reasons that have nothing to do with learning.
- So a **flat loss is not "done learning"**. `mask_kl` is the cleaner learning signal, and the
  `eval/*` probes are the real one. In DINO-style self-distillation the loss usually flattens early
  while the representations keep improving.

## Metrics

| Tag | What it is | Healthy | Warning sign |
| --- | --- | --- | --- |
| `loss` | weighted sum: ¼ `mask_loss` + ¼ `roll_mask_loss` + ½ `unmask_loss` | starts near 8.3, drops fast, then roughly flat | NaN; rising steadily after the warmups end |
| `mask_loss` | masked student global view vs. unmasked teacher, **same** view | below the other two (the easiest pair) | |
| `roll_mask_loss` | masked student view vs. teacher's **other** global view | slightly above `mask_loss` | |
| `unmask_loss` | student **local** views (not masked) vs. teacher's first global view | follows the others | |
| `target_entropy` | mean entropy of teacher targets (mask head, nats); exp(value) is about the number of prototypes each point is spread over | falls, then flat; well above 0 | → 0: collapse onto one-hot targets; stuck near 8.3: teacher not structured |
| `pred_entropy` | mean entropy of the student's softmax | a little above `target_entropy` | far below `target_entropy` (student over-confident) |
| `mask_kl` | KL(target ‖ student), mask head | slowly falls | rises after the warmups end; spikes that do not recover |
| `argmax_agree` | fraction of points where the student's top prototype equals the teacher's | rises slowly; small values (≈0.1) are normal while targets are this spread | stuck at chance (1/4096) |
| `protos_used` | distinct prototypes that are some point's top teacher choice, in this micro-batch | hundreds to a few thousand, stable | drifting toward a few dozen: prototype collapse |
| `protos_soft_used` | perplexity of the batch-mean target (4,096 = all used equally) | high; Sinkhorn keeps it high by construction | |
| `student_protos_used` | the same as `protos_used`, for the student's top choices | follows `protos_used` | far below it |
| `target_max` | mean top target probability | slowly rises | |
| `mask_match`, `roll_match`, `unmask_match` | fraction of student points with a teacher point within `match_max_r` (6.4 m) | `mask_match` = 1; the others are set by view overlap (≈0.7, ≈0.85) | sudden drops: a data or augmentation change |
| `mask_ratio`, `mask_size`, `teacher_temp` | the scheduled values (ramp over the first 5% of `total_steps`, then constant) | | |
| `lr` | learning rate of the last parameter group (one layer-decay step below the base lr) | linear warmup, then constant | |
| `grad_norm` | gradient norm before clipping (`clip_grad` = 3) | O(0.1–1), no trend toward 0 | → 0: nothing to learn (collapse); often > 3: unstable |
| `momentum` | teacher EMA momentum (fixed 0.994) | | |
| `sk_cookies` | cookies in each Sinkhorn normalization, over all GPUs (memory bank) | = `sinkhorn_cookies` (128) after the first steps | |
| `n_points` | voxels in the last global-view micro-batch, one GPU | varies a lot (median ≈ 310k, max ≈ 510k) | |
| `step_s`, `data_frac` | seconds per optimizer step; fraction of that spent waiting for data | `data_frac` near 0 | `data_frac` > 0.2: data-starved |
| `mem_gb` | peak GPU memory since the previous log line | moves up and down with `n_points` | close to 80 |
| `mem_max_gb` | peak GPU memory since the job started | steps up when an unusually large batch arrives, never down; **this is not a leak** | close to 80 |
| `eval/<task>/linear_bal_acc`, `…/linear_f1` | logistic-regression probe on frozen, mean-pooled teacher embeddings (e.g. task `nlcd`: NLCD land cover, spatial hold-out); `_f1` is macro F1 | rises above the step-0 (random init) value | flat at the random-init value |
| `eval/<task>/knn_bal_acc`, `…/knn_f1` | cosine kNN (k = 20) on the same embeddings | rises | |
| `eval/<task>/chance`, `…/seconds` | 1 / number of classes; time the task took | | `seconds` near 600: the NCCL timeout |
| `eval/embed/cookie_erank` | effective rank of the 256 cookie embeddings (how many directions they really spread over) | stays well above a few tens | falls steadily: cookies collapsing onto a few directions |
| `eval/embed/cookie_cos` | mean cosine similarity between pairs of cookie embeddings | well below 1 | rises toward 1: all cookies look alike |
| `eval/embed/point_erank` | effective rank of point features (64 per cookie) | stays in the hundreds | falls steadily: dimensional collapse of point features |
| `eval/embed/point_cos_within` | mean cosine similarity between points of the same cookie | below 1 | → 1: no spatial detail inside a cookie |
| `eval/embed/point_std` | mean per-dimension std of L2-normalised point features | stable | → 0: complete collapse |
| `eval_<task>/linear_f1/<class>`, `eval_<task>/knn_f1/<class>` | F1 of each class on the test rows (own section per task) | most classes rise | NLCD classes with few test rows (barren, high-intensity developed: 4 each) are noisy |

Before 2026-10-08, `mem_gb` was the peak since the job started (now `mem_max_gb`). That is why it
rose in steps every hour or two in the `ssl_s_2gpu` run.

## Collapse vs. slow learning

- **Collapse** (bad): `protos_used` and `student_protos_used` drop toward a few, `grad_norm` → 0,
  and the `eval/embed/*` curves show `cookie_cos` → 1 or effective rank falling.
  Sinkhorn targets hide collapse in the prototype metrics (`protos_soft_used` stays high), so check
  the encoder itself with `eval/embed/*` (or `check_embeddings.py` for runs without that task).
- **Slow learning** (normal): the loss is flat, `mask_kl` creeps down, the prototype metrics are
  stable, and the probes improve. Judge a run on the probes, not on the loss.
