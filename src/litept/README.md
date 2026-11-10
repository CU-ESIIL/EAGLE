# Vendored LitePT

Standalone LitePT code copied from https://github.com/prs-eth/LitePT
(commit `436d048`, MIT license, see `LICENSE`), following the upstream
"Use LitePT in Your Own Project" instructions.

| file | upstream source |
| --- | --- |
| `model.py` | `litept/model.py` |
| `serialization/` | `litept/serialization/` |
| `pointrope/` | `libs/pointrope/` |
| `transform.py` | `datasets/transform.py` (augmentations / preprocessing) |
| `registry.py` | `utils/registry.py` (+ `is_seq_of` from `utils/misc.py`) |

EAGLE patches are marked with `EAGLE patch` comments:

- `flash_attn` is optional. GPUs without flash-attn support (V100) use a
  `torch.nn.functional.scaled_dot_product_attention` fallback. Force it with `EAGLE_ATTN=sdpa`.
- PointROPE uses the pure PyTorch implementation unless the CUDA kernel has been
  compiled (`cd src/litept/pointrope && python setup.py install`, set the arch list in `setup.py`).
- Imports were made package-relative; `colorhash` is only needed for visualization helpers.

ALS-specific transforms live in `src/eagle_als/transforms.py` and register into the same
`TRANSFORMS` registry, so they can be mixed with the LitePT transforms in configs.
