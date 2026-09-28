# MyoDigits

Code for the MSc thesis *Continuous Hand Pose Regression from Wearable sEMG*.

## Code

| Path | Contents |
|---|---|
| `src/myodigits/` | Models, data loading, training, and analysis |
| `launch.py` and `scripts/` | Experiments, preprocessing, evaluation, and profiling |
| `tests/` | Synthetic CPU tests |

## Data

- PiMForce: [Seo et al. (2024)](https://arxiv.org/abs/2410.23629)
- EgoEMG: [Xi et al. (2026)](https://arxiv.org/abs/2605.05712)

## Quick start

Install PyTorch for your platform, then run:

```bash
pip install -r requirements-dev.txt
python -m pytest
```
