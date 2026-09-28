# MyoDigits

Source code for Amalia Contiero Syropoulou's MSc thesis,
*Continuous Hand Pose Regression from Wearable sEMG* (University of West Attica,
2026). The project estimates finger-joint angles from forearm sEMG using the
PiMForce dataset.

The thesis evaluates a Transformer (MyoDigits-TX), a bidirectional GRU
(MyoDigits-GRU), and a per-window model (MyoDigits-PW). It reports two
leave-one-subject-out protocols. Strict LOSO fits input and target statistics
on training participants only. Participant-calibrated LOSO also uses the
held-out participant's own input and labelled target statistics. The evaluated
pipeline is offline and non-causal.

MyoDigits-PW has no cross-window temporal decoder. Its source file retains the
historical name `src/myodigits/models/tcn_v1.py`.

This repository contains code and synthetic smoke tests. It does not contain
recordings, participant-level result records, cached predictions, checkpoints,
or pretrained weights. The numerical results are in the thesis and cannot be
recomputed from this clone alone.

## Repository guide

| Path | Contents |
|---|---|
| `src/myodigits/` | Models, data loading, training, and analysis code |
| `launch.py` and `scripts/` | Experiment entry point, preprocessing, evaluation, and profiling |
| `tests/` | Synthetic CPU smoke tests and import checks |

## Data

The PiMForce and EgoEMG datasets are not included. Obtain them from their
original sources. The pretrained stem for the transfer experiment and the
fixed-split EgoEMG evaluation records are also outside this repository.

## Quick start

Install the PyTorch build appropriate for your platform, then install the
remaining dependencies and run the synthetic tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Citation details are in `CITATION.cff`.
