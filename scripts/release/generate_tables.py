#!/usr/bin/env python3
"""Generate the displayed result tables from the published artifacts.

Nothing in RESULTS.md between ``<!-- BEGIN GENERATED: name -->`` and
``<!-- END GENERATED: name -->`` is written by hand, and REGISTRY.md is written
entirely by this script. ``--check`` exits non-zero if either file differs from
a fresh generation, so documentation cannot drift from the artifacts.

Inputs (all in this repository):
    results/registry.json
    results/<experiment>/fold*_metrics.json and loso_summary.json
    results/analysis/r2/<config>.json            (per-joint R² statistics)
    results/analysis/r2_reconciliation.json
    results/analysis/strict_tx_sensitivity.json

Usage
    python scripts/release/generate_tables.py           # rewrite RESULTS.md blocks and REGISTRY.md
    python scripts/release/generate_tables.py --check   # fail on drift
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from myodigits.analysis import paired  # noqa: E402
from myodigits.analysis import r2_variants as r2  # noqa: E402

RESULTS = REPO / "results"


# ── loading ──────────────────────────────────────────────────────────────────

def fold_records(name):
    recs = {}
    for f in sorted((RESULTS / name).glob("fold*_testUser*_metrics.json")):
        user = int(f.name.split("testUser")[1][:2])
        recs[user] = json.loads(f.read_text())
    return recs


def mae_vector(name):
    recs = fold_records(name)
    return np.array([recs[u]["test_mae"] for u in sorted(recs)]), sorted(recs)


def jc_r2(name):
    data = json.loads((RESULTS / "analysis" / "r2" / f"{name}.json").read_text())
    return r2.loso_mean([r2.r2_from_statistics(f) for f in data["folds"]], r2.JOINT_CENTRED)


def stored_jc_r2(name):
    """Joint-centred aggregate R² read straight from the fold records.

    Only valid for a run whose evaluation code wrote the joint-centred variant to
    ``test_r2``; the registry records which variant each run stored, and this
    refuses any other. Used for the strict runs, whose per-joint sufficient
    statistics are not published.
    """
    reg = json.loads((RESULTS / "registry.json").read_text())
    entry = next(e for e in reg["entries"] if e["id"] == name)
    if entry["stored_test_r2_variant"] != "joint_centred_aggregate":
        raise SystemExit(f"{name}: stored test_r2 is "
                         f"{entry['stored_test_r2_variant']}, not the joint-centred aggregate")
    recs = fold_records(name)
    return float(np.mean([recs[u]["test_r2"] for u in sorted(recs)]))


def f2(x):
    return f"{x:.2f}"


def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ── tables ───────────────────────────────────────────────────────────────────

def t_definitive():
    rows = []
    for name, label in (("myodigits_tx", "MyoDigits-TX"), ("myodigits_gru", "MyoDigits-GRU"),
                        ("myodigits_tcn", "MyoDigits-PW")):
        recs = fold_records(name)
        mae = np.array([r["test_mae"] for r in recs.values()])
        rmse = np.array([r["test_rmse"] for r in recs.values()])
        pr = np.array([r["mean_pearson_r"] for r in recs.values()])
        rows.append([label, f2(mae.mean()), f2(mae.std(ddof=0)), f2(rmse.mean()),
                     f"{jc_r2(name):.3f}", f"{pr.mean():.3f}", f2(mae.min()), f2(mae.max())])
    return ("Thesis Table 6.1, participant-calibrated rows. Mean over 21 held-out "
            "participants; SD is across participants (ddof=0). R² is the joint-centred "
            "aggregate; Pearson r is the uniform mean over joints.\n\n"
            + table(["Model", "MAE (deg)", "SD", "RMSE", "R² (joint-centred)",
                     "Pearson r", "Best", "Worst"], rows))


def t_strict_definitive():
    rows = []
    for name, label in (("strict_norm_tx", "MyoDigits-TX"),
                        ("strict_norm_gru", "MyoDigits-GRU")):
        recs = fold_records(name)
        mae = np.array([r["test_mae"] for r in recs.values()])
        rmse = np.array([r["test_rmse"] for r in recs.values()])
        pr = np.array([r["mean_pearson_r"] for r in recs.values()])
        rows.append([label, f2(mae.mean()), f2(mae.std(ddof=0)), f2(rmse.mean()),
                     f"{stored_jc_r2(name):.3f}", f"{pr.mean():.3f}",
                     f2(mae.min()), f2(mae.max())])
    return ("Thesis Table 6.1, Strict LOSO rows. Input and target statistics are fitted on "
            "the 17 training participants of each fold and used unchanged to return "
            "predictions to degrees, so the held-out participant supplies nothing. R² is "
            "read from the fold records, whose evaluation code wrote the joint-centred "
            "aggregate to `test_r2`.\n\n"
            + table(["Model", "MAE (deg)", "SD", "RMSE", "R² (joint-centred)",
                     "Pearson r", "Best", "Worst"], rows))


def t_pretraining():
    rows = []
    for name, label in (("myodigits_tx", "From scratch"),
                        ("transfer_egoemg_finetuned", "Pretrained, fine-tuned"),
                        ("transfer_egoemg_frozen", "Pretrained, parameters frozen")):
        recs = fold_records(name)
        mae = np.array([r["test_mae"] for r in recs.values()])
        rmse = np.array([r["test_rmse"] for r in recs.values()])
        rows.append([label, f2(mae.mean()), f2(rmse.mean()), f"{jc_r2(name):.3f}",
                     f"{mae.max() / mae.min():.2f}x"])
    return "Thesis Table 8.1.\n\n" + table(
        ["Initialisation", "MAE (deg)", "RMSE", "R² (joint-centred)", "Worst/best"], rows)


def t_population():
    rows = []
    for name, n in (("train_population_04", "4"), ("train_population_08", "8"),
                    ("train_population_12", "12"), ("myodigits_tx", "17 (full)")):
        mae, _ = mae_vector(name)
        q75, q25 = np.percentile(mae, [75, 25])
        rows.append([n, f2(mae.mean()), f2(mae.min()), f2(mae.max()), f2(q75 - q25),
                     f"{mae.max() / mae.min():.2f}x"])
    return ("Thesis Table 8.2. IQR uses numpy's default linear interpolation.\n\n"
            + table(["Training participants", "MAE (deg)", "Best", "Worst", "IQR", "Worst/best"], rows))


def t_paired_tests():
    base, _ = mae_vector("myodigits_tx")
    rows = []
    for name, label in (("myodigits_gru", "MyoDigits-TX vs MyoDigits-GRU"),
                        ("myodigits_tcn", "MyoDigits-TX vs MyoDigits-PW"),
                        ("transfer_egoemg_finetuned", "MyoDigits-TX vs pretrained, fine-tuned"),
                        ("transfer_egoemg_frozen", "MyoDigits-TX vs pretrained, parameters frozen")):
        other, _ = mae_vector(name)
        w = paired.paired_wilcoxon(other, base)
        p = "< 0.0001" if w["p"] < 1e-4 else f"{w['p']:.3f}"
        rows.append([label, f"{(other - base).mean():+.2f} deg", p])
    return ("Thesis Table 8.4. Exact two-sided Wilcoxon signed-rank test on per-participant MAE; "
            "a positive difference means the compared model is worse. Bonferroni threshold for "
            "the four comparisons: 0.0125.\n\n"
            + table(["Comparison", "Mean difference", "Wilcoxon p"], rows))


def t_bootstrap():
    """Replicates scripts/analysis/bootstrap_ci.py exactly: one RandomState(0)
    shared across the conditions in this order, per-fold order of each summary."""
    order = [("myodigits_tx", "MyoDigits-TX"), ("myodigits_gru", "MyoDigits-GRU"),
             ("myodigits_tcn", "MyoDigits-PW"), ("transfer_egoemg_finetuned", "Pretrained, fine-tuned"),
             ("transfer_egoemg_frozen", "Pretrained, parameters frozen"),
             ("train_population_04", "4 training participants"),
             ("train_population_08", "8 training participants")]
    rng = np.random.RandomState(0)
    rows = []
    for name, label in order:
        s = json.loads((RESULTS / name / "loso_summary.json").read_text())
        v = np.array([f["test_mae"] for f in s["per_fold"]])
        sp, mn = [], []
        for _ in range(10000):
            x = v[rng.randint(0, len(v), len(v))]
            sp.append(x.max() / x.min())
            mn.append(x.mean())
        lo, hi = np.percentile(sp, [2.5, 97.5])
        mlo, mhi = np.percentile(mn, [2.5, 97.5])
        rows.append([label, f2(v.mean()), f"[{mlo:.2f}, {mhi:.2f}]",
                     f"{v.max() / v.min():.2f}x", f"[{lo:.2f}, {hi:.2f}]"])
    return ("Thesis Table 8.5. Percentile bootstrap over participants, 10,000 resamples, "
            "numpy RandomState(0) shared across conditions in the order shown. The ratio "
            "interval's upper bound equals the point estimate by construction.\n\n"
            + table(["Condition", "MAE", "95% CI", "Worst/best", "95% CI"], rows))


def t_posture_baselines():
    d = json.loads((RESULTS / "analysis" / "posture_baselines.json").read_text())
    label = {"training_statistics_only": "Strict", "participant_calibrated": "Calibrated"}
    rows = [[label[r["protocol"]], r["model"], f2(r["posture_mean_mae_deg"]),
             f2(r["model_mean_mae_deg"]), f"{r['reduction_pct_from_mean_maes']:.1f}"]
            for r in d["reductions"]]
    return ("Thesis Table 7.4. Mean over 21 equally weighted held-out participants. "
            "Reduction is 100 (posture - model) / posture computed from the unrounded "
            "means, not the average of participant-level percentages, which is recorded "
            "separately in the record. The calibrated posture is fitted on each test "
            "participant's own labels and is diagnostic; only the strict posture is "
            "label-free for the held-out participant.\n\n"
            + table(["Protocol", "Model", "Posture MAE (deg)", "Model MAE (deg)",
                     "Reduction (%)"], rows, align=["l", "l", "r", "r", "r"]))


def t_posture_per_participant():
    d = json.loads((RESULTS / "analysis" / "posture_baselines.json").read_text())
    rows = [[p["participant"], f"{p['strict_posture']['mae_deg']:.3f}",
             f"{p['calibrated_posture']['mae_deg']:.3f}"] for p in d["participants"]]
    a = d["aggregate"]
    rows.append(["**Mean**", f"**{a['strict_posture']['mean_mae_deg']:.3f}**",
                 f"**{a['calibrated_posture']['mean_mae_deg']:.3f}**"])
    return ("Thesis Table A.4. Constant-posture MAE in degrees for each held-out "
            "participant, scored on the same sequence positions as the neural runs. Means "
            "use unrounded participant values.\n\n"
            + table(["Participant", "Strict training-population posture",
                     "Calibrated own posture"], rows))


def t_difficulty_correlation():
    base, _ = mae_vector("myodigits_tx")
    rows = []
    for name, label in (("myodigits_gru", "MyoDigits-GRU"), ("myodigits_tcn", "MyoDigits-PW"),
                        ("transfer_egoemg_finetuned", "MyoDigits-TX with pretraining, fine-tuned"),
                        ("transfer_egoemg_frozen", "MyoDigits-TX with pretraining, parameters frozen")):
        other, _ = mae_vector(name)
        rows.append([label, f2(paired.associations(base, other)["pearson_r"])])
    return "Thesis Table 7.3.\n\n" + table(
        ["Model compared with MyoDigits-TX", "Pearson correlation of per-participant error"], rows)


def t_per_participant():
    names = ["myodigits_tx", "myodigits_gru", "myodigits_tcn", "transfer_egoemg_finetuned",
             "transfer_egoemg_frozen"]
    vecs = {n: mae_vector(n)[0] for n in names}
    rows = [[k + 1, f"P{k + 1:02d}"] + [f2(vecs[n][k]) for n in names] for k in range(21)]
    return "Thesis Table A.1. Test MAE in degrees.\n\n" + table(
        ["Fold", "Participant", "MyoDigits-TX", "MyoDigits-GRU", "MyoDigits-PW",
         "Pretrained FT", "Pretrained, params frozen"], rows, align=["r", "l", "r", "r", "r", "r", "r"])


def t_population_per_participant():
    a, _ = mae_vector("train_population_04")
    b, _ = mae_vector("train_population_08")
    c, _ = mae_vector("myodigits_tx")
    def change(new, old):   # difference of the displayed (rounded) values, as in the thesis
        return f"{round(new, 2) - round(old, 2):+.2f}"
    rows = [[f"P{k + 1:02d}", f2(a[k]), f2(b[k]), f2(c[k]), change(c[k], a[k])] for k in range(21)]
    rows.append(["Mean", f2(a.mean()), f2(b.mean()), f2(c.mean()), change(c.mean(), a.mean())])
    return ("Thesis Table A.2. Change is 17 minus 4 training participants, taken between the "
            "displayed two-decimal values as in the thesis.\n\n") + table(
        ["Participant", "4 train", "8 train", "17 train", "Change"], rows)


def t_exploration_stft():
    recs = fold_records("baseline_stft")
    cells = [f"P{u:02d} {f2(recs[u]['test_mae'])}" for u in sorted(recs)]
    mean = np.mean([r["test_mae"] for r in recs.values()])
    return ("Thesis Table 5.1, MyoDigits-TXS row (spectrogram front end with the attention "
            "decoder), three-participant panel. Exploratory; not a generalisation claim.\n\n"
            + table(["Configuration", "Per participant (deg)", "Panel mean"],
                    [["MyoDigits-TXS", ", ".join(cells), f2(mean)]]))


def t_r2():
    data = json.loads((RESULTS / "analysis" / "r2_reconciliation.json").read_text())
    rows = []
    for c in data["configurations"].values():
        v = c["variants"]
        rows.append([DISPLAY.get(c["label"], c["label"]), f2(c["mae_deg_loso_mean"]),
                     f"{v[r2.JOINT_CENTRED]['loso_mean']:.3f}", f"{v[r2.UNIFORM_MEAN]['loso_mean']:.3f}",
                     f"{v[r2.GLOBALLY_POOLED]['loso_mean']:.3f}"])
    return ("Thesis Table D.2. Each variant is computed within a held-out participant and "
            "averaged uniformly over the 21 folds.\n\n"
            + table(["Configuration", "MAE (deg)", "Joint-centred aggregate R²",
                     "Uniform mean per-joint R²", "Globally pooled R²"], rows))


def t_strict():
    s = json.loads((RESULTS / "analysis" / "strict_tx_sensitivity.json").read_text())
    p, a = s["paired"], s["association_calibrated_vs_strict"]
    rows = [
        ["Mean MAE, participant-calibrated", f"{s['calibrated']['mean']:.4f} deg"],
        ["Mean MAE, training statistics only", f"{s['strict']['mean']:.4f} deg"],
        ["Mean paired increase", f"{p['mean_difference_deg']:.4f} deg"],
        ["95% bootstrap interval", f"[{p['bootstrap_ci']['low']:.3f}, {p['bootstrap_ci']['high']:.3f}] deg"],
        ["Participants with higher MAE under strict", f"{p['participants_higher_under_strict']} of {p['participants_total']}"],
        ["Wilcoxon signed-rank", f"W = {p['wilcoxon']['W']:.0f}, p = {p['wilcoxon']['p']:.3e}"],
        ["SD across participants (ddof=0)", f"{s['calibrated']['sd_ddof0']:.3f} to {s['strict']['sd_ddof0']:.3f} deg"],
        ["Worst/best ratio", f"{s['calibrated']['worst_over_best_ratio']:.2f} to {s['strict']['worst_over_best_ratio']:.2f}"],
        ["Pearson r of per-participant MAE", f"{a['pearson_r']:.4f}"],
        ["Spearman rho of per-participant MAE", f"{a['spearman_rho']:.4f}"],
    ]
    return ("Paired comparison behind thesis Section 6.2 and Section 10.1, for MyoDigits-TX. "
            "Two separately trained runs, not a controlled "
            "ablation or an exact calibration penalty: training seeds were neither fixed nor "
            "repeated, so the difference also contains variation from initialisation, batch "
            "order and early stopping. Bootstrap: percentile, participants resampled, 10,000 "
            "resamples, seed 20260910. Wilcoxon: two-sided, zero_method wilcox, no continuity "
            "correction, exact.\n\n" + table(["Quantity", "Value"], rows))


# Reader-facing name for a label stored under the name used while the runs were
# executed. The archived artifacts keep their original strings.
DISPLAY = {"MyoDigits-TCN": "MyoDigits-PW"}


NUMERICALLY_VERIFIED = {"myodigits_tx", "myodigits_gru", "myodigits_tcn", "transfer_egoemg_finetuned",
                        "transfer_egoemg_frozen", "strict_norm_tx"}


def t_stored_r2_variant():
    reg = json.loads((RESULTS / "registry.json").read_text())
    rows = []
    for e in reg["entries"]:
        if not (RESULTS / e["id"]).is_dir():
            continue
        recs = fold_records(e["id"])
        pooled_folds = sorted(u for u, r in recs.items() if "test_r2_pooled" in r)
        if e["stored_test_r2_variant"] == "mixed":
            by_fold = e["stored_test_r2_variant_by_fold"]
            if pooled_folds != by_fold["joint_centred_aggregate"]:
                raise SystemExit(f"{e['id']}: registry fold split disagrees with the files")
            g, j = by_fold["globally_pooled"], by_fold["joint_centred_aggregate"]
            rows.append([f"`{e['id']}`",
                         f"**mixed**: globally pooled in folds {g[0]}-{g[-1]}, "
                         f"joint-centred aggregate in folds {j[0]}-{j[-1]}",
                         f"folds {j[0]}-{j[-1]} only", "by field layout"])
            continue
        expect_pooled = e["stored_test_r2_variant"] == "joint_centred_aggregate"
        if pooled_folds != (sorted(recs) if expect_pooled else []):
            raise SystemExit(f"{e['id']}: test_r2_pooled presence disagrees with the registry")
        rows.append([f"`{e['id']}`", e["stored_test_r2_variant"],
                     "yes" if expect_pooled else "no",
                     "numerically" if e["id"] in NUMERICALLY_VERIFIED else "by field layout"])
    return table(["Experiment", "Variant in `test_r2`", "`test_r2_pooled` present", "Verified"],
                 rows, align=["l", "l", "l", "l"])


BLOCKS = {
    "strict_definitive": t_strict_definitive,
    "definitive": t_definitive, "pretraining": t_pretraining, "population": t_population,
    "paired_tests": t_paired_tests, "bootstrap": t_bootstrap,
    "posture_baselines": t_posture_baselines,
    "posture_per_participant": t_posture_per_participant,
    "difficulty_correlation": t_difficulty_correlation, "per_participant": t_per_participant,
    "population_per_participant": t_population_per_participant,
    "exploration_stft": t_exploration_stft, "r2_variants": t_r2, "strict_tx": t_strict,
    "stored_r2_variant": t_stored_r2_variant,
}


def registry_markdown():
    reg = json.loads((RESULTS / "registry.json").read_text())
    out = ["# Run and claim registry", "",
           "Generated from `results/registry.json` by `scripts/release/generate_tables.py`. "
           "Do not edit by hand.", "", "## Roles", ""]
    out += [f"* **{k}**: {v}" for k, v in reg["roles"].items()]
    out += ["", "## Protocols", ""]
    out += [f"* **`{k}`**: {v}" for k, v in reg["protocols"].items()]
    out += ["", "## Training seeds", "", reg["training_seed_policy"], "", "## Entries", ""]
    rows = []
    for e in reg["entries"]:
        rows.append([f"`{e['id']}`", e["role"], f"`{e['run_id']}`", e["participant_protocol"],
                     e["normalisation_protocol"], e["n_folds"],
                     f"`{e['primary_artifact']}`" if e["primary_artifact"] else "none",
                     "; ".join(e["thesis_locations"]) or "not reported"])
    out.append(table(["ID", "Role", "Run", "Participants", "Normalisation", "Folds",
                      "Primary artifact", "Thesis"], rows))
    out += ["", "## Details", ""]
    for e in reg["entries"]:
        out += [f"### `{e['id']}`", "", f"* Model: {e['model']}", f"* Configuration: {e['configuration']}",
                "* Analysis scripts: " + (", ".join(f"`{s}`" for s in e["analysis_scripts"]) or "none")]
        if e.get("notes"):
            out.append(f"* Notes: {e['notes']}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_results(text):
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n)(?:.*?\n)?(<!-- END GENERATED: {name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"RESULTS.md has no block named {name}")
        text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    res_path, reg_path = REPO / "RESULTS.md", REPO / "REGISTRY.md"
    new_results = render_results(res_path.read_text())
    new_registry = registry_markdown()
    if args.check:
        stale = [p.name for p, new in ((res_path, new_results), (reg_path, new_registry))
                 if not p.exists() or p.read_text() != new]
        if stale:
            print("STALE:", ", ".join(stale))
            return 1
        print("up to date: RESULTS.md, REGISTRY.md")
        return 0
    res_path.write_text(new_results)
    reg_path.write_text(new_registry)
    print("wrote RESULTS.md generated blocks and REGISTRY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
