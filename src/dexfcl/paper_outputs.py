"""Numerical reproduction from the archived JNSM CSVs, without model training.

The canonical files are explicit: overlapping ablation exports are never pooled.
Missing timing and undetected drift delay are preserved as missing observations.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

SEEDS = {7, 19, 31, 43, 59}
DATASETS = {"ciciot2023", "edgeiiotset"}
METHODS = {"fedavg", "fedprox", "scaffold", "fedavg_ewc", "dexfcl"}
LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
          "fedavg_ewc": "FedAvg+EWC", "dexfcl": "DEx-FCL"}
SPECS = {
    "primary": ("primary_runs.csv", ["dataset", "method"],
                ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "mcc", "auroc"]),
    "noniid": ("noniid_runs.csv", ["dataset", "method", "alpha"],
               ["macro_f1", "accuracy", "mcc", "auroc"]),
    "continual": ("continual_runs.csv", ["dataset", "method"],
                 ["final_macro_f1", "avg_forgetting", "bwt"]),
    "zero_family": ("zero_day_runs.csv", ["dataset", "heldout"],
                    [f"{det}_{metric}" for det in ["MSP", "Energy", "Prototype", "Combined"]
                     for metric in ["auroc", "aupr", "unknown_recall", "known_fpr", "fpr95"]]),
    "assimilation": ("assimilation_runs.csv", ["dataset", "shots"],
                     ["new_f1", "historical_macro_f1", "overall_macro_f1"]),
    "drift": ("drift_runs.csv", ["dataset", "mode"], ["detected", "false_alarm_rate"]),
    "explanation": ("explanation_runs.csv", ["dataset", "method"],
                    ["spearman", "kendall", "jaccard10"]),
    "ablation": ("ablation_all_runs.csv", ["dataset", "configuration"],
                 ["macro_f1", "accuracy", "mcc", "auroc"]),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_archive(results):
    root = Path(results)
    frames = {}
    for kind, (name, keys, metrics) in SPECS.items():
        d = pd.read_csv(root / name)
        required = keys + ["seed"] + metrics
        _require(set(required).issubset(d.columns), f"{name}: missing columns {set(required) - set(d.columns)}")
        _require(set(d.dataset) == DATASETS, f"{name}: expected both benchmark datasets")
        _require(not d[keys + ["seed"]].isna().any().any(), f"{name}: missing experiment key")
        _require(not d.duplicated(keys + ["seed"]).any(), f"{name}: duplicate experiment key")
        for key, g in d.groupby(keys, dropna=False):
            _require(len(g) == 5 and set(g.seed) == SEEDS, f"{name}: incomplete seeds for {key}")
        for metric in metrics:
            values = pd.to_numeric(d[metric], errors="raise").to_numpy(dtype=float)
            _require(np.isfinite(values).all(), f"{name}: non-finite scientific metric {metric}")
            lo = -1 if metric in {"mcc", "bwt", "spearman", "kendall"} else 0
            _require(((values >= lo) & (values <= 1)).all(), f"{name}: out-of-range {metric}")
        frames[kind] = d
    expected = {
        "primary": ("method", METHODS),
        "noniid": ("method", METHODS - {"fedavg_ewc"}),
        "continual": ("method", {"fedavg", "fedavg_ewc", "dexfcl"}),
        "explanation": ("method", {"fedavg", "dexfcl"}),
        "drift": ("mode", {"entropy", "latent", "dual"}),
        "ablation": ("configuration", {"full", "no_adaptive", "no_drift", "no_retention", "no_explanation"}),
        "assimilation": ("shots", {5, 10, 25, 50, 100}),
    }
    for kind, (column, values) in expected.items():
        for ds, d in frames[kind].groupby("dataset"):
            _require(set(d[column]) == values, f"{kind}/{ds}: incomplete {column} levels")
    for key, d in frames["noniid"].groupby(["dataset", "method"]):
        _require(set(d.alpha) == {0.1, 0.3, 0.5, 1.0}, f"noniid/{key}: incomplete alpha levels")
    families = {
        "ciciot2023": {"BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"},
        "edgeiiotset": {"DoS_DDoS", "Information_Gathering", "Injection", "MITM", "Malware"},
    }
    for ds, d in frames["zero_family"].groupby("dataset"):
        _require(set(d.heldout) == families[ds], f"zero_family/{ds}: incomplete held-out families")
    comm = pd.read_csv(root / "communication.csv")
    _require(len(comm) == 2 and set(comm.dataset) == DATASETS, "communication: expected one row per dataset")
    for c in ["parameters", "fedavg_bytes_per_round", "dexfcl_bytes_per_round", "overhead_pct"]:
        _require(np.isfinite(comm[c]).all() and (comm[c] > 0).all(), f"communication: invalid {c}")
    frames["communication"] = comm
    return frames


def _summary(d, keys, metrics):
    out = d.groupby(keys, sort=True)[metrics].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def _statistics(primary):
    rows = []
    for ds, d in primary.groupby("dataset"):
        p = d.pivot(index="seed", columns="method", values="macro_f1").sort_index()
        cols = sorted(METHODS)
        f = friedmanchisquare(*[p[c] for c in cols])
        rows.append(dict(dataset=ds, comparison="Friedman", statistic=float(f.statistic),
                         raw_p=float(f.pvalue), adjusted_p=float(f.pvalue)))
        pairs = []
        for method in cols:
            if method == "dexfcl":
                continue
            delta = p.dexfcl - p[method]
            w = wilcoxon(p.dexfcl, p[method], method="auto") if (delta != 0).any() else None
            pairs.append(dict(dataset=ds, comparison=f"DEx-FCL vs {LABELS[method]}",
                              statistic=float(w.statistic) if w else 0.0,
                              raw_p=float(w.pvalue) if w else 1.0,
                              median_delta=float(delta.median())))
        running = 0.0
        for rank, i in enumerate(sorted(range(len(pairs)), key=lambda i: pairs[i]["raw_p"])):
            running = max(running, min(1.0, (len(pairs) - rank) * pairs[i]["raw_p"]))
            pairs[i]["adjusted_p"] = running
        rows.extend(pairs)
    return pd.DataFrame(rows)


def write_tables(frames, tables):
    t = Path(tables)
    t.mkdir(parents=True, exist_ok=True)
    for kind, (_, keys, metrics) in SPECS.items():
        _summary(frames[kind], keys, metrics).to_csv(t / f"table_{kind}_summary.csv", index=False)
    # Zero-day summaries pool the observed family/seed rows, as in the archive.
    rows = []
    d = frames["zero_family"]
    for ds, g in d.groupby("dataset"):
        for detector in ["MSP", "Energy", "Prototype", "Combined"]:
            row = dict(dataset=ds, detector=detector, n=len(g))
            for metric in ["auroc", "aupr", "unknown_recall", "known_fpr", "fpr95"]:
                row[f"{metric}_mean"] = g[f"{detector}_{metric}"].mean()
                row[f"{metric}_std"] = g[f"{detector}_{metric}"].std(ddof=1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(t / "table_zero_score_summary.csv", index=False)
    # A missing delay for a detector that never triggered is not zero delay.
    drift = frames["drift"]
    _require("delay_samples" in drift, "drift: missing delay_samples")
    _require(drift.loc[drift.detected.astype(bool), "delay_samples"].notna().all(),
             "drift: detected event without delay")
    delay = _summary(drift, ["dataset", "mode"], ["detected", "false_alarm_rate", "delay_samples"])
    delay.to_csv(t / "table_drift_summary.csv", index=False, na_rep="NA")
    frames["communication"].to_csv(t / "table_communication_summary.csv", index=False)
    _statistics(frames["primary"]).to_csv(t / "table_statistics.csv", index=False)


def make_paper_figures(frames, figures, dpi=600):
    f = Path(figures)
    f.mkdir(parents=True, exist_ok=True)
    _require(dpi >= 600, "Paper raster exports require at least 600 dpi")
    def save(fig, name):
        fig.tight_layout()
        for suffix in ["png", "pdf", "eps"]:
            target = f / f"{name}.{suffix}"
            temporary = f / f"{name}.tmp.{suffix}"
            fig.savefig(temporary, dpi=dpi)
            _require(temporary.stat().st_size > 0, f"Empty figure export: {name}.{suffix}")
            temporary.replace(target)
        plt.close(fig)
    def bars(d, key, metric, ylabel, name):
        s = d.groupby(key, sort=True)[metric].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar([LABELS.get(v, v.replace("_", " ")) for v in s.index], s["mean"], yerr=s["std"], capsize=3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=22)
        save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (ds, d) in zip(axes, frames["primary"].groupby("dataset", sort=True)):
        s = d.groupby("method").macro_f1.agg(["mean", "std"])
        ax.bar([LABELS[v] for v in s.index], s["mean"], yerr=s["std"], capsize=3)
        ax.set_xlabel("CICIoT2023" if ds == "ciciot2023" else "Edge-IIoTset")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Macro-F1")
    save(fig, "fig_primary_macrof1")
    for ds in sorted(DATASETS):
        d = frames["noniid"].query("dataset == @ds")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for method, g in d.groupby("method"):
            s = g.groupby("alpha").macro_f1.agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", label=LABELS[method], capsize=3)
        ax.set_xlabel("Dirichlet alpha"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1)
        ax.legend(framealpha=1)
        save(fig, f"fig_noniid_{ds}")
        bars(frames["continual"].query("dataset == @ds"), "method", "avg_forgetting", "Average forgetting", f"fig_forgetting_{ds}")
        bars(frames["ablation"].query("dataset == @ds"), "configuration", "macro_f1", "Macro-F1", f"fig_ablation_{ds}")
        d = frames["zero_family"].query("dataset == @ds")
        s = d.groupby("heldout").Combined_auroc.agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.barh(s.index, s["mean"], xerr=s["std"], capsize=3)
        ax.set_xlabel("Combined-score zero-day AUROC")
        ax.set_xlim(0, max(1.05, float((s['mean']+s['std']).max())+0.03))
        save(fig, f"fig_zero_family_{ds}")
        d = frames["assimilation"].query("dataset == @ds")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        upper = 1.05
        for metric, label in [("new_f1", "New-family F1"), ("historical_macro_f1", "Historical Macro-F1")]:
            s = d.groupby("shots")[metric].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", label=label, capsize=3)
            upper = max(upper, float((s['mean']+s['std']).max())+0.03)
        ax.set_xlabel("Verified support samples"); ax.set_ylabel("F1 score"); ax.set_ylim(0, upper)
        ax.legend(framealpha=1)
        save(fig, f"fig_assimilation_{ds}")


def reproduce_archive(results, tables, figures, dpi=600):
    r, t, f = Path(results).resolve(), Path(tables).resolve(), Path(figures).resolve()
    _require(t != f and r not in t.parents and r not in f.parents and t != r and f != r,
             "Use separate output directories outside the archived results directory")
    frames = load_archive(r)
    write_tables(frames, t)
    make_paper_figures(frames, f, dpi)
    sources = [r / name for name, _, _ in SPECS.values()] + [r / "communication.csv"]
    outputs = sorted(t.glob("*.csv")) + sorted(p for p in f.glob("*.*") if '.tmp.' not in p.name)
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "scope": "numerical reproduction of archived CSVs; no model training",
        "training_provenance_verified": False,
        "seeds": sorted(SEEDS), "raster_dpi": dpi,
        "source_sha256": {p.name: digest(p) for p in sources},
        "output_sha256": {("tables/" if p.parent == t else "figures/") + p.name: digest(p) for p in outputs},
        "row_counts": {kind: len(d) for kind, d in frames.items()},
        "software": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "matplotlib"]},
        "missingness": {
            "zero_day_seconds_missing": int(frames["zero_family"].seconds.isna().sum()),
            "drift_delay_missing": int(frames["drift"].delay_samples.isna().sum()),
        },
    }
    (t / "paper_outputs_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
