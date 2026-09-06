"""Summarize nested-probe and object-tangent audits, with explicit non-goals."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("outputs/linear_float_oracle_50um").resolve()
    output = root / "covariance_gradient_convergence"
    rows, full = [], {}
    for count in (100, 800):
        report = json.loads((output / f"n{count}/report.json").read_text())
        if not report["complete"]:
            raise RuntimeError("Incomplete gradient audit")
        full[count] = report
        previous = json.loads((root / f"exact_population_diagnostics/n{count}_w0.json").read_text())
        repeated = next(row for row in report["results"] if row["probes"] == 64 and row["loss"] == "per_probe")
        parity_error = max(abs(previous["gradient_agreement"][key] - value)
                           for key, value in repeated["full_gradient_agreement"].items())
        if parity_error > 1e-4:
            raise ValueError(f"64-probe regression failed: {parity_error}")
        for row in report["results"]:
            passed = [item for item in row["tangent_subspaces"] if item["direction_gate_pass"]]
            selected = passed[-1] if passed else None
            rows.append({"frames": count, "probes": row["probes"], "loss": row["loss"],
                         **row["full_gradient_agreement"],
                         "train_gradient_cosine_to_largest_prefix": row["cosine_to_largest_probe_prefix"]["train"],
                         "holdout_gradient_cosine_to_largest_prefix": row["cosine_to_largest_probe_prefix"]["holdout"],
                         "largest_passing_dct_side": selected["dct_side"] if selected else None,
                         "passing_subspace_population_gradient_energy": selected["retained_gradient_energy"]["population"] if selected else None,
                         "64_probe_previous_audit_max_cosine_difference": parity_error})
    result = {"complete": True, "rows": rows, "details": full,
              "scope": "Initial fixed-50-um covariance-only gradients. No reconstruction was trained in this audit. More probes are computational measurements, not additional acquired frames.",
              "interpretation": "A passing low-frequency DCT tangent subspace is NOT a resolution improvement. It suppresses other degrees of freedom and may retain only coarse intensity information. Largest probe prefix is a convergence reference, not the exact expectation."}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, count in enumerate((100, 800)):
        for mode, style in (("per_probe", "-o"), ("global_fixed", "--s")):
            subset = [row for row in rows if row["frames"] == count and row["loss"] == mode]
            axes[0, column].plot([r["probes"] for r in subset], [r["train_holdout_cosine"] for r in subset], style, label=mode)
            last = next(row for row in full[count]["results"] if row["loss"] == mode and row["probes"] == full[count]["arguments"]["probes"])
            axes[1, column].plot([r["dct_side"] for r in last["tangent_subspaces"]],
                                 [r["train_holdout_cosine"] for r in last["tangent_subspaces"]], style, label=mode)
        axes[0, column].set_title(f"N={count}: full object gradient")
        axes[0, column].set_xlabel("Sensor probes (same acquired frames)")
        axes[1, column].set_title(f"N={count}: fixed object-space restriction")
        axes[1, column].set_xlabel("DCT square side (larger = finer freedom)")
        for ax in axes[:, column]:
            ax.set_xscale("log", base=2); ax.axhline(.5, color="gray", linestyle=":")
            ax.set_ylabel("Train / holdout gradient cosine"); ax.set_ylim(-.5, 1)
            ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / "gradient_reliability.png", dpi=140)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
