"""Combine completed no-training covariance screens and mean forward checks."""
import json
from pathlib import Path


def main():
    root = Path("outputs/linear_float_oracle_50um").resolve()
    output = root/"localized_covariance_audit"
    rows = []
    for n in (100,800):
        for width in (0,16,32,64):
            path = (root/"oracle_sensor_bandwidth"/f"empirical_transfer_n{n}.json" if width == 0
                    else output/f"n{n}_w{width}.json")
            s = json.loads(path.read_text())
            g = s["gradient_agreement"]
            scores = s["scores"]
            margins = {split:(scores["local_anchor"][split]["relative_mse"]-scores["truth"][split]["relative_mse"])
                            /scores["local_anchor"][split]["relative_mse"] for split in ("train","holdout")}
            passes = (g["train_holdout_cosine"] >= .5 and g["train_population_cosine"] >= .5
                      and g["holdout_population_cosine"] >= .5 and min(margins.values()) > 0)
            rows.append({"frames":n,"window_sigma":width,
                         **{key:g[key] for key in ("train_population_cosine","holdout_population_cosine","train_holdout_cosine")},
                         "truth_over_anchor_relative_margin":margins,"passes_engineering_prescreen":passes,
                         "source":str(path)})
    mean = json.loads((output/"linear_mean_forward_audit.json").read_text())
    report = {"complete":True, "pending":[], "new_no_training_covariance_cases":6,
              "new_reconstruction_runs":0,
              "scope":"Fixed known 50 um; no network updates; same stationary analytic Cs and per-probe relative MSE.",
              "prescreen_rule":"Require all three initial gradient cosines >=0.5 and truth ranked ahead of local anchor in both frame splits. Engineering screening heuristic, not an identifiability theorem.",
              "caution":"Each taper has a different objective; its numerical loss is not comparable with untapered loss. Gradient agreement is local to initialization and not an image-quality guarantee.",
              "rows":rows, "mean_forward_audit":mean,
              "decision":"Do not launch localized covariance reconstruction. A joint mean/covariance diagnostic would change the prior no-mean-backprop constraint and awaits user permission; not implemented or run."}
    (output/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"complete":True,"rows":rows,"decision":report["decision"]},indent=2))


if __name__ == "__main__":
    main()
