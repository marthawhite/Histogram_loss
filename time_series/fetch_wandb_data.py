import argparse
import pandas as pd
import wandb
import json

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
ARCHS = [("linear", "Linear"),
         ("LSTM", "LSTM"),
         ("transformer", "Transformer")]
LOSSES = [("regression", r"$\ell_2$"),
          ("HL", "HL-Gaussian")]


def fetch_runs(entity: str, project: str) -> pd.DataFrame:
    api = wandb.Api(timeout=60)
    rows, skipped = [], []
    for run in api.runs(f"{entity}/{project}"):
        if run.state != "finished":
            continue
        cfg = run.config
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v)
               for k, v in cfg.items()}

        summary = run.summary._json_dict
        if isinstance(summary, str):
            summary = json.loads(summary)
        try:
            rows.append({
                "dataset":    cfg["dataset"],
                "base_model": cfg["base_model"],
                "loss":       cfg["loss"],
                "mse":        float(summary["test_mse"]),
                "mae":        float(summary["test_mae"]),
                "run_id":     run.id,
            })
        except (KeyError, TypeError) as e:
            skipped.append((run.id, f"missing field: {e}"))
            continue

    if skipped:
        print(f"[info] skipped {len(skipped)} run(s); first few: {skipped[:5]}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Average across seeds; switch to .min() if you'd rather report best run.
    return (df.groupby(["dataset", "base_model", "loss"], as_index=False)
              [["mse", "mae"]].mean())


def fmt(x: float) -> str:
    return f"{x:.4g}"


def to_latex(df: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lllrr}",
             r"\toprule",
             r"Task & Architecture & Loss & Test MSE & Test MAE \\",
             r"\midrule"]
    for di, ds in enumerate(DATASETS):
        first = True
        for arch_key, arch_label in ARCHS:
            for loss_key, loss_label in LOSSES:
                row = df[(df.dataset == ds) &
                         (df.base_model == arch_key) &
                         (df.loss == loss_key)]
                if row.empty:
                    continue
                mse, mae = row.iloc[0][["mse", "mae"]]
                task = ds if first else ""
                lines.append(
                    f"{task} & {arch_label} & {loss_label} & "
                    f"{fmt(mse)} & {fmt(mae)} \\\\")
                first = False
        lines.append(r"\midrule" if di < len(DATASETS) - 1 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def to_plain(df: pd.DataFrame) -> str:
    out = []
    header = f"{'Task':<8}{'Architecture':<14}{'Loss':<14}{'Test MSE':>10}{'Test MAE':>10}"
    out.append(header)
    out.append("-" * len(header))
    for ds in DATASETS:
        first = True
        for arch_key, arch_label in ARCHS:
            for loss_key, loss_label in LOSSES:
                row = df[(df.dataset == ds) &
                         (df.base_model == arch_key) &
                         (df.loss == loss_key)]
                if row.empty:
                    continue
                mse, mae = row.iloc[0][["mse", "mae"]]
                task = ds if first else ""
                plain_loss = "l2" if loss_key == "regression" else "HL-Gaussian"
                out.append(
                    f"{task:<8}{arch_label:<14}{plain_loss:<14}"
                    f"{fmt(mse):>10}{fmt(mae):>10}")
                first = False
        out.append("-" * len(header))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default ="esraa_",
                   help="wandb entity (your username or team)")
    p.add_argument("--project", default="hl_loss_results")
    p.add_argument("--tex", default=None, help="optional path to write LaTeX")
    p.add_argument("--csv", default=None,
                   help="optional path to write the aggregated CSV")
    args = p.parse_args()

    raw = fetch_runs(args.entity, args.project)
    if raw.empty:
        raise SystemExit("No finished runs with the expected fields were found.")

    df = aggregate(raw)
    if args.csv:
        df.to_csv(args.csv, index=False)

    print(to_plain(df))
    print()
    tex = to_latex(df)
    print(tex)
    if args.tex:
        with open(args.tex, "w") as f:
            f.write(tex)


if __name__ == "__main__":
    main()