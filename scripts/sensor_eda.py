"""Server-side sensor summaries. Run as python -m scripts.sensor_eda from the repo root.

Only a human runs this on competition data. Agents may test independent synthetic inputs.
No images, labels, sample IDs, Test split, remote services, or preprocessing fitting are used.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from schema import RAW_SENSOR_COLUMNS
from sensor_features import (
    SENSOR_FEATURE_VERSION,
    TEMPERATURE_SENSOR_FEATURES,
    build_sensor_features,
)


def summarize(frame: pd.DataFrame, feature_names: tuple[str, ...], bins: int) -> tuple:
    if frame.empty:
        raise ValueError("Sensor report requires at least one row.")
    if not 2 <= bins <= 200:
        raise ValueError("bins must be between 2 and 200.")
    features = build_sensor_features(frame, feature_names)
    summaries, histograms = [], {}
    for name in features:
        column = features[name]
        finite = column.dropna().to_numpy(dtype=float)
        row = {
            "feature": name,
            "rows": len(column),
            "valid_count": len(finite),
            "unavailable_count": int(column.isna().sum()),
            "unavailable_pct": float(column.isna().mean() * 100),
            "zero_count": int((finite == 0).sum()),
            "unique_valid_count": int(column.nunique()),
        }
        if name in RAW_SENSOR_COLUMNS:
            source = frame[name]
            missing = source.isna() | source.astype("string").str.strip().eq("").fillna(False)
            numeric = pd.to_numeric(source, errors="coerce")
            row.update(
                source_missing_count=int(missing.sum()),
                parse_error_count=int((~missing & numeric.isna()).sum()),
                infinite_count=int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()),
            )
        if len(finite):
            for key, percentile in (
                ("min", 0),
                ("p01", 1),
                ("p25", 25),
                ("median", 50),
                ("p75", 75),
                ("p99", 99),
                ("max", 100),
            ):
                row[key] = float(np.percentile(finite, percentile))
            counts, edges = np.histogram(finite, bins=bins)
        else:
            counts, edges = np.array([], dtype=int), np.array([], dtype=float)
        summaries.append(row)
        histograms[name] = {"counts": counts.tolist(), "edges": edges.tolist()}
    return pd.DataFrame(summaries).set_index("feature"), histograms, features


def save_plots(features: pd.DataFrame, output_dir: Path, bins: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    files = []
    for name in features:
        values = features[name].dropna().to_numpy(dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        fig.suptitle(name)
        if len(values):
            low, high = np.percentile(values, [1, 99])
            middle = values[(values >= low) & (values <= high)]
            axes[0].hist(values, bins=bins, color="#2367a8")
            axes[1].hist(middle, bins=bins, color="#159a8c")
            axes[1].set_title(f"P01-P99 view; {len(values) - len(middle)} excluded")
        else:
            for axis in axes:
                axis.text(0.5, 0.5, "No finite values", ha="center", transform=axis.transAxes)
            axes[1].set_title("P01-P99 view")
        axes[0].set_title("Full finite range")
        for axis in axes:
            axis.set_xlabel("Feature value (source units)")
            axis.set_ylabel("Count")
        filename = f"{name}.png"
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)
        files.append(filename)
    return files


def generate_report(
    data_dir: Path,
    split: str,
    feature_set: str,
    output_dir: Path,
    bins: int = 40,
    plots: bool = False,
) -> None:
    if split not in ("train", "validation"):
        raise ValueError("EDA supports train or validation only; Test is inference-only.")
    if feature_set not in ("raw", "temperature"):
        raise ValueError("feature_set must be raw or temperature.")
    if plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-eda.txt for --plots, or omit --plots."
            ) from error
    if output_dir.exists():
        raise FileExistsError("Report directory already exists. Choose a new output directory.")
    # Explicit columns: never load labels, identifiers, image paths, or unknown extra columns.
    frame = pd.read_csv(
        data_dir / split / f"{split}.csv",
        usecols=list(RAW_SENSOR_COLUMNS),
        dtype="string",
        keep_default_na=False,
    )
    names = RAW_SENSOR_COLUMNS if feature_set == "raw" else TEMPERATURE_SENSOR_FEATURES
    summary, histograms, features = summarize(frame, names, bins)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output_dir / "summary.csv")
    (output_dir / "histograms.json").write_text(
        json.dumps(histograms, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "split": split,
                "feature_names": list(names),
                "feature_version": SENSOR_FEATURE_VERSION,
                "bins": bins,
                "plots": plots,
                "imputation_or_scaling_applied": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plots_html = ""
    if plots:
        plots_html = "".join(
            f'<img src="{html.escape(filename)}" alt="{html.escape(filename)}">'
            for filename in save_plots(features, output_dir, bins)
        )
    report = (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Sensor distribution report</title>"
        "<style>body{font:16px system-ui;margin:24px;background:#f7f9fc;color:#17233b}"
        "table{border-collapse:collapse;background:white}th,td{padding:8px;border:1px solid #ddd}"
        "img{display:block;max-width:100%;margin:24px 0}.table{overflow:auto}</style>"
        f"<h1>Sensor distribution: {html.escape(split)}</h1>"
        "<p>서버에서 직접 확인하는 실제 데이터 요약입니다. "
        "외부 AI나 공개 저장소에 공유하지 마세요.</p>"
        "<p>결측 대체·정규화 전의 유효값 분포입니다. unavailable은 결측·변환 실패·무한대 또는 "
        "원본 온도 누락으로 계산할 수 없는 파생값을 포함합니다. 1~99백분위 그래프는 시각화만 "
        "확대한 것으로 전처리의 클리핑 기준이 아닙니다.</p>"
        '<div class="table">'
        + summary.to_html(float_format=lambda x: f"{x:.4g}", escape=True)
        + "</div>"
        + plots_html
        + "</html>"
    )
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--feature_set", choices=("raw", "temperature"), default="temperature")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    generate_report(**vars(args))
    print("Sensor report saved. Review the output directory yourself on the server.")


if __name__ == "__main__":
    main()
