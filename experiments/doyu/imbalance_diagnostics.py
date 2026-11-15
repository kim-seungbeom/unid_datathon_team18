"""클래스 불균형 진단: 분포, 실험별 구성, OOF 혼동행렬, 예측 비율, 오류 방향.

읽는 것: Train/Validation CSV의 label·experiment_id, prior_correction oof의 oof.npz.
Validation은 분포 집계만 하고 예측하지 않는다. Test는 읽지 않는다. 아무것도 학습하지 않는다.
출력은 사람이 tmux에서 읽는다.

  python experiments/doyu/imbalance_diagnostics.py \
    --data_dir /home/ubuntu/data/battery_data_light --oof_dir runs/doyu/tau_oof \
    [--correction runs/doyu/tau_select/correction.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from schema import LABELS, read_frame, validate_labels  # noqa: E402

K = len(LABELS)


def counts(labels):
    labels = validate_labels(labels)
    return np.array([(labels == label).sum() for label in LABELS], dtype=np.int64)


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def row(name, values, fmt="{:>8}"):
    print(f"{name:<22}" + "".join(fmt.format(v) for v in values))


def class_header(first="", extra=()):
    row(first, [f"c{label}" for label in LABELS] + list(extra))


def distribution(frames, fold_frame):
    header("1. 클래스 분포 (행 수 / 비율)")
    class_header("split", ["max/min"])
    for name, frame in frames.items():
        c = counts(frame["label"])
        row(name + " rows", list(c) + [f"{c.max() / max(c.min(), 1):.1f}x"])
        row(name + " share", [f"{v:.3f}" for v in c / c.sum()] + [""])
    for f in sorted(fold_frame["fold"].unique()):
        c = counts(fold_frame.loc[fold_frame["fold"] == f, "label"])
        row(f"oof fold {f} share", [f"{v:.3f}" for v in c / c.sum()] + [f"n={c.sum()}"])


def experiments(frames):
    header("2. 실험(experiment_id)별 클래스 구성")
    for name, frame in frames.items():
        groups = frame.groupby("experiment_id")["label"]
        n_exp = frame["experiment_id"].nunique()
        print(f"[{name}] 실험 수 = {n_exp}")
        class_header("")
        per_class_exp = [frame.loc[frame.label == c, "experiment_id"].nunique() for c in LABELS]
        row("#experiments w/ class", per_class_exp)
        table = frame.groupby(["experiment_id", "label"]).size().unstack(fill_value=0)
        table = table.reindex(columns=list(LABELS), fill_value=0)
        med = [int(table.loc[table[c] > 0, c].median()) if (table[c] > 0).any() else 0
               for c in LABELS]
        row("median rows/exp", med)
        top1 = [f"{table[c].max() / max(table[c].sum(), 1):.2f}" for c in LABELS]
        row("top-1 exp share", top1)
        top3 = [f"{table[c].nlargest(3).sum() / max(table[c].sum(), 1):.2f}" for c in LABELS]
        row("top-3 exp share", top3)
        n_classes = groups.nunique()
        print(f"  실험당 포함 클래스 수: min={n_classes.min()} median={int(n_classes.median())} "
              f"max={n_classes.max()}")


def confusion(labels, pred):
    m = np.zeros((K, K), dtype=np.int64)
    np.add.at(m, (validate_labels(labels) - 1, validate_labels(pred) - 1), 1)
    return m


def show_confusion(name, labels, pred):
    m = confusion(labels, pred)
    header(f"3. OOF 혼동행렬 [{name}] (행=정답, 열=예측, 행 정규화)")
    class_header("true \\ pred", ["recall"])
    for i, label in enumerate(LABELS):
        row(f"c{label} (n={m[i].sum()})",
            [f"{v:.3f}" for v in m[i] / max(m[i].sum(), 1)] + [f"{m[i, i] / max(m[i].sum(), 1):.3f}"])
    true_n, pred_n = m.sum(1), m.sum(0)
    class_header("")
    row("pred/true count", [f"{p / max(t, 1):.2f}" for p, t in zip(pred_n, true_n)])
    precision = np.diag(m) / np.maximum(pred_n, 1)
    row("precision", [f"{v:.3f}" for v in precision])
    return m


def error_direction(name, m):
    header(f"4. 오류 방향 [{name}] (틀린 행 중 비율)")
    majority = int(np.argmax(m.sum(1)))
    print(f"  다수 클래스 = c{LABELS[majority]}")
    class_header("", ["n_err"])
    adj, maj, far = [], [], []
    errs = []
    for i in range(K):
        wrong = m[i].copy()
        wrong[i] = 0
        total = wrong.sum()
        errs.append(total)
        a = sum(wrong[j] for j in (i - 1, i + 1) if 0 <= j < K)
        mj = wrong[majority] if majority not in (i - 1, i, i + 1) else 0
        adj.append(a / max(total, 1))
        maj.append(mj / max(total, 1))
        far.append((total - a - mj) / max(total, 1))
    row("-> 인접 단계(±1)", [f"{v:.2f}" for v in adj] + [""])
    row("-> 다수 클래스(비인접)", [f"{v:.2f}" for v in maj] + [""])
    row("-> 그 외 먼 단계", [f"{v:.2f}" for v in far] + [""])
    row("틀린 행 수", errs + [int(sum(errs))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--oof_dir", type=Path, required=True)
    parser.add_argument("--correction", type=Path)
    args = parser.parse_args(argv)

    frames = {
        "train": read_frame(args.data_dir / "train/train.csv", labeled=True),
        "validation": read_frame(args.data_dir / "validation/validation.csv", labeled=True),
    }
    z = np.load(args.oof_dir / "oof.npz", allow_pickle=False)
    labels, fold = z["labels"], z["fold"]
    oof_frame = pd.DataFrame({"label": labels, "fold": fold})
    distribution(frames, oof_frame)
    experiments(frames)

    neural, tree = z["neural"], z["extra_trees"]
    members = {
        "neural (proxy)": neural,
        "extra_trees": tree,
        "blend 40:60": 0.4 * neural + 0.6 * tree,
    }
    if args.correction is not None:
        c = json.loads(args.correction.read_text(encoding="utf-8"))
        prior = np.asarray(c["prior"])
        members[f"blend 40:60 + tau={c['tau']:.2f}"] = (0.4 * neural + 0.6 * tree) / prior ** c["tau"]
    for name, p in members.items():
        pred = np.asarray(p).argmax(1) + 1
        m = show_confusion(name, labels, pred)
        error_direction(name, m)

    header("읽는 법")
    print("- 1: fold별 share가 크게 흔들리면 OOF로 고른 값의 신뢰도가 낮다.")
    print("- 2: '#experiments w/ class'가 작거나 top-1 share가 크면 행 수보다 실험 수가 부족한 것.")
    print("- 3: 소수 클래스의 pred/true < 1 이고 recall이 낮으면 과소 예측(불균형 영향).")
    print("- 4: 오류 대부분이 인접 단계면 경계 모호성 문제(불균형 대응 효과 제한적),")
    print("     다수 클래스로 많이 가면 불균형 대응(가중치/logit adjustment)이 효과적일 가능성.")


if __name__ == "__main__":
    main()
