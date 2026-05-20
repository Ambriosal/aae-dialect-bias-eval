"""match model csvs to 1k gold, print metrics, write figures via viz.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

import viz
from gold_dataset import load_gold_rows
from labels import LABEL_ORDER, canon_pred_maybe
from text_norm import norm_text

MODEL_FILES: dict[str, str] = {
    "gemma3_vllm": "gemma3_vllm_aae_results.csv",
    "mistral31_vllm": "mistral31_vllm_aae_results.csv",
    "deepseekr1_vllm": "deepseekr1_vllm_aae_results.csv",
    "phi3_medium": "phi3_medium_aae_results.csv",
    "llama31_70b_vllm": "llama31_70b_vllm_aae_results.csv",
    "phi4_vllm": "phi4_vllm_aae_results.csv",
}


def load_preds_csv(path: Path) -> tuple[dict[str, str | None], dict[str, Any]]:
    preds: dict[str, str | None] = {}
    n_dup_same = 0
    conflicts: list[tuple[str, str | None, str | None]] = []
    n_bad_label = 0
    bad_samples: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "aae_text" not in rdr.fieldnames or "sentiment" not in rdr.fieldnames:
            raise ValueError(f"{path.name}: need columns aae_text, sentiment")
        for row in rdr:
            key = norm_text(row["aae_text"])
            raw = row["sentiment"]
            lab = canon_pred_maybe(raw)
            if lab is None and raw.strip() != "":
                n_bad_label += 1
                if len(bad_samples) < 5:
                    bad_samples.append(raw.strip()[:40])
            if key not in preds:
                preds[key] = lab
            elif preds[key] == lab:
                n_dup_same += 1
            else:
                conflicts.append((key[:80], preds[key], lab))

    meta = {
        "path": str(path),
        "rows_unique_keys": len(preds),
        "duplicate_rows_same_label": n_dup_same,
        "conflicting_duplicate_labels": len(conflicts),
        "conflict_samples": conflicts[:5],
        "n_cells_bad_sentiment_token": n_bad_label,
        "bad_sentiment_samples": bad_samples,
    }
    return preds, meta


def build_eval_records(
    gold: list[dict[str, Any]],
    preds: dict[str, str | None],
) -> list[dict[str, Any]]:
    rows = []
    for r in gold:
        k = r["text_key"]
        gl = r["gold_label"]
        sub = r["subset"]
        gid = r["gold_id"]
        if k not in preds:
            rows.append(
                {
                    "gold_id": gid,
                    "subset": sub,
                    "gold_label": gl,
                    "pred": None,
                    "correct": False,
                }
            )
            continue
        p = preds[k]
        if p is None:
            rows.append(
                {
                    "gold_id": gid,
                    "subset": sub,
                    "gold_label": gl,
                    "pred": None,
                    "correct": False,
                }
            )
        else:
            rows.append(
                {
                    "gold_id": gid,
                    "subset": sub,
                    "gold_label": gl,
                    "pred": p,
                    "correct": p == gl,
                }
            )
    return rows


def pred_mix_labels(gold: list[dict[str, Any]], preds: dict[str, str | None]) -> list[str]:
    out: list[str] = []
    for r in gold:
        k = r["text_key"]
        if k not in preds or preds[k] is None:
            out.append("invalid")
        else:
            out.append(preds[k] or "invalid")
    return out


def eval_against_gold(
    gold: list[dict[str, Any]],
    preds: dict[str, str | None],
) -> tuple[list[str], list[str], list[dict[str, Any]], int]:
    y_true: list[str] = []
    y_pred: list[str] = []
    missing: list[dict[str, Any]] = []
    n_invalid = 0
    for r in gold:
        k = r["text_key"]
        if k not in preds:
            missing.append(r)
            continue
        p = preds[k]
        y_true.append(r["gold_label"])
        if p is None:
            n_invalid += 1
            y_pred.append("__invalid__")
        else:
            y_pred.append(p)
    return y_true, y_pred, missing, n_invalid


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    out.mkdir(exist_ok=True)

    gold = load_gold_rows(root)
    gold_labels = [r["gold_label"] for r in gold]

    summary_lines: list[str] = []
    grid_panels: list[tuple[str, list[str], list[str]]] = []
    cmp_names: list[str] = []
    acc_list: list[float] = []
    f1_list: list[float] = []
    per_f1_rows: list[tuple[float, float, float]] = []
    subset_mat: list[np.ndarray] = []

    for model_name, fname in MODEL_FILES.items():
        path = root / fname
        preds, pmeta = load_preds_csv(path)
        records = build_eval_records(gold, preds)
        subset_mat.append(viz.subset_accuracy_vector(records))

        y_true, y_pred, missing, n_invalid = eval_against_gold(gold, preds)

        print(f"\n=== {model_name} ===")
        print(
            "pred file meta:",
            {k: v for k, v in pmeta.items() if k not in ("conflict_samples", "bad_sentiment_samples")},
        )
        if pmeta.get("bad_sentiment_samples"):
            print("non-empty unmapped sentiment samples:", pmeta["bad_sentiment_samples"])
        if pmeta["conflicting_duplicate_labels"]:
            print("WARNING: same text_key -> different labels in model csv, first label kept")
            for s in pmeta["conflict_samples"]:
                print(" ", s)

        n = len(gold)
        m = len(missing)
        matched = n - m
        print(
            f"gold coverage: {matched}/{n} matched, {m} missing, "
            f"{n_invalid} ERROR/invalid token on matched rows"
        )
        if missing:
            miss_path = out / f"{model_name}_missing_gold_rows.csv"
            with miss_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["gold_id", "subset", "gold_label", "aae_text"],
                )
                w.writeheader()
                for r in missing:
                    w.writerow(
                        {
                            "gold_id": r["gold_id"],
                            "subset": r["subset"],
                            "gold_label": r["gold_label"],
                            "aae_text": r["aae_text"],
                        }
                    )
            print(f"wrote {miss_path.name}")

        viz.label_mix_stacked_bars(
            model_name,
            gold_labels,
            pred_mix_labels(gold, preds),
            out / f"{model_name}_label_mix.png",
        )

        if not y_true:
            print("no matched rows, skip confusion + top-line metrics")
            continue

        ok_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "__invalid__"]
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

        if ok_pairs:
            yt2 = [t for t, _ in ok_pairs]
            yp2 = [p for _, p in ok_pairs]
            macro_f1 = f1_score(
                yt2, yp2, average="macro", labels=list(LABEL_ORDER), zero_division=0
            )
            per_f1_t = tuple(
                f1_score(
                    yt2,
                    yp2,
                    average=None,
                    labels=list(LABEL_ORDER),
                    zero_division=0,
                )
            )
        else:
            macro_f1 = 0.0
            per_f1_t = (0.0, 0.0, 0.0)
            yt2, yp2 = [], []

        print(f"accuracy (invalid=wrong): {acc:.4f}")
        print(f"macro F1 (valid preds only, n={len(ok_pairs)}): {macro_f1:.4f}")
        for lab, f1v in zip(LABEL_ORDER, per_f1_t):
            print(f"  F1 {lab}: {f1v:.4f}")

        summary_lines.append(
            f"{model_name},{matched},{m},{n_invalid},{acc:.6f},{macro_f1:.6f}"
        )
        cmp_names.append(model_name)
        acc_list.append(acc)
        f1_list.append(macro_f1)
        per_f1_rows.append(per_f1_t)

        note_short = f"{model_name} | invalid excl from cm: {n_invalid}"
        if ok_pairs:
            viz.confusion_heatmap_counts(
                yt2,
                yp2,
                f"counts — {note_short}",
                out / f"{model_name}_cm_counts.png",
            )
            viz.confusion_heatmap_row_norm(
                yt2,
                yp2,
                note_short,
                out / f"{model_name}_cm_row_norm.png",
            )
            grid_panels.append((model_name, yt2, yp2))
        else:
            print("no valid preds for confusion matrix")

    sum_path = out / "summary.csv"
    sum_path.write_text(
        "model,matched,n_missing,n_invalid,accuracy,macro_f1\n"
        + "\n".join(summary_lines)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {sum_path}")

    if grid_panels:
        viz.grid_row_norm_confusions(
            grid_panels,
            out / "all_models_cm_row_norm_grid.png",
        )
    if cmp_names:
        viz.metrics_grouped_bars(cmp_names, acc_list, f1_list, out / "metrics_comparison.png")
        viz.per_class_f1_grouped(cmp_names, per_f1_rows, out / "per_class_f1_by_model.png")
    if subset_mat:
        viz.subset_accuracy_heatmap(
            list(MODEL_FILES.keys()),
            np.vstack(subset_mat),
            out / "subset_accuracy_heatmap.png",
        )

    print("figures in", out)


if __name__ == "__main__":
    main()
