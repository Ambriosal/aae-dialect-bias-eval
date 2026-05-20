# AAE dialect sentiment evaluation (`aae-dialect-bias-eval`)

A Python evaluation pipeline for benchmarking **LLM sentiment classification** on **African American English (AAE)** tweet text. It scores models against a fixed **1,000-example human gold set**, aligned to large model-result CSVs (~17k rows each) by **normalized AAE text** (no stable row ID in the exports).

This repository supports a **follow-up study** on dialect-related behavior in LLMs, building on work that studied dialectic preference bias and the **Dialectic Group Invariance (DGI)** framing.

---

## Background

Hassan et al. (2025) show that LLMs can assign **different sentiment** to **semantically aligned** content depending on dialect (e.g., AAE vs. SAE), and introduce metrics such as DGI to quantify that bias across models.

**This repo’s job** is narrower and practical: given **human gold labels** for 1,000 AAE items and **model sentiment predictions** on a much larger pool, it **joins on normalized text**, computes **accuracy / macro-F1 / per-class F1**, **subset breakdowns**, and **confusion-matrix-style figures** for comparing models on the same gold slice.

---

## Gold dataset (1,000 items)

| Block | Count | Description |
|--------|--------|-------------|
| Common | **100** | Triple-annotated; label is **majority vote**. If all three annotators disagree, a **tie-break policy** applies (default: annotator 1 — see `scripts/tie_policies.py`). |
| Unique | **300 × 3** | **300** items per annotator, **single** annotator label each. |

**Order in the pipeline:** common 100 (sorted by `item_index`), then annotator 1’s 300, then 2’s, then 3’s → **gold_id 0–999**. The loader enforces **exactly 1,000 rows** and **no duplicate normalized text keys**.

### Annotation context (research)

During collection, items could be rated along multiple dimensions (semantic alignment, naturalness, toxicity, etc.). **This evaluation script uses only the overall AAE sentiment** (`positive` / `neutral` / `negative`) as the classification target.

### Gold inputs (this repository)

Placed under `annotations/`:

- `common_100_majority_vote.csv` — needs `aae_text`, `sentiment_a1`, `sentiment_a2`, `sentiment_a3`, `item_index`
- `annotator_1_annotations_300_unique.csv` (and analogous files for annotators 2–3) — needs `aae_text`, `sentiment`, `item_index`

Per-annotator exports for the common 100 may also exist under `annotations/common/` for provenance; the **merged** common file above is what the loader uses.

---

## Model outputs

Model CSVs live under `models/`. Each file must include:

- `aae_text`
- `sentiment` — values mapped to `negative` / `neutral` / `positive`; empty or unmapped tokens (e.g. `ERROR`) are treated as invalid (see `scripts/labels.py`).

**Models configured in code** (`scripts/run_eval.py` → `MODEL_FILES`):

| Logical name | File |
|--------------|------|
| `gemma3_vllm` | `gemma3_vllm_aae_results.csv` |
| `mistral31_vllm` | `mistral31_vllm_aae_results.csv` |
| `deepseekr1_vllm` | `deepseekr1_vllm_aae_results.csv` |
| `phi3_medium` | `phi3_medium_aae_results.csv` |
| `llama31_70b_vllm` | `llama31_70b_vllm_aae_results.csv` |
| `phi4_vllm` | `phi4_vllm_aae_results.csv` |

To add or rename a model, edit **`MODEL_FILES`** and place the CSV in **`models/`**.

**Matching:** `scripts/text_norm.py` defines `norm_text()` (strip, NFC Unicode, collapse whitespace). Duplicate normalized lines in a model file: **first row wins**; conflicting duplicates are reported at runtime.

---

## What the pipeline does

1. **Load gold** — `scripts/gold_dataset.py` builds the ordered list of 1,000 examples.
2. **Load each model CSV** — build a map `norm_text(aae_text) → sentiment`.
3. **Evaluate** — coverage (missing keys), accuracy (invalid prediction counts as wrong), **macro-F1** on **valid** predictions only, per-class F1, per-subset accuracy.
4. **Visualize** — confusion heatmaps, label-mix bars, grouped metric plots, subset heatmap (`scripts/viz.py`).
5. **Write results** under `outputs/` (see below).

Main entrypoint: **`scripts/run_eval.py`**.

---

## Key metrics (as implemented)

| Metric | Definition in this repo |
|--------|-------------------------|
| **Accuracy** | Fraction of gold rows where prediction equals gold (**invalid** model token → wrong). |
| **Macro-F1** | sklearn `f1_score(..., average='macro')` over `negative`, `neutral`, `positive`, computed on **matched rows with valid** predictions only. |
| **Per-class F1** | One F1 per class (same label order as above). |
| **Confusion-style plots** | 3×3 gold vs. predicted (invalid excluded from matrix plots per implementation). |
| **Subset accuracy** | Accuracy within `common_100`, `annotator_1_unique`, … |

**Inter-annotator agreement (e.g. Fleiss’ κ)** and **pairwise significance tests** are **not** produced by `run_eval.py`. If you revive that analysis, plan a separate script or a small `archive/` workflow and document dependencies (e.g. `statsmodels`) there.

---

## Outputs

After a successful run, **`outputs/`** includes:

| Artifact | Role |
|----------|------|
| `summary.csv` | Per model: matched count, missing, invalid, accuracy, macro-F1 |
| `per_class_f1.csv` | Per model per-class F1 |
| `subset_accuracy_matrix.csv` | Models × subsets |
| `*_cm_counts.png`, `*_cm_row_norm.png` | Confusion figures per model |
| `*_label_mix.png` | Gold vs prediction label mix |
| `metrics_comparison.png`, `per_class_f1_by_model.png` | Cross-model summaries |
| `all_models_cm_row_norm_grid.png`, `subset_accuracy_heatmap.png` | Paper-style composites |

Treat the **CSVs as the numeric source of truth** for writing results; figure files may be regenerated whenever you rerun the pipeline.

If any model has **no matching** normalized keys for some gold rows, the script writes **`{model_name}_missing_gold_rows.csv`** for debugging.

---

## Repository layout (actual)

```text
aae-dialect-bias-eval/
├── annotations/           # Gold CSVs (common 100 + 300×3 unique)
├── models/                # Model sentiment CSVs (~17k rows each)
├── scripts/
│   ├── run_eval.py        # Main entry: load gold + models → metrics + figures
│   ├── gold_dataset.py    # Build 1k gold list; paths relative to annotations/
│   ├── labels.py          # Canonical labels + parsing for gold vs preds
│   ├── text_norm.py       # Normalization key for joins
│   ├── tie_policies.py    # 3-way tie-break on common 100
│   ├── viz.py             # Matplotlib figures
│   └── requirements.txt
├── outputs/               # Generated CSVs + figures (git may ignore or commit)
├── golden/                # Extra tables / HTML (not required by run_eval.py)
└── HTML/                  # Standalone dashboards / sketches (not required by run_eval.py)
```

---

## Requirements

- **Python 3.10+**
- Dependencies: **`matplotlib`**, **`scikit-learn`** (see `scripts/requirements.txt`; NumPy is pulled in as a dependency of scikit-learn).

This pipeline uses the **stdlib `csv` module** — not pandas.

---

## How to run

From the **repository root**:

```bash
pip install -r scripts/requirements.txt
python scripts/run_eval.py
```

Paths are resolved from the repo root: gold from `annotations/`, models from `models/`, artifacts to `outputs/`.

---

## Reproducibility & path changes

- If you **move** `annotations/` or `models/`, update **`scripts/run_eval.py`** (`MODEL_FILES` / parent paths) and **`scripts/gold_dataset.py`** (`COMMON_CSV`, `ANNOTATOR_FILES`, or the directory passed to `load_gold_rows`).
- Tie-break behavior is **pluggable** via `scripts/tie_policies.py`.

---

## Research references

**Original symposium paper:**  
Hassan, M. F., Khattak, F. K., & Seyyed-Kalantari, L. (2025). *Dialectic Preference Bias in Large Language Models.* Proceedings of the AAAI Symposium Series, 5(1), 365–369. https://doi.org/10.1609/aaaiss.v5i1.35613  

**Affiliation (paper):** York University, Vector Institute, Monark Health.

**Note:** The model set in **`run_eval.py`** reflects the **follow-up** experiments (e.g. additional open-weight / vLLM-served models), not necessarily the exact three models named in the 2025 symposium abstract.

