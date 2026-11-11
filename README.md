# Lexical vs. Semantic Approaches for Detecting Code-Understandability Comments in Software Reviews

A comparative NLP and machine learning project investigating whether code-review comments that concern **code understandability** can be identified automatically. The study compares lexical **word- and character-level TF-IDF** features against pretrained **Sentence-Transformer (`all-mpnet-base-v2`) embeddings**, using **Logistic Regression** for binary classification in both pipelines.

Developed for **CSI 5187 — Applications of NLP and ML**, University of Ottawa.

## Research questions

1. **RQ1:** Can NLP and machine learning models automatically identify code-review comments related to code-understandability improvements?
2. **RQ2:** Do semantic representations provide better classification performance than traditional lexical representations for this task?

## Dataset

The project uses the publicly available **CODE-UP (Code Understandability Improvement)** dataset, which contains manually annotated real-world code-review comments from 363 software projects.

- **Original dataset:** 2,401 comments (1,012 `Yes`, 1,333 `No`, 56 `Discard`).
- **Binary classification dataset:** 2,345 comments after excluding `Discard` entries.
- **Input:** `bodyText` (text of the code-review comment).
- **Target:** `label` (`0` = No, `1` = Yes).
- **Metadata retained:** `id` and `project` for traceability and grouped evaluation; these are not text-classification features.

Original dataset: [CODE-UP repository](https://github.com/codeupcrc/codeupcrc.github.io) | [Source TSV](https://github.com/codeupcrc/codeupcrc.github.io/blob/main/csv/dataset.tsv)

### Dataset partitions

Two train/validation/test strategies are used (approximately **70% / 15% / 15%**):

- **Stratified split:** Preserves the binary label distribution across partitions.
- **Project-grouped split:** Keeps projects disjoint between training, validation, and testing to assess performance on unseen projects.

| Partition | Stratified | Project-grouped |
|---|---:|---:|
| Train | 1,641 | 1,639 |
| Validation | 352 | 352 |
| Test | 352 | 354 |

## Methods

**Lexical pipeline**

- Word-level TF-IDF and character-level TF-IDF combined with scikit-learn `FeatureUnion`.
- Validation-based selection of n-gram settings and Logistic Regression regularization (`C`).
- `class_weight="balanced"`.
- An automatically selected decision threshold based on validation macro-F1.

**Semantic pipeline**

- Frozen pretrained `sentence-transformers/all-mpnet-base-v2` model, producing 768-dimensional sentence embeddings.
- Logistic Regression with `class_weight="balanced"`.
- Validation-based selection of regularization and optional `StandardScaler`; the final selected configurations do not use additional standardization.
- An automatically selected decision threshold based on validation macro-F1.

Both pipelines use the same dataset partitions within each evaluation strategy. **Macro-F1** is the primary model-selection and comparison metric; accuracy, macro-precision, macro-recall, class-specific metrics, and confusion matrices are also reported.

## Final experimental results

The results below are from the **updated models with validation-selected decision thresholds**, not the original baseline runs.

| Split | Model | Test accuracy | Test macro-F1 | Decision threshold |
|---|---|---:|---:|---:|
| Stratified | Word + character TF-IDF + LR | 0.7301 | 0.7180 | 0.5465 |
| Stratified | MPNet embeddings + LR | 0.7244 | 0.7214 | 0.4829 |
| Project-grouped | Word + character TF-IDF + LR | **0.7910** | **0.7839** | 0.5002 |
| Project-grouped | MPNet embeddings + LR | 0.7373 | 0.7353 | 0.4665 |

**Interpretation:** The models achieve similar macro-F1 under stratified evaluation. The lexical pipeline has the higher test macro-F1 on the project-grouped split, while MPNet achieves higher recall for the `Yes` class in both evaluation settings. Differences across split strategies are not directly attributable to the splitting method alone because the test sets contain different comments.

**Evaluation caveat:** These updated experiments followed inspection of earlier test results; treat them as exploratory comparisons rather than a newly untouched confirmatory test.

## Error analysis

The project exports false-positive/false-negative summaries, paired model disagreements, cases where only one model is correct, and cases where both models fail. A reviewed qualitative sample of **60 comments** includes 10 examples from each of three error categories for each splitting strategy.

The qualitative annotations consider linguistic characteristics including code-specific terminology, implicit and explicit understandability, context dependence, ambiguity, and documentation-related concerns. Categories are **non-exclusive** and counts describe the purposively selected sample, not the entire dataset.

## Repository structure

```text
.
├── data/
│   ├── raw/
│   │   └── dataset.tsv
│   └── processed/
│       ├── dataset_cleaned.csv
│       ├── stratified/
│       │   ├── train.csv
│       │   ├── validation.csv
│       │   └── test.csv
│       └── project_grouped/
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
├── notebooks/
│   ├── dataset_exploration.ipynb
│   └── results_analysis.ipynb
├── results/
│   ├── error_analysis/
│   │   ├── stratified/
│   │   ├── project_grouped/
│   │   ├── qualitative_annotation.csv
│   │   └── qualitative_annotation_reviewed.csv
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── predictions/
├── src/
│   ├── data_preprocessing.py
│   ├── dat_splitting_stratified.py
│   ├── data_splitting_project_grouped.py
│   ├── tfidf_model.py
│   ├── semantic_model.py
│   ├── evaluation.py
│   └── error_analysis.py
├── .gitignore
└── README.md
```

The `results/models/` folder contains saved scikit-learn model pipelines / classifiers (`.joblib`); the pretrained MPNet encoder itself is loaded separately by `sentence-transformers`.

## Setup

Use Python in a virtual environment. Required libraries include `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `torch`, `transformers`, `matplotlib`, `joblib`, and Jupyter/IPython for notebook execution.

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn sentence-transformers torch transformers matplotlib joblib jupyter ipython
```

For CUDA-enabled PyTorch installations, consult the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a wheel compatible with your system. Package versions are not pinned here; for strict reproducibility, record the versions used in the original experiment.

## Reproduce the workflow

Run these commands **from the repository root**, in the following order:

```powershell
python src/data_preprocessing.py
python src/dat_splitting_stratified.py
python src/data_splitting_project_grouped.py
python src/tfidf_model.py
python src/semantic_model.py
python src/evaluation.py
python src/error_analysis.py
```

The first three commands recreate processed data and partitions. Model training writes checkpoints and validation configurations under `results/`; evaluation loads these models and the selected thresholds to generate test metrics, predictions, and confusion matrices. Error analysis uses the saved test predictions.

Open `notebooks/dataset_exploration.ipynb` for exploratory data analysis and `notebooks/results_analysis.ipynb` for aggregate results and qualitative annotation workflows. **Do not rerun cells that regenerate the blank annotation worksheet if you intend to preserve the reviewed annotations.**

## Limitations

- Comments are classified using `bodyText` alone, without the surrounding code diff or full review discussion.
- A small dataset and a single set of partitions limit precision of performance estimates.
- Some identical comment texts may appear in multiple partitions, even when projects are group-disjoint.
- The qualitative sample was purposefully selected and does not estimate feature prevalence in all code-review comments.
- Threshold and model changes made after earlier test-set inspection are exploratory.

## Attribution

The dataset comes from the CODE-UP research project:

> D. Oliveira, R. Santos, B. de Oliveira, M. Monperrus, F. Castor, and F. Madeiral, “Understanding Code Understandability Improvements in Code Reviews,” *IEEE Transactions on Software Engineering*, 51(1), 2025 (published online in 2024). DOI: [10.1109/TSE.2024.3453783](https://doi.org/10.1109/TSE.2024.3453783).

This repository is a separate educational classification study and is **not** the original CODE-UP project. Consult the dataset's source repository for attribution and reuse terms before redistributing the data publicly.
