# SpecMine Regression Analysis

This folder contains a fixed-target regression analysis using the local
`ssw625` MySQL database and the specification-text JSONL file. Predictors are
the static candidates in `input_candidates.csv`; no LLM-derived input-feature
extraction is performed. Optional rubric scoring uses the full specification
text to produce two separate outcomes: completeness and quality. Those scores
are never predictors and are not part of the aggregate `target_success`.

## Files

- `run_analysis.py` — database extraction, optional rubric scoring, filtering,
  regression, reporting, and plot generation.
- `extract.sql` — one-row-per-spec outcomes and the selected database
  predictors.
- `input_candidates.csv` — static predictor manifest; `selected` is the
  manual inclusion switch.
- `METHODOLOGY.md` — target formulas, model choices, configuration, and
  interpretation notes.
- `output/` — generated dataset, score caches, regression tables, and images.

## Setup

Install dependencies into the workspace Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\specmine_analysis\requirements.txt
```

The runner reads `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`,
and `MYSQL_DATABASE` from the process environment or the workspace-root `.env`
file. The default database is `ssw625`. Database credentials are never written
to outputs.

## Run

From the workspace root:

```powershell
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py extract
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py model
```

To refresh extraction and run the models in one command:

```powershell
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py all
```

The separate target outcomes include PR and OpenSpec measures described in
`METHODOLOGY.md`. Only four designated measures contribute to
`target_success`; the restored count, binary, task-list, and rubric outcomes
are modeled independently and excluded from that composite.

## Specification rubric scores

The `spec-scores` phase reads the complete Markdown `content` field from
`specmine-zenodo/specs.jsonl/specs.jsonl`, scores completeness and quality from
0 to 5 using a local OpenAI-compatible LLM endpoint, and caches evidence
summaries in `output/spec_scores.csv`. Scores join the extracted data by
`file_url_sha16`. To use a local model:

```powershell
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py spec-scores --llm-model <local-model>
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py model
```

The endpoint defaults to loopback Ollama and must remain on the local machine.
Scoring may process up to 470,795 specifications; use `--spec-score-limit` to
score a small number of new records before a full run. Long specifications are
sent as sequential 8,000-character segments and combined in a final local
request. Results resume from the versioned cache.

Until a local model is available, deterministic regex heuristics can score the
corpus without network access:

```powershell
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py spec-scores-heuristic --spec-score-limit 100
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py spec-scores-heuristic
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py model
```

The heuristic scores are transparent proxies, not LLM judgments or validated
ground truth. When these scores are modeled, selected `spec_content_features`
and `file_content_bytes` inputs are excluded from those two models because
their meanings overlap directly with the heuristic scoring rules. Their
target-specific exclusion reasons appear in `feature_results.csv`.

The `all` command does not score rubric outcomes unless explicitly requested
with `--run-spec-scores --llm-model <local-model>`.

## Validation

```powershell
.\.venv\Scripts\python.exe .\specmine_analysis\run_analysis.py self-test
```

The synthetic-data check exercises composite-target isolation, restored
non-composite outcomes, heuristic input exclusions, model selection, reporting,
and plots without connecting to MySQL.

## Outputs

- `output/modeling_dataset.csv`
- `output/spec_scores.csv` and `output/spec_scores_heuristic.csv` when scored
- `output/target_model_results.csv`
- `output/target_inventory.csv`
- `output/feature_results.csv`
- `output/model_comparison.csv`
- `output/predictions.csv`
- `output/individual_vs_success_performance.csv`
- `output/success_prediction_comparison.csv`
- `output/analysis_config.json`
- `output/images/` with per-target residual, coefficient, predicted-vs-actual,
  input-correlation, and aggregate-comparison plots

Initial input filters, target-specific leakage exclusions, and backward
elimination decisions are reported in the result tables.
