# SpecMine Analysis

This project runs three analysis stages in order:

1. Manually score a small sample of specifications.
2. Extract the SpecMine data, calculate corpus-wide heuristic completeness
   and quality scores, and fit regression models in RQ1.
3. Use commit history and those scores to compare specification authorship
   groups in RQ2.

Run commands below from the root of this repository (the directory containing
`manual_scoring`, `rq1_analysis`, and `rq2_analysis`).

## 1. Setup

### Project files and data

The SpecMine dataset is not included in this repository. Download it from
[Zenodo record 10.5281/zenodo.22102779](https://doi.org/10.5281/zenodo.22102779)
and extract it into a `specmine-zenodo` folder beside the repository, keeping
the dataset's directory structure intact. By default, RQ1 looks for
`specmine-zenodo/specs.jsonl/specs.jsonl` beside the repository. To use a
different location, set `SPECMINE_DATASET_ROOT` to the extracted dataset
directory before running RQ1; the optional sample utility reads Parquet files
under its `data/` directory. RQ2 downloads public commit-history data from
Hugging Face, so it needs internet access.

For example, in PowerShell:

```powershell
$env:SPECMINE_DATASET_ROOT = "D:\datasets\specmine-zenodo"
```

The expected workspace layout is:

```text
workspace/
  specmine-zenodo/          # downloaded and extracted dataset
  <analysis-repository>/    # this repository
    manual_scoring/
    rq1_analysis/
    rq2_analysis/
    .venv/
```

### Python environment

Create a virtual environment and install the dependencies used across all
three stages:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt duckdb datasets huggingface_hub pyarrow ipykernel
```

Use the `.venv` interpreter for RQ1. For RQ2, install `ipykernel` in that
environment and select it as the kernel in your Jupyter environment.

### MySQL database

Create a MySQL database using the downloaded dataset's `schema.sql` and load
its table data using the SQL/data scripts included with the dataset. RQ1
connects to that local database using `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`,
`MYSQL_PASSWORD`, and `MYSQL_DATABASE`. Set them in the process environment or
in a `.env` file at the workspace root (the parent directory containing both
the dataset and this repository). The default database name is `ssw625`.

The dataset also provides a MySQL dump that can be restored with a local MySQL
client. **The dump contains `DROP` and `CREATE` statements**; review the target
database and back it up before restoring.

## 2. Manual sample scoring

This stage manually evaluates a sample of specifications using the provided
completeness and quality rubric, then compares human assessments with the
computer-generated scores. The supplied sample order lists 40 specs.
`manual_scoring_complete.xlsx` holds the manual scores for the 30 that were
reviewed; this is the data behind Table 2 in the paper. For the
full process, rubric definitions, scoring rules, and comparison measures, see
[manual_scoring/README.md](./manual_scoring/README.md). RQ1 uses deterministic
heuristic measures for the full corpus.

To regenerate `sample_order.csv`, install `duckdb` as above and run the utility
from the `specmine-zenodo` directory, where its `data/` paths resolve:

```powershell
$analysisRepository = (Get-Location).Path
Set-Location ..\specmine-zenodo
& "$analysisRepository\.venv\Scripts\python.exe" "$analysisRepository\manual_scoring\draw_sample.py" "$analysisRepository\manual_scoring\sample_order.csv"
Set-Location $analysisRepository
```

Regeneration replaces the sample-order file; it is not needed when using the
provided sample and completed workbook.

To recompute Table 2 (manual vs. computer scores) and check for duplicate IDs
and missing specs:

```powershell
.\.venv\Scripts\python.exe .\manual_scoring\compare_scores.py
```

## 3. RQ1: extraction, heuristic scoring, and regression modeling

RQ1 needs the populated MySQL database, the expanded JSONL spec-text file, and
the Python dependencies above. Run extraction, heuristic scoring, and
regression modeling together:

```powershell
.\.venv\Scripts\python.exe .\rq1_analysis\run_analysis.py all
```

The `all` phase runs `extract`, `spec-scores-heuristic`, and `model` in order.
The individual phases can still be run separately when only one step needs to
be repeated.

`extract` writes the full `rq1_analysis/output/modeling_dataset.csv`.
`spec-scores-heuristic` reads the full Markdown corpus from
the `specs.jsonl/specs.jsonl` file under `SPECMINE_DATASET_ROOT` (or the default
`specmine-zenodo` sibling directory) and scores completeness and quality from
0 to 5 using deterministic text heuristics—including regular-expression
matching and counts of words, lists, and headings—to approximate the rubric's
criteria. It updates the full dataset and creates
`rq1_analysis/output/modeling_dataset_sub.csv`. The subset retains every row
and only the specification hash and the two heuristic scores.
`model` runs the RQ1 regression analysis and writes its tables, configuration,
and plots under `rq1_analysis/output/`.

The full extraction and score cache are large and need not be committed for
RQ2 to run. The compact subset is the RQ2 input and should be kept with the
project or committed if repository limits permit. If the extraction or
heuristic-scoring rules change, rerun these stages in order to refresh it.

For a quick verification that does not connect to MySQL or score the full
corpus:

```powershell
.\.venv\Scripts\python.exe .\rq1_analysis\run_analysis.py self-test
```

## 4. RQ2: authorship and score comparisons

Run `rq2_analysis/SpecMine RQ2.ipynb` in Jupyter Notebook, JupyterLab, or
another compatible notebook environment, using the project's `.venv` kernel.
The notebook expects its working directory to be the `rq2_analysis` folder;
this is an environment-independent requirement. In VS Code, set **Jupyter:
Notebook File Root** to `${fileDirname}` to meet it.

The notebook installs `datasets`, `huggingface_hub`, and `pyarrow` in its first
cell, then loads public commit history, reads
`../rq1_analysis/output/modeling_dataset_sub.csv`, assigns authorship groups,
compares the heuristic completeness and quality scores, and saves two PNG
plots in the working directory. An unauthenticated Hugging Face warning is
informational; a token is not required for this public dataset.

RQ2 requires the compact subset generated in the RQ1 scoring step. If it is
missing, run the three RQ1 commands above; the full extraction does not need to
be repeated if `modeling_dataset.csv` and the heuristic score cache are already
available.

## Main outputs

- `manual_scoring/manual_scoring_complete.xlsx` — completed manual sample.
- `rq1_analysis/output/modeling_dataset_sub.csv` — all specs and the two
  heuristic rubric scores used by RQ2.
- `rq1_analysis/output/` — RQ1 model tables, configuration, and figures.
- `rq2_analysis/rq2_completeness_by_authorship.png`
- `rq2_analysis/rq2_quality_by_authorship.png`