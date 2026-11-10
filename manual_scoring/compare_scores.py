"""Compare the manual scores with the computer's (heuristic) scores: Table 2 in the paper.

Reads:
- manual_scoring/manual_scoring_complete.xlsx  (sheet "Scoring": the manual scores)
- manual_scoring/sample_order.csv              (the 40-spec sample)
- rq1_analysis/output/modeling_dataset_sub.csv (the computer's scores for every spec)

Only rows with both a manual Completeness and Quality score are compared.
Differences are computer minus manual, so a negative number means the computer scored lower.

Run from the project root:
    .venv/bin/python manual_scoring/compare_scores.py
"""
import sys

import pandas as pd

WORKBOOK = "manual_scoring/manual_scoring_complete.xlsx"
SAMPLE = "manual_scoring/sample_order.csv"
HEURISTIC = "rq1_analysis/output/modeling_dataset_sub.csv"
MEASURES = {"Completeness": "target_spec_completeness", "Quality": "target_spec_quality"}


def main():
    manual = pd.read_excel(WORKBOOK, sheet_name="Scoring", dtype={"Spec ID": str})
    manual = manual.dropna(subset=list(MEASURES))
    sample = pd.read_csv(SAMPLE, dtype={"file_url_sha16": str})
    heuristic = pd.read_csv(HEURISTIC, dtype={"file_url_sha16": str})

    # Checks: no duplicate IDs, every scored spec is in the sample and has a computer score.
    problems = []
    for name, ids in [("workbook", manual["Spec ID"]), ("sample", sample["file_url_sha16"]),
                      ("heuristic scores", heuristic["file_url_sha16"])]:
        dupes = ids[ids.duplicated()].unique()
        if len(dupes):
            problems.append(f"duplicate IDs in {name}: {', '.join(dupes)}")
    for name, ids in [("sample", sample["file_url_sha16"]), ("heuristic scores", heuristic["file_url_sha16"])]:
        missing = sorted(set(manual["Spec ID"]) - set(ids))
        if missing:
            problems.append(f"scored specs missing from {name}: {', '.join(missing)}")
    if problems:
        sys.exit("\n".join(problems))

    joined = manual.merge(heuristic, left_on="Spec ID", right_on="file_url_sha16", how="inner")
    n = len(joined)
    print(f"{n} scored specs; all found in the sample and in the heuristic scores; no duplicate IDs.\n")

    print(f"{'Measure':<13}{'Exact':>8}{'Within 1':>10}{'Lower':>7}{'Same':>6}{'Higher':>8}{'Mean signed':>13}{'Mean abs':>10}")
    for measure, column in MEASURES.items():
        diff = joined[column] - joined[measure]
        print(f"{measure:<13}{f'{(diff == 0).sum()}/{n}':>8}{f'{(diff.abs() <= 1).sum()}/{n}':>10}"
              f"{(diff < 0).sum():>7}{(diff == 0).sum():>6}{(diff > 0).sum():>8}"
              f"{diff.mean():>13.2f}{diff.abs().mean():>10.2f}")


if __name__ == "__main__":
    main()
