"""Command-line entry point for the RQ1 analysis."""

from __future__ import annotations

import sys

from analysis_utils import (
    AnalysisError,
    Path,
    SEED,
    SPEC_SCORE_TARGETS,
    argparse,
    csv,
    heuristic_spec_scores,
    sys,
    tempfile,
    write_rq2_dataset,
)
from extract_analysis import extract_dataset, run_heuristic_spec_scores, run_spec_scores
from modeling import model_and_report, split_repositories
from results import create_target_inventory, write_model_results


def self_test() -> None:
    import tempfile

    import pandas as pd

    empty_scores = heuristic_spec_scores("")
    if any(empty_scores[target] != 0 for target in SPEC_SCORE_TARGETS):
        raise AnalysisError("Empty specifications must receive zero rubric scores.")

    frame = pd.DataFrame(
        {
            "repo_name": [f"repo-{index // 2}" for index in range(20)],
        }
    )
    train_mask, test_mask = split_repositories(frame, SEED)
    if set(frame.loc[train_mask, "repo_name"]) & set(
        frame.loc[test_mask, "repo_name"]
    ):
        raise AnalysisError("Repository groups overlap in the modular holdout split.")
    if not train_mask.any() or not test_mask.any():
        raise AnalysisError("The modular holdout split must produce train and test rows.")

    with tempfile.TemporaryDirectory(prefix="specmine-modular-test-") as temporary:
        import analysis_utils

        old_output_dir = analysis_utils.OUTPUT_DIR
        try:
            analysis_utils.OUTPUT_DIR = Path(temporary)
            source_path = Path(temporary) / "modeling_dataset.csv"
            source_path.write_text(
                "file_url_sha16,repo_name\n"
                "spec-1,repo-1\n"
                "spec-2,repo-2\n",
                encoding="utf-8",
            )
            cache_path = Path(temporary) / "spec_scores_heuristic.csv"
            cache_path.write_text(
                "file_url_sha16,target_spec_completeness,target_spec_quality\n"
                "spec-1,4,3\n"
                "spec-2,2,1\n",
                encoding="utf-8",
            )
            subset_path = write_rq2_dataset(source_path, required=True)
            with subset_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 2 or rows[1]["target_spec_quality"] != "1":
                raise AnalysisError("RQ2 subset generation failed its modular smoke test.")
        finally:
            analysis_utils.OUTPUT_DIR = old_output_dir

    with tempfile.TemporaryDirectory(prefix="specmine-results-test-") as temporary:
        import results

        old_output_dir, old_image_dir = results.OUTPUT_DIR, results.IMAGE_DIR
        try:
            results.OUTPUT_DIR = Path(temporary)
            results.IMAGE_DIR = Path(temporary) / "images"
            write_model_results(
                frame=pd.DataFrame(
                    {"file_url_sha16": ["spec-1"], "repo_name": ["repo-1"]}
                ),
                target_names=["target_success"],
                comparison_rows=[],
                screen_reasons_by_target={},
                best_by_target={},
                skipped_by_target={"target_success": {"reason": "smoke test"}},
                manifest=[
                    {
                        "name": "example",
                        "source": "spec_files",
                        "selected": "False",
                        "reason": "Smoke-test feature",
                    }
                ],
                train_mask=pd.Series([True]),
                success_min_groups=2,
            )
            expected_results = {
                "model_comparison.csv",
                "target_model_results.csv",
                "feature_results.csv",
                "individual_vs_success_performance.csv",
            }
            if not expected_results.issubset(
                {path.name for path in Path(temporary).iterdir()}
            ):
                raise AnalysisError("The modular results writer did not create its tables.")
        finally:
            results.OUTPUT_DIR, results.IMAGE_DIR = old_output_dir, old_image_dir

    print(
        "Modular smoke tests passed (rubric utility, grouped split, RQ2 subset, "
        "and results tables)."
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "extract",
            "spec-scores",
            "spec-scores-heuristic",
            "model",
            "all",
            "self-test",
        ],
    )
    parser.add_argument(
        "--run-spec-scores",
        action="store_true",
        help="Score full spec Markdown using the local LLM during all.",
    )
    parser.add_argument("--llm-model", help="Model name for a local OpenAI-compatible endpoint.")
    parser.add_argument(
        "--llm-endpoint",
        help="Loopback-only endpoint; defaults to LLM_BASE_URL or local Ollama.",
    )
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument(
        "--spec-score-limit",
        type=int,
        default=0,
        help="Limit new spec-score requests; 0 means no limit.",
    )
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-category-count", type=int, default=20)
    parser.add_argument("--max-missing-fraction", type=float, default=0.8)
    parser.add_argument("--max-correlation", type=float, default=0.98)
    parser.add_argument("--success-min-groups", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        if args.phase == "extract":
            extract_dataset()
        elif args.phase == "spec-scores":
            run_spec_scores(args)
        elif args.phase == "spec-scores-heuristic":
            run_heuristic_spec_scores(args)
        elif args.phase == "model":
            create_target_inventory()
            model_and_report(args)
        elif args.phase == "all":
            extract_dataset()
            if args.run_spec_scores:
                run_spec_scores(args)
            create_target_inventory()
            model_and_report(args)
        elif args.phase == "self-test":
            self_test()
        return 0
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
