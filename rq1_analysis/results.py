"""Evaluation tables, comparisons, and plotting for RQ1."""

from __future__ import annotations

from analysis_utils import (
    ADDITIONAL_TARGETS,
    ALL_TARGETS,
    Any,
    FeatureModel,
    IMAGE_DIR,
    OUTPUT_DIR,
    SPEC_SCORE_TARGETS,
    TARGETS,
    TARGET_GROUPS,
    json,
    manifest_selected,
    math,
    normalized_target,
    re,
)


def write_model_results(
    frame: Any,
    target_names: list[str],
    comparison_rows: list[dict[str, Any]],
    screen_reasons_by_target: dict[str, dict[str, str]],
    best_by_target: dict[str, dict[str, Any]],
    skipped_by_target: dict[str, dict[str, str]],
    manifest: list[dict[str, str]],
    train_mask: Any,
    success_min_groups: int,
) -> None:
    import pandas as pd

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    prediction_frames: list[Any] = []
    target_result_rows: list[dict[str, Any]] = []
    selected_by_target: dict[str, set[str]] = {}
    effects_by_target: dict[str, dict[str, dict[str, Any]]] = {}
    for target in target_names:
        best = best_by_target.get(target)
        if not best:
            target_result_rows.append(
                {
                    "target": target,
                    "target_label": ALL_TARGETS.get(target, {}).get("label", "Composite success metric"),
                    "model_type": "",
                    "status": "skipped",
                    "reason": skipped_by_target.get(target, {}).get("reason", "no model fit"),
                }
            )
            continue
        selected = set(best["selected_features"])
        selected_by_target[target] = selected
        effects_by_target[target] = best["effects"]
        test_indices = best["test_indices"]
        actual = best["actual"]
        predicted = best["predicted"]
        target_prediction = pd.DataFrame(
            {
                "row_index": test_indices,
                "file_url_sha16": frame.loc[test_indices, "file_url_sha16"].to_numpy(),
                "repo_name": frame.loc[test_indices, "repo_name"].to_numpy(),
                "target": target,
                "actual": actual,
                "predicted": predicted,
                "residual": actual - predicted,
                "model_type": best["model_type"],
                "variant": best["variant"],
            }
        )
        prediction_frames.append(target_prediction)
        target_result_rows.append(
            {
                "target": target,
                "target_label": ALL_TARGETS.get(target, {}).get("label", "Composite success metric"),
                "target_group": ALL_TARGETS.get(target, {}).get("group", "success"),
                "status": "ok",
                "model_type": best["model_type"],
                "variant": best["variant"],
                "train_rows": best["train_observed_count"],
                "test_rows": best["test_observed_count"],
                "model_p_value": best["model_p_value"],
                "aic": best["aic"],
                "train_r2": best["train_r2"],
                "pseudo_r2": best["pseudo_r2"],
                "test_rmse": best["rmse"],
                "test_log_rmse": best.get("log_rmse", math.nan),
                "selection_metric": (
                    "log1p_rmse"
                    if ALL_TARGETS.get(target, {}).get("kind")
                    in {"positive_continuous", "count"}
                    else "rmse"
                ),
                "selection_rmse": best["selection_rmse"],
                "test_nrmse": best["test_nrmse"],
                "test_r2": best["test_r2"],
                "selected_input_features": ";".join(best["selected_features"]),
                "stage_effects": json.dumps(best.get("stage_effects", {}), sort_keys=True),
                "strongest_predictors": json.dumps(
                    sorted(
                        [
                            {
                                "feature": name,
                                **info,
                            }
                            for name, info in best["effects"].items()
                        ],
                        key=lambda item: abs(item["standardized_coefficient"]),
                        reverse=True,
                    )[:10]
                ),
            }
        )
        create_target_plots(
            target, actual, predicted, best
        )

    predictions_path = OUTPUT_DIR / "predictions.csv"
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    pd.DataFrame(target_result_rows).to_csv(
        OUTPUT_DIR / "target_model_results.csv", index=False
    )
    create_feature_results(
        manifest,
        target_names,
        selected_by_target,
        effects_by_target,
        screen_reasons_by_target,
    )
    create_individual_vs_success(target_result_rows)
    create_success_comparison(
        frame, best_by_target, train_mask, success_min_groups
    )
    print(f"Wrote regression results to {OUTPUT_DIR}")
    print(f"Best-model summary: {OUTPUT_DIR / 'target_model_results.csv'}")
    print(f"Model comparisons: {comparison_path}")

def make_short_description(row: dict[str, str]) -> str:
    text = re.sub(r"[^A-Za-z0-9' -]", "", row.get("reason", "")).strip()
    words = text.split()
    if len(words) < 3:
        words = row["name"].replace("_", " ").split() + ["feature measure"]
    return " ".join(words[:10])

def create_feature_results(
    manifest: list[dict[str, str]],
    target_names: list[str],
    selected_by_target: dict[str, set[str]],
    effects_by_target: dict[str, dict[str, dict[str, Any]]],
    screen_reasons_by_target: dict[str, dict[str, str]],
) -> None:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for item in manifest:
        name = item["name"]
        per_target_selected = [
            target
            for target in target_names
            if name in selected_by_target.get(target, set())
        ]
        reasons = [
            reasons[name]
            for reasons in screen_reasons_by_target.values()
            if name in reasons
        ]
        if per_target_selected:
            exclusion = ""
        elif reasons:
            exclusion = reasons[0]
        elif not manifest_selected(item):
            exclusion = "manually disabled in input_candidates.csv"
        else:
            exclusion = "backward elimination: not significant in final target models"
        output: dict[str, Any] = {
            "input_feature": name,
            "source": item.get("source", ""),
            "calculation_formula": (
                "COUNT rows grouped by spec and relation"
                if item.get("source") == "spec_links"
                else f"{item.get('source', '')}.{name}"
            ),
            "short_description": make_short_description(item),
            "primary_exclusion_reason": exclusion,
            "selected_for_targets": ";".join(per_target_selected),
        }
        for target in target_names:
            effect = effects_by_target.get(target, {}).get(name)
            output[f"{target}_relationship_magnitude"] = (
                effect["standardized_coefficient"] if effect else ""
            )
            output[f"{target}_relationship_type"] = (
                effect["direction"] if effect else ""
            )
            output[f"{target}_exclusion_reason"] = (
                ""
                if effect
                else screen_reasons_by_target.get(target, {}).get(name, "")
            )
        rows.append(output)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "feature_results.csv", index=False)

def create_individual_vs_success(target_results: list[dict[str, Any]]) -> None:
    import pandas as pd

    by_target = {row["target"]: row for row in target_results}
    aggregate = by_target.get("target_success", {})
    rows = []
    comparison_targets = [
        *TARGETS,
        *ADDITIONAL_TARGETS,
        *(target for target in SPEC_SCORE_TARGETS if target in by_target),
    ]
    for target in comparison_targets:
        result = by_target.get(target, {})
        rows.append(
            {
                "individual_target": target,
                "included_in_success_composite": target in TARGETS,
                "individual_model_type": result.get("model_type", ""),
                "individual_test_rmse": result.get("test_rmse", ""),
                "individual_test_nrmse": result.get("test_nrmse", ""),
                "individual_test_r2": result.get("test_r2", ""),
                "success_model_type": aggregate.get("model_type", ""),
                "success_test_rmse": aggregate.get("test_rmse", ""),
                "success_test_nrmse": aggregate.get("test_nrmse", ""),
                "success_test_r2": aggregate.get("test_r2", ""),
                "note": "RMSE is on different target scales; compare normalized RMSE and R2.",
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "individual_vs_success_performance.csv", index=False)

def create_target_plots(
    target: str,
    actual: Any,
    predicted: Any,
    best: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    image_dir = IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", target)
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residuals = actual_array - predicted_array
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predicted_array, residuals, s=10, alpha=0.35)
    axis.axhline(0, color="red", linestyle="--", linewidth=1)
    axis.set(xlabel="Predicted", ylabel="Residual (actual - predicted)", title=f"{target}: residuals")
    fig.tight_layout()
    fig.savefig(image_dir / f"{safe_name}_residuals.png", dpi=140)
    plt.close(fig)

    effects = best["effects"]
    strongest = sorted(
        effects.items(),
        key=lambda item: abs(item[1]["standardized_coefficient"]),
        reverse=True,
    )[:20]
    if strongest:
        names = [item[0] for item in strongest][::-1]
        values = [item[1]["standardized_coefficient"] for item in strongest][::-1]
        fig, axis = plt.subplots(figsize=(9, max(4, 0.32 * len(names))))
        colors = ["#b33c2e" if value < 0 else "#276a8c" for value in values]
        axis.barh(names, values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(
            xlabel="Standardized coefficient (largest category-level effect)",
            title=f"{target}: selected input relationships",
        )
        fig.tight_layout()
        fig.savefig(image_dir / f"{safe_name}_coefficients.png", dpi=140)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual_array, predicted_array, s=12, alpha=0.35)
    low = float(min(np.nanmin(actual_array), np.nanmin(predicted_array)))
    high = float(max(np.nanmax(actual_array), np.nanmax(predicted_array)))
    axis.plot([low, high], [low, high], color="red", linestyle="--", linewidth=1)
    axis.set(xlabel="Actual", ylabel="Predicted", title=f"{target}: predicted vs actual")
    fig.tight_layout()
    fig.savefig(image_dir / f"{safe_name}_predicted_vs_actual.png", dpi=140)
    plt.close(fig)

    feature_model: FeatureModel = best["feature_model"]
    selected = best["selected_features"]
    encoded_names = [column for feature in selected for column in feature_model.groups[feature]]
    if encoded_names:
        selected_values = feature_model.test.loc[:, encoded_names]
        corr = selected_values.corr(method="pearson").fillna(0.0)
        fig, axis = plt.subplots(figsize=(max(6, len(encoded_names) * 0.35), max(5, len(encoded_names) * 0.3)))
        image = axis.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(encoded_names)), encoded_names, rotation=90, fontsize=7)
        axis.set_yticks(range(len(encoded_names)), encoded_names, fontsize=7)
        axis.set_title(f"{target}: input correlation heatmap")
        fig.colorbar(image, ax=axis, shrink=0.8)
        fig.tight_layout()
        fig.savefig(image_dir / f"{safe_name}_input_correlations.png", dpi=140)
        plt.close(fig)

def clear_generated_plots() -> None:
        suffixes = (
            "residuals",
            "coefficients",
            "predicted_vs_actual",
            "input_correlations",
        )
        for target in [*ALL_TARGETS, "target_success"]:
            for suffix in suffixes:
                (IMAGE_DIR / f"{target}_{suffix}.png").unlink(missing_ok=True)
        (IMAGE_DIR / "success_prediction_comparison.png").unlink(missing_ok=True)

def create_success_comparison(
    frame: Any,
    best_by_target: dict[str, dict[str, Any]],
    train_mask: Any,
    min_groups: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    comparison_csv = OUTPUT_DIR / "success_prediction_comparison.csv"
    comparison_csv.unlink(missing_ok=True)
    success_best = best_by_target.get("target_success")
    if not success_best:
        return
    success_predictions = pd.Series(
        success_best["predicted"], index=success_best["test_indices"]
    )
    test_index = success_predictions.index
    posthoc_values: dict[str, Any] = {}
    actual_values: dict[str, Any] = {}
    for target in TARGETS:
        best = best_by_target.get(target)
        if not best:
            continue
        prediction = pd.Series(best["predicted"], index=best["test_indices"])
        common = test_index.intersection(prediction.index)
        kind = TARGETS[target]["kind"]
        reference = pd.to_numeric(frame.loc[train_mask, target], errors="coerce").dropna()
        scaled_pred = normalized_target(
            target, prediction.loc[common].to_numpy(), reference.to_numpy(), kind
        )
        actual_target = pd.to_numeric(frame.loc[common, target], errors="coerce")
        scaled_actual = normalized_target(
            target, actual_target.to_numpy(), reference.to_numpy(), kind
        )
        scaled_actual[actual_target.isna().to_numpy()] = np.nan
        posthoc_values[target] = pd.Series(scaled_pred, index=common)
        actual_values[target] = pd.Series(scaled_actual, index=common)
    rows: list[dict[str, Any]] = []
    for index in test_index:
        group_estimates: dict[str, float] = {}
        for group, members in TARGET_GROUPS.items():
            predictions: list[float] = []
            actuals: list[float] = []
            for target in members:
                if target not in posthoc_values or index not in posthoc_values[target].index:
                    continue
                prediction = posthoc_values[target].loc[index]
                actual = actual_values[target].loc[index]
                if pd.notna(actual) and pd.notna(prediction):
                    predictions.append(float(prediction))
                    actuals.append(float(actual))
            if predictions and actuals:
                group_estimates[group] = float(np.mean(predictions))
        if len(group_estimates) < min_groups:
            continue
        rows.append(
            {
                "row_index": index,
                "file_url_sha16": frame.at[index, "file_url_sha16"],
                "repo_name": frame.at[index, "repo_name"],
                "actual_success": frame.at[index, "target_success"],
                "aggregate_model_prediction": success_predictions.loc[index],
                "posthoc_individual_model_aggregation": float(np.mean(list(group_estimates.values()))),
                "observed_group_count": frame.at[index, "success_group_count"],
            }
        )
    result = pd.DataFrame(rows).dropna(subset=["actual_success"])
    if result.empty:
        return
    result.to_csv(comparison_csv, index=False)
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        result["actual_success"],
        result["aggregate_model_prediction"],
        s=14,
        alpha=0.45,
        label="Direct success model",
        color="#276a8c",
    )
    axis.scatter(
        result["actual_success"],
        result["posthoc_individual_model_aggregation"],
        s=14,
        alpha=0.45,
        label="Aggregate of individual target models",
        color="#b33c2e",
    )
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Actual composite success",
        ylabel="Predicted composite success",
        title="Direct composite model vs post-hoc aggregation",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "success_prediction_comparison.png", dpi=140)
    plt.close(fig)

def create_target_inventory() -> None:
    import pandas as pd

    source_path = OUTPUT_DIR / "spec_score_source.json"
    score_source = (
        json.loads(source_path.read_text(encoding="utf-8"))
        if source_path.exists()
        else {}
    )
    definitions = []
    for target, metadata in ALL_TARGETS.items():
        group = metadata["group"]
        definitions.append(
            {
                "target": target,
                "target_group": group,
                "description": metadata["label"],
                "source": (
                    "pull_requests + pr_files"
                    if group in {"pr_merging", "pr_merge_time"}
                    else "openspec_artifact_files"
                    if group in {"all_openspec_tasks", "tasks_md"}
                    else "openspec_code_refs"
                    if group == "code_ref_task_completion"
                    else "specmine-zenodo/specs.jsonl/specs.jsonl + "
                    + score_source.get("method", "rubric scorer")
                ),
                "formula": (
                    "LLM or deterministic rubric score from 0 to 5"
                    if group == "spec_rubric_assessment"
                    else "Derived in extract.sql; see METHODOLOGY.md"
                ),
                "composite_weight": (
                    "excluded from target_success"
                    if target not in TARGETS
                    else "equal across four outcome groups"
                ),
                "included_in_target_success": target in TARGETS,
                "scoring_method": (
                    score_source.get("method", "unspecified")
                    if group == "spec_rubric_assessment"
                    else ""
                ),
            }
        )
    pd.DataFrame(definitions).to_csv(OUTPUT_DIR / "target_inventory.csv", index=False)
