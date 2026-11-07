"""Regression preparation, fitting, model selection, and orchestration."""

from __future__ import annotations

from analysis_utils import (
    ADDITIONAL_TARGETS,
    ALL_TARGETS,
    AnalysisError,
    Any,
    FeatureModel,
    HEURISTIC_TARGET_INPUT_EXCLUSIONS,
    IMAGE_DIR,
    OUTPUT_DIR,
    RIDGE_ALPHA,
    SEED,
    SPEC_SCORE_TARGETS,
    TARGETS,
    TARGET_GROUPS,
    argparse,
    json,
    manifest_selected,
    math,
    normalized_target,
    read_manifest,
    warnings,
    write_rq2_dataset,
)
from results import clear_generated_plots, write_model_results


def split_repositories(frame: Any, seed: int) -> tuple[Any, Any]:
    import numpy as np

    indices = np.arange(len(frame))
    groups = frame["repo_name"].fillna("__MISSING_REPO__").astype(str)
    group_values = groups.unique()
    if len(group_values) > 1:
        generator = np.random.default_rng(seed)
        shuffled = generator.permutation(group_values)
        test_group_count = min(max(1, round(len(group_values) * 0.2)), len(group_values) - 1)
        test_groups = set(shuffled[:test_group_count])
        test_mask = groups.isin(test_groups).to_numpy()
        train_mask = ~test_mask
    else:
        generator = np.random.default_rng(seed)
        shuffled_indices = generator.permutation(indices)
        test_count = max(1, round(len(frame) * 0.2))
        test_indices = set(shuffled_indices[:test_count])
        test_mask = np.asarray([index in test_indices for index in indices])
        train_mask = ~test_mask
    return train_mask, test_mask

def add_success_metric(frame: Any, train_mask: Any, min_groups: int):
    import numpy as np
    import pandas as pd

    transformed: dict[str, Any] = {}
    for target, metadata in TARGETS.items():
        numeric = pd.to_numeric(frame[target], errors="coerce")
        reference = numeric.loc[train_mask].dropna().to_numpy(dtype=float)
        values = numeric.to_numpy(dtype=float)
        transformed[target] = normalized_target(
            target, values, reference, metadata["kind"]
        )
        transformed[target][~np.isfinite(values)] = np.nan
    group_scores: dict[str, Any] = {}
    for group, members in TARGET_GROUPS.items():
        scores = np.column_stack([transformed[name] for name in members])
        valid = np.isfinite(scores)
        count = valid.sum(axis=1)
        values = np.divide(
            np.nansum(scores, axis=1),
            count,
            out=np.full(len(frame), np.nan),
            where=count > 0,
        )
        group_scores[group] = values
    grouped = np.column_stack(list(group_scores.values()))
    observed_groups = np.isfinite(grouped)
    count = observed_groups.sum(axis=1)
    success = np.divide(
        np.nansum(grouped, axis=1),
        count,
        out=np.full(len(frame), np.nan),
        where=count >= min_groups,
    )
    frame = frame.copy()
    frame["success_group_count"] = count
    frame["target_success"] = success
    return frame

def prepare_features(
    train_frame: Any,
    test_frame: Any,
    features: list[dict[str, str]],
    min_category_count: int,
    max_missing_fraction: float,
    max_correlation: float,
) -> FeatureModel:
    import pandas as pd

    usable: list[dict[str, str]] = []
    dropped: dict[str, str] = {}
    numeric_columns: list[str] = []
    numeric_train: dict[str, Any] = {}
    for row in features:
        name = row["name"]
        kind = row["type"].strip().lower()
        if name not in train_frame or name not in test_frame:
            dropped[name] = "not present in extracted data"
            continue
        series = train_frame[name]
        missing_fraction = float(series.isna().mean())
        if missing_fraction > max_missing_fraction:
            dropped[name] = f"initial filter: {missing_fraction:.0%} missing"
            continue
        if kind in {"numeric", "binary"}:
            values = pd.to_numeric(series, errors="coerce")
            observed = values.dropna()
            if observed.nunique() < 2:
                dropped[name] = "initial filter: constant or near-constant"
                continue
            usable.append(row)
            numeric_columns.append(name)
            numeric_train[name] = values
        elif kind == "categorical":
            values = series.fillna("__MISSING__").astype(str)
            if values.nunique() < 2:
                dropped[name] = "initial filter: constant or near-constant"
                continue
            if values.nunique() > 60:
                dropped[name] = "initial filter: high-cardinality category (>60 levels)"
                continue
            usable.append(row)
        else:
            dropped[name] = "not a numeric or categorical model input"

    correlated: set[str] = set()
    for position, left in enumerate(numeric_columns):
        if left in correlated:
            continue
        left_values = numeric_train[left]
        for right in numeric_columns[position + 1 :]:
            if right in correlated:
                continue
            pair = pd.concat([left_values, numeric_train[right]], axis=1).dropna()
            if len(pair) < 10:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
            if pd.notna(corr) and abs(float(corr)) >= max_correlation:
                correlated.add(right)
                dropped[right] = (
                    f"initial filter: collinear with {left} (|Spearman rho|={abs(corr):.3f})"
                )
    usable = [row for row in usable if row["name"] not in correlated]
    train_encoded: dict[str, Any] = {}
    test_encoded: dict[str, Any] = {}
    groups: dict[str, list[str]] = {}

    for row in usable:
        name = row["name"]
        kind = row["type"].strip().lower()
        if kind in {"numeric", "binary"}:
            train_values = pd.to_numeric(train_frame[name], errors="coerce")
            median = train_values.median()
            if pd.isna(median):
                dropped[name] = "initial filter: no numeric observations"
                continue
            train_values = train_values.fillna(median).astype(float)
            test_values = pd.to_numeric(test_frame[name], errors="coerce").fillna(median).astype(float)
            mean = float(train_values.mean())
            std = float(train_values.std(ddof=0))
            if not math.isfinite(std) or std == 0:
                dropped[name] = "initial filter: zero variance after imputation"
                continue
            train_encoded[name] = (train_values - mean) / std
            test_encoded[name] = (test_values - mean) / std
            groups[name] = [name]
        else:
            train_values = train_frame[name].fillna("__MISSING__").astype(str)
            test_values = test_frame[name].fillna("__MISSING__").astype(str)
            counts = train_values.value_counts()
            common = set(counts[counts >= min_category_count].index)
            if len(common) < 2:
                dropped[name] = "initial filter: fewer than two sufficiently common categories"
                continue
            train_values = train_values.where(train_values.isin(common), "__OTHER__")
            test_values = test_values.where(test_values.isin(common), "__OTHER__")
            levels = sorted(train_values.unique())
            reference = levels[0]
            test_values = test_values.where(test_values.isin(levels), reference)
            columns: list[str] = []
            for level in levels:
                if level == reference:
                    continue
                column = f"{name}__{level}"
                train_encoded[column] = (train_values == level).astype(float)
                test_encoded[column] = (test_values == level).astype(float)
                columns.append(column)
            if not columns:
                dropped[name] = "initial filter: fewer than two usable categories"
                continue
            for column in columns:
                train_values_encoded = train_encoded[column]
                mean = float(train_values_encoded.mean())
                std = float(train_values_encoded.std(ddof=0))
                if std == 0:
                    del train_encoded[column]
                    del test_encoded[column]
                    continue
                train_encoded[column] = (train_values_encoded - mean) / std
                test_encoded[column] = (test_encoded[column] - mean) / std
            columns = [column for column in columns if column in train_encoded]
            if not columns:
                dropped[name] = "initial filter: category dummies have zero variance"
                continue
            groups[name] = columns

    train_design = pd.DataFrame(train_encoded, index=train_frame.index, dtype=float)
    test_design = pd.DataFrame(test_encoded, index=test_frame.index, dtype=float)
    if not train_design.empty:
        correlations = train_design.corr().abs()
        owner = {
            column: feature
            for feature, columns in groups.items()
            for column in columns
        }
        retained_columns: list[str] = []
        for column in train_design.columns:
            duplicate_of = next(
                (
                    retained
                    for retained in retained_columns
                    if pd.notna(correlations.at[column, retained])
                    and correlations.at[column, retained] >= 0.999999
                ),
                None,
            )
            if duplicate_of is None:
                retained_columns.append(column)
                continue
            feature = owner[column]
            retained_feature = owner[duplicate_of]
            dropped.setdefault(
                feature,
                f"initial filter: encoded column is collinear with {retained_feature}",
            )
            groups[feature].remove(column)
        empty_groups = [feature for feature, columns in groups.items() if not columns]
        for feature in empty_groups:
            del groups[feature]
        train_design = train_design.loc[:, retained_columns]
        test_design = test_design.loc[:, retained_columns]
    return FeatureModel(train_design, test_design, groups, dropped)

def fit_regression(
    model_type: str,
    train_x: Any,
    train_y: Any,
    test_x: Any,
    train_successes: Any | None = None,
    train_trials: Any | None = None,
):
    import numpy as np
    import statsmodels.api as sm
    from scipy.stats import chi2
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning,
        PerfectSeparationWarning,
    )

    unique = np.unique(np.asarray(train_y, dtype=float))
    if model_type in {"logit", "binary_logit_ridge"} and len(unique) < 2:
        raise AnalysisError("logistic regression requires both target classes in training data")
    if model_type in {"fractional_logit", "fractional_logit_ridge"} and (
        np.nanmin(train_y) < 0 or np.nanmax(train_y) > 1
    ):
        raise AnalysisError("fractional-logit regression requires outcomes in [0, 1]")
    if model_type == "grouped_binomial_glm":
        if train_successes is None or train_trials is None:
            raise AnalysisError("grouped-binomial regression requires success and trial counts")
        successes = np.asarray(train_successes, dtype=float)
        trials = np.asarray(train_trials, dtype=float)
        if (
            not np.isfinite(successes).all()
            or not np.isfinite(trials).all()
            or np.any(trials <= 0)
            or np.any(successes < 0)
            or np.any(successes > trials)
        ):
            raise AnalysisError("grouped-binomial counts must be finite and satisfy 0 <= successes <= trials")
    if model_type in {"poisson_glm"} and np.nanmin(train_y) < 0:
        raise AnalysisError("Poisson regression requires nonnegative outcomes")
    if model_type == "gamma_glm" and np.nanmin(train_y) <= 0:
        raise AnalysisError("Gamma regression requires strictly positive outcomes")

    if model_type == "intercept_only":
        train_mean = float(np.mean(train_y))
        predicted = np.full(len(test_x), train_mean, dtype=float)
        return {
            "result": None,
            "predicted": predicted,
            "params": np.asarray([train_mean]),
            "pvalues": np.asarray([math.nan]),
            "model_p_value": math.nan,
            "train_r2": math.nan,
            "pseudo_r2": math.nan,
            "aic": math.nan,
            "design_train": None,
        }

    design_train = sm.add_constant(train_x, has_constant="add")
    design_test = sm.add_constant(test_x, has_constant="add")
    fit_y = np.asarray(train_y, dtype=float)
    if model_type == "lognormal_ols":
        fit_y = np.log1p(fit_y)
    elif model_type == "grouped_binomial_glm":
        fit_y = np.column_stack((successes, trials - successes))
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        if model_type in {"ols", "ols_lpm", "ordinal_score_ols"}:
            result = sm.OLS(train_y, design_train).fit(cov_type="HC3")
        elif model_type == "logit":
            result = sm.Logit(train_y, design_train).fit(
                disp=0, maxiter=150, cov_type="HC3"
            )
        elif model_type in {"fractional_logit_ridge", "binary_logit_ridge"}:
            penalty = np.full(design_train.shape[1], RIDGE_ALPHA, dtype=float)
            penalty[0] = 0.0
            result = sm.GLM(
                train_y, design_train, family=sm.families.Binomial()
            ).fit_regularized(
                alpha=penalty,
                L1_wt=0.0,
                maxiter=1000,
                cnvrg_tol=1e-8,
            )
        elif model_type in {"fractional_logit", "grouped_binomial_glm"}:
            result = sm.GLM(
                fit_y, design_train, family=sm.families.Binomial()
            ).fit(maxiter=100, cov_type="HC3")
        elif model_type == "lognormal_ols":
            result = sm.OLS(fit_y, design_train).fit(cov_type="HC3")
        elif model_type == "poisson_glm":
            result = sm.GLM(
                train_y, design_train, family=sm.families.Poisson()
            ).fit(maxiter=100, cov_type="HC3")
        elif model_type == "gamma_glm":
            result = sm.GLM(
                train_y,
                design_train,
                family=sm.families.Gamma(sm.families.links.Log()),
            ).fit(maxiter=100, cov_type="HC3")
        else:
            raise AnalysisError(f"Unknown regression type: {model_type}")
        linear_prediction = np.asarray(result.predict(design_test), dtype=float)
        if model_type == "lognormal_ols":
            smearing = float(np.mean(np.exp(np.asarray(result.resid, dtype=float))))
            predicted = np.maximum(np.exp(linear_prediction) * smearing - 1.0, 0.0)
        elif model_type == "ordinal_score_ols":
            predicted = np.clip(linear_prediction, 0.0, 5.0)
        else:
            predicted = linear_prediction
        params = np.asarray(result.params, dtype=float)
        pvalues = (
            np.full(len(params), np.nan)
            if model_type in {"fractional_logit_ridge", "binary_logit_ridge"}
            else np.asarray(result.pvalues, dtype=float)
        )
        aic = (
            math.nan
            if model_type in {"fractional_logit_ridge", "binary_logit_ridge"}
            else float(getattr(result, "aic", math.nan))
        )
    unstable_warnings = [
        warning
        for warning in fit_warnings
        if issubclass(
            warning.category,
            (RuntimeWarning, ConvergenceWarning, PerfectSeparationWarning),
        )
    ]
    if unstable_warnings:
        raise AnalysisError(
            f"{model_type} produced an unstable numerical warning: "
            f"{unstable_warnings[0].message}"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(params).all():
        raise AnalysisError(f"{model_type} produced non-finite coefficients or predictions.")
    if (
        model_type not in {"fractional_logit_ridge", "binary_logit_ridge"}
        and (not math.isfinite(aic) or not np.isfinite(pvalues).all())
    ):
        raise AnalysisError(f"{model_type} produced non-finite fit statistics.")
    if model_type in {
        "logit",
        "fractional_logit",
        "fractional_logit_ridge",
        "binary_logit_ridge",
        "grouped_binomial_glm",
        "ols_lpm",
    }:
        predicted = np.clip(predicted, 0.0, 1.0)
    elif model_type == "ols" and 0 <= np.nanmin(train_y) and np.nanmax(train_y) <= 1:
        predicted = np.clip(predicted, 0.0, 1.0)
    elif model_type in {"poisson_glm", "gamma_glm", "lognormal_ols"}:
        predicted = np.maximum(predicted, 0.0)
    model_p = math.nan
    r_squared = math.nan
    pseudo_r_squared = math.nan
    if model_type in {"ols", "ols_lpm", "lognormal_ols", "ordinal_score_ols"}:
        model_p = float(getattr(result, "f_pvalue", math.nan))
        r_squared = float(getattr(result, "rsquared", math.nan))
    else:
        null_deviance = float(getattr(result, "null_deviance", math.nan))
        deviance = float(getattr(result, "deviance", math.nan))
        df_model = float(getattr(result, "df_model", 0))
        if math.isfinite(null_deviance) and math.isfinite(deviance) and df_model > 0:
            model_p = float(chi2.sf(max(0.0, null_deviance - deviance), df_model))
            pseudo_r_squared = (
                1.0 - deviance / null_deviance if null_deviance > 0 else math.nan
            )
    info = {
        "result": result,
        "predicted": predicted,
        "params": params,
        "pvalues": pvalues,
        "model_p_value": model_p,
        "train_r2": r_squared,
        "pseudo_r2": pseudo_r_squared,
        "aic": aic,
        "design_train": design_train,
    }
    return info

def subset_training_design(design: FeatureModel, mask: Any) -> FeatureModel:
    return FeatureModel(
        train=design.train.loc[mask].copy(),
        test=design.test,
        groups={name: columns.copy() for name, columns in design.groups.items()},
        dropped=design.dropped.copy(),
    )

def group_p_value(result: Any, columns: list[str]) -> float:
    if not columns:
        return math.nan
    import numpy as np

    names = list(result.model.exog_names)
    positions = [names.index(name) for name in columns if name in names]
    if not positions:
        return math.nan
    contrast = np.zeros((len(positions), len(names)))
    for row, position in enumerate(positions):
        contrast[row, position] = 1.0
    try:
        test = result.wald_test(contrast, scalar=True)
        value = float(np.asarray(test.pvalue).squeeze())
        return value
    except (ValueError, np.linalg.LinAlgError):
        return math.nan

def backward_elimination(
    model_type: str,
    design: FeatureModel,
    train_y: Any,
    test_y: Any,
    test_observed_mask: Any,
    alpha: float,
    train_successes: Any | None = None,
    train_trials: Any | None = None,
    selection_kind: str = "proportion",
):
    import numpy as np
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    is_ridge = model_type in {"fractional_logit_ridge", "binary_logit_ridge"}
    active = (
        []
        if model_type == "intercept_only"
        else [name for name, columns in design.groups.items() if columns]
    )
    history: list[dict[str, Any]] = []
    while True:
        columns = [column for name in active for column in design.groups[name]]
        try:
            fitted = fit_regression(
                model_type,
                design.train[columns],
                train_y,
                design.test[columns],
                train_successes=train_successes,
                train_trials=train_trials,
            )
        except (
            AnalysisError,
            ValueError,
            np.linalg.LinAlgError,
            OverflowError,
            PerfectSeparationError,
        ) as exc:
            return {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "history": history,
            }
        result = fitted["result"]
        feature_pvalues = {
            name: group_p_value(result, design.groups[name]) for name in active
        } if result is not None and not is_ridge else {}
        if is_ridge:
            break
        if not active:
            break
        worst_name = max(
            feature_pvalues,
            key=lambda name: (
                feature_pvalues[name]
                if math.isfinite(feature_pvalues[name])
                else 1.0
            ),
        )
        worst_p = feature_pvalues[worst_name]
        history.append(
            {
                "step": len(history) + 1,
                "feature": worst_name,
                "joint_p_value": worst_p,
                "removed": bool(math.isfinite(worst_p) and worst_p > alpha and len(active) > 1),
            }
        )
        if math.isfinite(worst_p) and worst_p > alpha and len(active) > 1:
            active.remove(worst_name)
            continue
        break

    test_pred = fitted["predicted"][test_observed_mask]
    observed_test_y = np.asarray(test_y, dtype=float)[test_observed_mask]
    rmse = (
        float(np.sqrt(np.mean((observed_test_y - test_pred) ** 2)))
        if len(observed_test_y)
        else math.nan
    )
    test_mean = float(np.mean(observed_test_y)) if len(observed_test_y) else math.nan
    ss_total = (
        float(np.sum((observed_test_y - test_mean) ** 2))
        if len(observed_test_y)
        else math.nan
    )
    ss_residual = (
        float(np.sum((observed_test_y - test_pred) ** 2))
        if len(observed_test_y)
        else math.nan
    )
    test_r2 = 1.0 - ss_residual / ss_total if ss_total > 0 else math.nan
    test_sd = float(np.std(observed_test_y, ddof=1)) if len(observed_test_y) > 1 else math.nan
    log_rmse = math.nan
    if selection_kind in {"positive_continuous", "count"} and len(observed_test_y):
        log_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.log1p(np.maximum(observed_test_y, 0.0))
                        - np.log1p(np.maximum(test_pred, 0.0))
                    )
                    ** 2
                )
            )
        )
    fitted.update(
        {
            "status": "ok",
            "selected_features": active.copy(),
            "feature_pvalues": {
                name: group_p_value(fitted["result"], design.groups[name])
                for name in active
            } if fitted["result"] is not None and not is_ridge else {},
            "history": history,
            "rmse": rmse,
            "log_rmse": log_rmse,
            "selection_rmse": (
                log_rmse if selection_kind in {"positive_continuous", "count"} else rmse
            ),
            "test_r2": test_r2,
            "test_nrmse": rmse / test_sd if test_sd > 0 else math.nan,
            "test_observed_count": len(observed_test_y),
        }
    )
    return fitted

def two_part_fractional_regression(
    design: FeatureModel,
    train_y: Any,
    test_y: Any,
    test_observed_mask: Any,
    alpha: float,
):
    import numpy as np

    train_values = np.asarray(train_y, dtype=float)
    test_values = np.asarray(test_y, dtype=float)
    complete_train = train_values >= 1.0 - 1e-10
    complete_test = test_values >= 1.0 - 1e-10
    partial_train = ~complete_train
    partial_test_observed = test_observed_mask & ~complete_test
    complete_labels = complete_train.astype(float)
    complete_test_values = np.where(test_observed_mask, complete_test.astype(float), np.nan)

    if np.unique(complete_labels).size > 1:
        complete_fit = backward_elimination(
            "binary_logit_ridge",
            design,
            complete_labels,
            complete_test_values,
            test_observed_mask,
            alpha,
        )
        if complete_fit["status"] != "ok":
            return complete_fit
        complete_probability = complete_fit["predicted"]
    else:
        complete_probability = np.full(len(test_values), complete_labels[0], dtype=float)
        complete_fit = {
            "status": "ok",
            "selected_features": [],
            "feature_pvalues": {},
            "history": [],
            "effects": {},
        }

    partial_y = train_values[partial_train]
    if not len(partial_y):
        return {
            "status": "failed",
            "failure_reason": "two-part model has no incomplete training observations",
            "history": [],
        }
    partial_design = subset_training_design(design, partial_train)
    if np.unique(partial_y).size > 1:
        partial_fit = backward_elimination(
            "fractional_logit_ridge",
            partial_design,
            partial_y,
            test_values,
            partial_test_observed,
            alpha,
        )
        if partial_fit["status"] != "ok":
            return partial_fit
        partial_expectation = partial_fit["predicted"]
    else:
        partial_expectation = np.full(len(test_values), float(partial_y[0]), dtype=float)
        partial_fit = {
            "status": "ok",
            "selected_features": [],
            "feature_pvalues": {},
            "history": [],
            "effects": {},
        }

    predicted = complete_probability + (1.0 - complete_probability) * partial_expectation
    effects_by_stage = {
        "fully_complete_probability": (
            standardized_effects(complete_fit, design)
            if complete_fit.get("result") is not None
            else {}
        ),
        "incomplete_completion_proportion": (
            standardized_effects(partial_fit, partial_design)
            if partial_fit.get("result") is not None
            else {}
        ),
    }
    effect_candidates: dict[str, list[dict[str, Any]]] = {}
    for stage, effects in effects_by_stage.items():
        for feature, effect in effects.items():
            effect_candidates.setdefault(feature, []).append({**effect, "stage": stage})
    combined_effects: dict[str, dict[str, Any]] = {}
    for feature, candidates in effect_candidates.items():
        strongest = max(
            candidates,
            key=lambda item: abs(item["standardized_coefficient"]),
        )
        combined_effects[feature] = {
            **strongest,
            "direction": "stage-specific; see model comparison",
        }

    selected = sorted(
        set(complete_fit.get("selected_features", []))
        | set(partial_fit.get("selected_features", []))
    )
    feature_pvalues: dict[str, float] = {}
    for fitted in (complete_fit, partial_fit):
        for feature, value in fitted.get("feature_pvalues", {}).items():
            feature_pvalues[feature] = min(feature_pvalues.get(feature, 1.0), value)

    observed = np.asarray(test_y, dtype=float)[test_observed_mask]
    predictions = predicted[test_observed_mask]
    rmse = float(np.sqrt(np.mean((observed - predictions) ** 2)))
    total = float(np.sum((observed - observed.mean()) ** 2))
    residual = float(np.sum((observed - predictions) ** 2))
    standard_deviation = float(np.std(observed, ddof=1)) if len(observed) > 1 else math.nan
    histories = [
        {"stage": "fully_complete_probability", **item}
        for item in complete_fit.get("history", [])
    ] + [
        {"stage": "incomplete_completion_proportion", **item}
        for item in partial_fit.get("history", [])
    ]
    return {
        "status": "ok",
        "model_type": "two_part_fractional",
        "result": None,
        "predicted": np.clip(predicted, 0.0, 1.0),
        "selected_features": selected,
        "feature_pvalues": feature_pvalues,
        "history": histories,
        "effects": combined_effects,
        "stage_effects": effects_by_stage,
        "stage_feature_p_values": {
            "fully_complete_probability": complete_fit.get("feature_pvalues", {}),
            "incomplete_completion_proportion": partial_fit.get("feature_pvalues", {}),
        },
        "model_p_value": math.nan,
        "aic": math.nan,
        "train_r2": math.nan,
        "pseudo_r2": math.nan,
        "rmse": rmse,
        "selection_rmse": rmse,
        "test_r2": 1.0 - residual / total if total > 0 else math.nan,
        "test_nrmse": rmse / standard_deviation if standard_deviation > 0 else math.nan,
        "test_observed_count": len(observed),
        "train_observed_count": len(train_values),
    }

def model_types(kind: str, y_train: Any) -> list[str]:
    import numpy as np

    if kind == "ordinal_score":
        return ["ordinal_score_ols", "intercept_only"]
    if kind == "binary":
        return ["logit", "binary_logit_ridge", "ols_lpm", "intercept_only"]
    if kind == "count":
        return ["poisson_glm", "lognormal_ols", "intercept_only"]
    if kind == "grouped_proportion":
        return ["grouped_binomial_glm", "intercept_only"]
    if kind == "two_part_proportion":
        return [
            "two_part_fractional",
            "fractional_logit_ridge",
            "fractional_logit",
            "intercept_only",
        ]
    if kind == "proportion":
        return ["fractional_logit_ridge", "fractional_logit", "ols", "intercept_only"]
    if kind == "positive_continuous" and np.nanmin(y_train) >= 0:
        candidates = ["lognormal_ols", "intercept_only"]
        if np.nanmin(y_train) > 0:
            candidates.insert(0, "gamma_glm")
        return candidates
    return ["ols"]

def standardized_effects(
    model: dict[str, Any], design: FeatureModel
) -> dict[str, dict[str, Any]]:
    params = model["params"]
    pvalues = model["pvalues"]
    names = list(model["result"].model.exog_names)
    effects: dict[str, dict[str, Any]] = {}
    for feature in model["selected_features"]:
        columns = design.groups[feature]
        indices = [names.index(name) for name in columns if name in names]
        if not indices:
            continue
        values = [(float(params[index]), float(pvalues[index])) for index in indices]
        strongest = max(values, key=lambda pair: abs(pair[0]))
        if len(values) == 1:
            direction = "positive" if strongest[0] > 0 else "negative"
        else:
            direction = "mixed category-level effects"
        effects[feature] = {
            "standardized_coefficient": strongest[0],
            "p_value": strongest[1],
            "direction": direction,
        }
    return effects

def regression_candidates(kind: str, y_train: Any) -> list[str]:
    return model_types(kind, y_train)

def evaluate_models_for_target(
    frame: Any,
    target: str,
    train_mask: Any,
    test_mask: Any,
    manifest: list[dict[str, str]],
    args: argparse.Namespace,
):
    import numpy as np
    import pandas as pd

    metadata = ALL_TARGETS.get(
        target,
        {"group": "success", "kind": "two_part_proportion", "label": "Composite success metric"},
    )
    numeric_target = pd.to_numeric(frame[target], errors="coerce")
    usable_mask = numeric_target.notna() & np.isfinite(numeric_target)
    success_column = metadata.get("success_column")
    trials_column = metadata.get("trials_column")
    if success_column and trials_column:
        target_successes = pd.to_numeric(frame[success_column], errors="coerce")
        target_trials = pd.to_numeric(frame[trials_column], errors="coerce")
        usable_mask &= target_successes.notna() & target_trials.notna() & target_trials.gt(0)
    train_rows = frame.loc[train_mask & usable_mask].copy()
    test_rows = frame.loc[test_mask].copy()
    if len(train_rows) > args.max_train_rows > 0:
        train_rows = train_rows.sample(args.max_train_rows, random_state=SEED)
    train_y = numeric_target.loc[train_rows.index].to_numpy(dtype=float)
    test_y = numeric_target.loc[test_rows.index].to_numpy(dtype=float)
    test_observed_mask = np.isfinite(test_y)
    train_successes = (
        pd.to_numeric(train_rows[success_column], errors="coerce").to_numpy(dtype=float)
        if success_column
        else None
    )
    train_trials = (
        pd.to_numeric(train_rows[trials_column], errors="coerce").to_numpy(dtype=float)
        if trials_column
        else None
    )
    if len(train_rows) < args.min_train_rows:
        return [], {}, {"status": "skipped", "reason": f"only {len(train_rows)} training rows"}
    if np.unique(train_y).size < 2:
        return (
            [
                {
                    "target": target,
                    "target_label": metadata["label"],
                    "variant": "all",
                    "model_type": "",
                    "status": "skipped",
                    "train_rows": len(train_rows),
                    "test_rows": int(test_observed_mask.sum()),
                    "failure_reason": "target is constant in training data",
                }
            ],
            {},
            {"status": "skipped", "reason": "target is constant in training data"},
        )
    if not test_observed_mask.any():
        return (
            [],
            {},
            {"status": "skipped", "reason": "no observed held-out target values"},
        )

    selected_manifest = [
        row
        for row in manifest
        if manifest_selected(row)
        and row["type"].strip().lower() in {"numeric", "binary", "categorical"}
        and (row["name"] in frame)
    ]
    target_exclusions: dict[str, str] = {}
    if (
        target in SPEC_SCORE_TARGETS
        and getattr(args, "spec_score_method", "") == "deterministic_regex_heuristic"
    ):
        for row in selected_manifest:
            if row["name"] in HEURISTIC_TARGET_INPUT_EXCLUSIONS:
                target_exclusions[row["name"]] = (
                    "overlaps with heuristic score logic: "
                    + HEURISTIC_TARGET_INPUT_EXCLUSIONS[row["name"]]
                )
        selected_manifest = [
            row for row in selected_manifest
            if row["name"] not in target_exclusions
        ]
    variants = {"all_inputs": selected_manifest}
    comparison_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    screen_reasons: dict[str, str] = dict(target_exclusions)
    for variant, variant_features in variants.items():
        feature_model = prepare_features(
            train_rows,
            test_rows,
            variant_features,
            args.min_category_count,
            args.max_missing_fraction,
            args.max_correlation,
        )
        for feature, reason in feature_model.dropped.items():
            screen_reasons.setdefault(feature, reason)
        for regression_type in regression_candidates(metadata["kind"], train_y):
            if regression_type == "two_part_fractional":
                result = two_part_fractional_regression(
                    feature_model,
                    train_y,
                    test_y,
                    test_observed_mask,
                    args.alpha,
                )
            else:
                result = backward_elimination(
                    regression_type,
                    feature_model,
                    train_y,
                    test_y,
                    test_observed_mask,
                    args.alpha,
                    train_successes=train_successes,
                    train_trials=train_trials,
                    selection_kind=metadata["kind"],
                )
            row = {
                "target": target,
                "target_label": metadata["label"],
                "variant": variant,
                "model_type": regression_type,
                "status": result["status"],
                "train_rows": len(train_rows),
                "test_rows": int(test_observed_mask.sum()),
                "failure_reason": result.get("failure_reason", ""),
            }
            if result["status"] == "ok":
                row.update(
                    {
                        "selected_features": ";".join(result["selected_features"]),
                        "n_selected_features": len(result["selected_features"]),
                        "model_p_value": result["model_p_value"],
                        "aic": result["aic"],
                        "train_r2": result["train_r2"],
                        "pseudo_r2": result["pseudo_r2"],
                        "test_rmse": result["rmse"],
                        "test_log_rmse": result.get("log_rmse", math.nan),
                        "selection_metric": (
                            "log1p_rmse"
                            if metadata["kind"] in {"positive_continuous", "count"}
                            else "rmse"
                        ),
                        "selection_rmse": result["selection_rmse"],
                        "test_r2": result["test_r2"],
                        "test_nrmse": result["test_nrmse"],
                        "backward_steps": len(result["history"]),
                        "selected_feature_p_values": json.dumps(
                            result["feature_pvalues"], sort_keys=True
                        ),
                        "backward_elimination_history": json.dumps(
                            result["history"], sort_keys=True
                        ),
                        "stage_feature_p_values": json.dumps(
                            result.get("stage_feature_p_values", {}), sort_keys=True
                        ),
                        "stage_effects": json.dumps(
                            result.get("stage_effects", {}), sort_keys=True
                        ),
                    }
                )
                effects = result.get("effects")
                if effects is None:
                    effects = (
                        standardized_effects(result, feature_model)
                        if result.get("result") is not None
                        else {}
                    )
                result.update(
                    {
                        "target": target,
                        "variant": variant,
                        "model_type": regression_type,
                        "feature_model": feature_model,
                        "effects": effects,
                        "stage_effects": result.get("stage_effects", {}),
                        "test_indices": test_rows.index.to_list(),
                        "actual": test_y,
                        "train_observed_count": len(train_rows),
                    }
                )
                successful.append(result)
            comparison_rows.append(row)
    if not successful:
        return comparison_rows, screen_reasons, {"status": "skipped", "reason": "no regression fit succeeded"}
    best = min(
        successful,
        key=lambda result: (
            result["selection_rmse"]
            if math.isfinite(result["selection_rmse"])
            else float("inf"),
            len(result["selected_features"]),
        ),
    )
    return comparison_rows, screen_reasons, best

def model_and_report(args: argparse.Namespace) -> None:
    import pandas as pd

    dataset_path = OUTPUT_DIR / "modeling_dataset.csv"
    if not dataset_path.exists():
        raise AnalysisError("Modeling dataset missing; run the extract phase first.")
    write_rq2_dataset(dataset_path)
    manifest = read_manifest()
    frame = pd.read_csv(
        dataset_path,
        low_memory=False,
        dtype={"file_url_sha16": str, "repo_name": str},
    )
    missing_targets = sorted(
        (set(TARGETS) | set(ADDITIONAL_TARGETS)) - set(frame.columns)
    )
    if missing_targets:
        raise AnalysisError(f"Extracted dataset is missing targets: {missing_targets}")
    if "repo_name" not in frame:
        raise AnalysisError("Extracted dataset is missing its repository grouping key.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    clear_generated_plots()
    train_mask, test_mask = split_repositories(frame, SEED)
    frame = add_success_metric(frame, train_mask, args.success_min_groups)
    score_source_path = OUTPUT_DIR / "spec_score_source.json"
    score_source = (
        json.loads(score_source_path.read_text(encoding="utf-8"))
        if score_source_path.exists()
        else {}
    )
    configuration = {
        "dataset_rows": len(frame),
        "target_count": len(TARGETS)
        + len(ADDITIONAL_TARGETS)
        + sum(target in frame.columns for target in SPEC_SCORE_TARGETS),
        "spec_score_targets_present": [
            target for target in SPEC_SCORE_TARGETS if target in frame.columns
        ],
        "spec_score_method": score_source.get("method", "unspecified"),
        "spec_score_cache": score_source.get("cache_file", ""),
        "composite_target": "target_success",
        "train_repository_fraction": 0.8,
        "repository_grouped_split": True,
        "random_seed": SEED,
        "max_train_rows_per_target": args.max_train_rows,
        "min_train_rows": args.min_train_rows,
        "min_category_count": args.min_category_count,
        "max_missing_fraction": args.max_missing_fraction,
        "max_numeric_spearman_correlation": args.max_correlation,
        "backward_elimination_alpha": args.alpha,
        "ridge_penalty_alpha": RIDGE_ALPHA,
        "success_min_groups": args.success_min_groups,
        "aggregate_targets": [
            target for members in TARGET_GROUPS.values() for target in members
        ],
        "additional_targets_excluded_from_aggregate": list(ADDITIONAL_TARGETS),
        "heuristic_score_model_input_exclusion": (
            list(HEURISTIC_TARGET_INPUT_EXCLUSIONS)
        ),
    }
    args.spec_score_method = score_source.get(
        "method", getattr(args, "spec_score_method", "unspecified")
    )
    (OUTPUT_DIR / "analysis_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    target_names = [
        *TARGETS,
        *ADDITIONAL_TARGETS,
        *(target for target in SPEC_SCORE_TARGETS if target in frame.columns),
        "target_success",
    ]
    comparison_rows: list[dict[str, Any]] = []
    screen_reasons_by_target: dict[str, dict[str, str]] = {}
    best_by_target: dict[str, dict[str, Any]] = {}
    skipped_by_target: dict[str, dict[str, str]] = {}
    for target in target_names:
        comparisons, screen_reasons, best = evaluate_models_for_target(
            frame, target, train_mask, test_mask, manifest, args
        )
        comparison_rows.extend(comparisons)
        screen_reasons_by_target[target] = screen_reasons
        if best.get("status") == "ok":
            best_by_target[target] = best
        else:
            skipped_by_target[target] = best

    write_model_results(
        frame=frame,
        target_names=target_names,
        comparison_rows=comparison_rows,
        screen_reasons_by_target=screen_reasons_by_target,
        best_by_target=best_by_target,
        skipped_by_target=skipped_by_target,
        manifest=manifest,
        train_mask=train_mask,
        success_min_groups=args.success_min_groups,
    )
