# SpecMine Regression Methodology

## Goal and analysis unit

The analysis estimates associations between a fixed set of specification and
repository characteristics and independently modeled outcome measures. The
observation is one `spec_files.file_url_sha16`. A deterministic
repository-grouped holdout split is used so a repository does not ordinarily
appear in both training and test data.

Input candidates are direct database values or simple relation counts listed
in `input_candidates.csv`. The completeness and quality
rubric scores are outcome measures only.

## Outcome definitions

PR records are associated to a specification through an exact
`pr_files.repo_name` and `pr_files.file_path` match. The join does not filter on
`pr_files.is_spec` or require its tool to match `spec_files.spec_tool`. Each
matched PR is then joined using the tool, repository, and PR number carried by
the PR-file row. Duplicate PR associations are collapsed before aggregation.
OpenSpec artifacts are scoped to the change directory represented by the
specification's path before `/specs/`; code references are first aggregated
within each artifact.

| Outcome | Calculation | Modeling type | In `target_success`? |
|---|---|---|---|
| `target_pr_closed_merge_proportion` | Associated closed PRs marked merged divided by associated closed PRs; missing if no associated PR is closed. The numerator and denominator are retained as `target_pr_closed_merged_count` and `target_pr_closed_count`. | Grouped-binomial GLM | Yes |
| `target_avg_pr_merge_hours` | Mean elapsed hours from PR creation to merge across associated merged PRs with both timestamps. | Positive continuous | Yes; reverse-scored |
| `target_all_tasks_completion_proportion_avg` | Mean of per-artifact `n_tasks_done / n_tasks`, excluding artifacts with zero or missing planned tasks. | Two-part proportion | Yes |
| `target_code_refs_task_done_proportion_avg` | For each artifact, distinct code references marked done divided by distinct references with a known done status; mean those artifact-level proportions. | Proportion | Yes |
| `target_pr_any_merged` | 1 when any associated PR has `is_merged = 1`; otherwise 0. | Binary | No |
| `target_all_tasks_done_count` | Sum of `n_tasks_done` over matched OpenSpec artifacts; 0 when there are no matched artifacts. | Count | No |
| `target_all_tasks_remaining_count` | Sum of `n_tasks - n_tasks_done` over matched artifacts with known task values; 0 when there are no matched artifacts. Negative artifact differences are floored at zero. | Count | No |
| `target_all_tasks_done_incomplete_ratio_avg` | Mean per-artifact ratio `n_tasks_done / (n_tasks - n_tasks_done)` where both counts are known and the incomplete-task denominator is positive. | Nonnegative continuous | No |
| `target_tasks_md_done_count` | Sum of `n_tasks_done` for matched artifacts with `kind = 'tasks.md'`; 0 when none match. | Count | No |
| `target_tasks_md_remaining_count` | Sum of `n_tasks - n_tasks_done` for matched `tasks.md` artifacts with known task values; 0 when none match. Negative differences are floored at zero. | Count | No |
| `target_tasks_md_done_incomplete_ratio_avg` | Mean per-artifact `n_tasks_done / (n_tasks - n_tasks_done)` for `tasks.md` artifacts with known counts and a positive incomplete-task denominator. | Nonnegative continuous | No |
| `target_spec_completeness` | Score from 0 to 5 for how fully the specification addresses the completeness rubric. | Ordinal score | No |
| `target_spec_quality` | Score from 0 to 5 for testability, ambiguity, contradiction, and structure under the quality rubric. | Ordinal score | No |

Undefined proportions or ratios remain missing rather than being converted to
zero. The count outcomes are zero when no matching artifacts exist. All
additional outcomes are modeled separately; they do not alter the aggregate.
The `tasks.md` count and ratio outcomes are kept separate even where their
values may coincide with all-artifact measures in this dataset.

## Aggregate `target_success`

`target_success` is a continuous score on the 0-to-1 scale. Its definition is
intentionally limited to the four rows marked **Yes** above:

1. Reverse-score average PR merge hours using the empirical distribution of
   training rows, so shorter merge times receive higher scores.
2. Keep the three proportion targets on their existing 0-to-1 scales.
3. Treat each of those four outcomes as its own group with equal group weight.
4. Average the group scores available for each observation and require at
   least two observed groups; otherwise `target_success` is missing.
5. Fit the percentile reference for merge time on training rows only, then
   apply it to training and held-out rows.

The additional PR/task outcomes and the two specification-rubric outcomes are
never included in the aggregate. `target_inventory.csv` records inclusion
status for every modeled outcome.

## Specification rubric scoring

Both rubric outcomes use the full `content` field in
`specmine-zenodo/specs.jsonl/specs.jsonl`, joined to extracted rows by
`file_url_sha16`. They are scored independently and never become predictors.

The optional `spec-scores` phase calls a local OpenAI-compatible LLM endpoint.
The cache in `output/spec_scores.csv` is versioned by model and prompt and
stores a SHA-256 source digest and concise evidence. Text is submitted in
sequential, non-overlapping chunks of up to 8,000 characters; a final local
request combines the chunk evidence into whole-spec scores.

The `spec-scores-heuristic` phase provides a deterministic proxy without a
model or network. The versioned `output/spec_scores_heuristic.csv` cache stores
the method, source digest, and evidence. Completeness checks the five required
section types (Architecture, Error Handling, Security, Edge Cases, and Data
Models), along with actionability, word count, requirement-like lines, lists,
and code blocks. Quality uses requirement-like lines, measurable conditions,
vague-language signals, limited direct-conflict detection, and structural
signals. These rules can miss paraphrases or misread benign wording; the
result is a proxy, not an LLM assessment or validated ground truth.

When the active method is the deterministic heuristic, selected inputs whose
meaning directly overlaps with score rules are excluded from both rubric-score
models. The affected candidates are specification byte/word/line length,
heading, list, checkbox, and code-fence counts; normative and Gherkin counts;
EARS, Gherkin, requirements, acceptance-criteria, user-scenario, edge-case,
mandatory-marker, and tiny-document indicators. The exclusion is target-specific and is recorded in the matching
`*_exclusion_reason` columns of `feature_results.csv`. Other specification
content characteristics remain eligible subject to ordinary screening. This
prevents direct reuse of the heuristic's ingredients as predictors without
discarding every content-derived feature.

The score anchors are:

| Score | Completeness | Quality |
|---:|---|---|
| 5 | All required sections are present. | Requirements are quantifiable and testable, without ambiguous language. |
| 3 | Core functionality is detailed but secondary operational sections are missing. | Functional requirements rely on some vague terms. |
| 1 | High-level summary without actionable technical detail. | Contradictory, poorly structured, or impractical requirements. |
| 0, 2, 4 | 0 means absent or unusable text; 2 and 4 interpolate between anchors. | 0 means absent or unusable text; 2 and 4 interpolate between anchors. |

The score models treat the ordered 0-to-5 outcome as approximately numeric.
Coefficients are score-point associations per standard-deviation change in an
input; equal spacing between rubric points is not assumed to be exact.

## Static input candidates and screening

The candidate manifest records a feature name, type, source, reason, default
inclusion switch, and timing/leakage risk. Selected predictors are direct
`spec_files` or `spec_content_features` values or simple `spec_links` counts
grouped by specification and link relation. Text is not sent to an LLM to
create predictors. Edit `selected` in the manifest to curate the feature set.

Screening and transformations use training rows only:

| Decision | Default | Mechanism |
|---|---:|---|
| Holdout | 20% of repositories | Deterministic seed 625; repository groups remain disjoint. If only one repository is available, split rows instead. |
| Training-size cap | 50,000 rows per target | Deterministic subsample when eligible training data exceed the cap; held-out rows are not capped. Set `--max-train-rows 0` to disable the cap. |
| Minimum training size | 100 rows | A target below this threshold is reported as skipped. |
| Missingness | Remove above 80% | Fraction is computed in the target's training rows. Numeric missing values otherwise use the training median. |
| Numeric variance | At least two values and nonzero post-imputation variance | Constant inputs are removed. |
| Numeric collinearity | Absolute Spearman correlation at least 0.98 | The later candidate is dropped, and the retained feature is named in the reason. |
| Category support | Up to 60 levels; at least 20 observations per retained level | Rare levels are grouped; unseen levels map to the training reference category. |
| Encoded collinearity | Absolute correlation at least 0.999999 | Duplicate design columns are removed before fitting. |
| Scaling | Training mean 0, standard deviation 1 | Numeric and one-hot columns use training statistics. |
| Backward elimination | Robust Wald p-value threshold 0.05 | For unregularized models, remove the least-significant input term iteratively, refitting until terms meet the threshold or one remains. Categorical dummy columns are tested jointly. |

## Regression models

The model runner compares only candidates appropriate to each outcome. It
chooses the successful candidate with the lowest held-out RMSE, using
log1p-RMSE for merge durations and count outcomes; fewer predictors break ties.
Every candidate includes an intercept-only benchmark. The target is skipped if
it is constant in training data.

| Outcome shape | Compared models | Rationale |
|---|---|---|
| Closed-PR merge proportion | Grouped-binomial GLM with logit link; intercept-only | Uses the count of merged PRs and the count of closed PRs, so estimates reflect available PR trials. Holdout error is still measured on spec-level proportions. |
| Any associated PR merged | Logistic regression, ridge logistic regression, linear probability model, intercept-only | Logistic likelihood suits a binary outcome; ridge stabilizes high-dimensional fits, and the linear probability model is a bounded-scale benchmark. |
| Average merge hours | Gamma GLM with log link when strictly positive; lognormal-style OLS on `log1p(hours)`; intercept-only | Handles right-skew and nonnegative duration; model selection uses log1p-RMSE to reduce domination by extreme waits. |
| Task completion proportions | Two-part ridge-fractional model, ridge/unregularized fractional-logit, OLS benchmark where appropriate, intercept-only | The two-part model estimates perfect completion probability separately from completion among non-perfect cases. Fractional-logit handles values at 0 and 1. |
| Completed/remaining task counts | Poisson GLM, lognormal-style OLS on `log1p(count)`, intercept-only | Poisson models nonnegative counts; the lognormal-style alternative offers a robust comparison for overdispersed counts. Selection uses log1p-RMSE. |
| Completed-to-incomplete ratios | Gamma GLM when values are strictly positive; lognormal-style OLS on `log1p(ratio)`; intercept-only | Models nonnegative, right-skewed ratios; zero values rule out Gamma and leave the lognormal-style alternative. |
| Code-reference completion proportion | Ridge or unregularized fractional-logit GLM, clipped OLS benchmark, intercept-only | Models an artifact-averaged bounded fraction without pooling references across artifacts. |
| Completeness/quality scores | HC3 OLS on the original 0-to-5 scale; intercept-only | Keeps score-point effects interpretable; predictions are clipped to the rubric range. |
| Aggregate success | Two-part ridge-fractional, ridge/unregularized fractional-logit, intercept-only | Accommodates a bounded score that may include a concentration at perfect success. |

Regularized fits are predictive comparisons and do not provide coefficient
p-values or use p-value-based backward elimination. Unregularized models use
HC3 robust covariance estimates. Reported test statistics include RMSE,
normalized RMSE, R-squared or pseudo-R-squared as appropriate, AIC, and model
p-value where defined. A small p-value is not evidence of practical importance;
coefficients are interpreted with held-out performance and the feature
selection history.

## Analysis flow

| Step | Action | Purpose and result |
|---:|---|---|
| 1 | Read the fixed candidate manifest and build a safe SQL projection. | Only manually selected database predictors enter the extracted dataset. |
| 2 | Join specification rows to content metrics and simple link counts. | Creates one modeling row per specification hash. |
| 3 | Match associated PRs and calculate merge outcomes. | Produces the closed-PR denominator/numerator, merge proportion, merge duration, and any-merge indicator. |
| 4 | Match OpenSpec artifacts to the specification's change path. | Prevents unrelated changes in the same repository from contributing task outcomes. |
| 5 | Aggregate artifacts and code references. | Produces task counts, proportions, ratios, and artifact-averaged code-reference completion. |
| 6 | Optionally score full spec Markdown by local LLM or deterministic heuristic. | Adds two separate rubric outcomes and records method-specific provenance. |
| 7 | Split by repository and compute `target_success` from only its four designated outcomes. | Avoids repository leakage and keeps non-composite targets separate. |
| 8 | Exclude direct heuristic-rule inputs from rubric-score models when heuristic scores are active. | Reduces direct target leakage; records each exclusion per target. |
| 9 | Apply missingness, variance, category, and correlation screens. | Removes uninformative or redundant predictors without using held-out rows. |
| 10 | Fit candidate regressions and perform backward elimination where supported. | Estimates adjusted associations and compares suitable model families per target. |
| 11 | Select by held-out error and calculate standardized effects. | Provides predictive comparison and within-model relative association strength. |
| 12 | Write target, feature, comparison, and prediction tables. | Preserves formulas, decisions, effects, and evaluation statistics for review. |
| 13 | Create residual, coefficient, predicted-vs-actual, correlation, and composite-comparison plots. | Helps inspect fit, residual structure, relative coefficients, and aggregate consistency. |

## Output files

| File | Contents |
|---|---|
| `output/modeling_dataset.csv` | Extracted candidate inputs, outcomes, PR trial counts, and task metrics. |
| `output/spec_scores.csv` | Optional versioned local-LLM rubric cache, source digest, and evidence. |
| `output/spec_scores_heuristic.csv` | Optional heuristic rubric cache, source digest, and rule evidence. |
| `output/spec_score_source.json` | Active rubric-scoring method and cache provenance. |
| `output/target_inventory.csv` | Every modeled target, source, formula, and aggregate inclusion flag. |
| `output/analysis_config.json` | Row count, split, screening/model configuration, aggregate list, and heuristic exclusion policy. |
| `output/model_comparison.csv` | Candidate model types, statistics, held-out errors, selected terms, and elimination history. |
| `output/target_model_results.csv` | Best model per individual outcome and composite, with selected predictors and effects. |
| `output/feature_results.csv` | One row per candidate, provenance, description, exclusion reason, and target-specific effects. |
| `output/predictions.csv` | Held-out actuals, predictions, residuals, and model type for each final model. |
| `output/individual_vs_success_performance.csv` | Performance for each modeled individual outcome compared with the aggregate; an inclusion flag distinguishes composite members. |
| `output/success_prediction_comparison.csv` | Direct composite predictions compared with post-hoc aggregation of individual predictions. |
| `output/images/*.png` | Per-target residual, coefficient, predicted-vs-actual, and correlation plots plus composite comparison. |

Relationship columns in `feature_results.csv` are blank for a feature excluded
from that target's final model. `*_exclusion_reason` columns explain screening,
heuristic-rule exclusions, or other target-specific omissions.

## Configuration and validation

The CLI supports `--max-train-rows`, `--min-train-rows`,
`--min-category-count`, `--max-missing-fraction`, `--max-correlation`,
`--success-min-groups`, `--alpha`, `--spec-score-limit`, and the local-score
model/endpoint/timeout arguments. See `README.md` for commands.

Run `python .\\specmine_analysis\\run_analysis.py self-test` for synthetic
validation of composite isolation, model families, heuristic leakage
exclusions, reporting, score-cache behavior, and plots without connecting to
MySQL.
