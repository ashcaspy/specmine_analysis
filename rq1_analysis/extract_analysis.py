"""Database extraction and rubric-score preparation for RQ1."""

from __future__ import annotations

from analysis_utils import (
    AnalysisError,
    BASE_DIR,
    HEURISTIC_SCORE_VERSION,
    OUTPUT_DIR,
    Path,
    SPEC_SCORE_PROMPT_VERSION,
    SPEC_SCORE_TARGETS,
    WORKSPACE_ROOT,
    argparse,
    build_feature_select,
    connect_db,
    csv,
    hashlib,
    heuristic_spec_scores,
    json,
    os,
    read_manifest,
    score_spec_content,
    write_rq2_dataset,
    write_spec_score_source,
)


def extract_dataset() -> Path:
    import pymysql

    manifest = read_manifest()
    query_path = BASE_DIR / "extract.sql"
    query = query_path.read_text(encoding="utf-8").replace(
        "{FEATURE_SELECT}", build_feature_select(manifest)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "modeling_dataset.csv"
    temporary_path = OUTPUT_DIR / "modeling_dataset.csv.tmp"
    connection = connect_db()
    count = 0
    try:
        with connection.cursor(pymysql.cursors.SSCursor) as cursor:
            cursor.execute(query)
            names = [item[0] for item in cursor.description]
            with temporary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(names)
                while True:
                    rows = cursor.fetchmany(5000)
                    if not rows:
                        break
                    writer.writerows(rows)
                    count += len(rows)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    temporary_path.replace(output_path)
    (OUTPUT_DIR / "spec_score_source.json").unlink(missing_ok=True)
    (OUTPUT_DIR / "modeling_dataset_sub.csv").unlink(missing_ok=True)
    print(f"Extracted {count:,} specs into {output_path}")
    return output_path

def run_heuristic_spec_scores(args: argparse.Namespace) -> Path:
    import pandas as pd

    source_path = WORKSPACE_ROOT / "specmine-zenodo" / "specs.jsonl" / "specs.jsonl"
    dataset_path = OUTPUT_DIR / "modeling_dataset.csv"
    cache_path = OUTPUT_DIR / "spec_scores_heuristic.csv"
    if not source_path.exists():
        raise AnalysisError(f"Spec text source is missing: {source_path}")
    if not dataset_path.exists():
        raise AnalysisError("Run the extract phase before heuristic spec scoring.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = HEURISTIC_SCORE_VERSION
    completed: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("cache_key") != cache_key:
                    raise AnalysisError(
                        "Heuristic score cache version differs from this run. "
                        "Move output/spec_scores_heuristic.csv aside to rescore."
                    )
                completed[row["file_url_sha16"]] = row

    processed = 0
    mode = "a" if cache_path.exists() else "w"
    fields = [
        "file_url_sha16",
        *SPEC_SCORE_TARGETS,
        "completeness_evidence",
        "quality_evidence",
        "content_sha256",
        "scoring_method",
        "cache_key",
    ]
    with source_path.open(encoding="utf-8") as source, cache_path.open(
        mode, encoding="utf-8", newline=""
    ) as cache:
        writer = csv.DictWriter(cache, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(
                    f"Invalid JSON in spec text source at line {line_number}: {exc}"
                ) from exc
            file_id = record.get("file_url_sha16")
            if not isinstance(file_id, str) or not file_id:
                raise AnalysisError(
                    f"Missing file_url_sha16 in spec text source at line {line_number}."
                )
            content = record.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise AnalysisError(
                    f"Invalid spec content in JSONL record at line {line_number}."
                )
            content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            cached = completed.get(file_id)
            if cached and cached.get("content_sha256") == content_digest:
                continue
            result = heuristic_spec_scores(content)
            writer.writerow(
                {
                    "file_url_sha16": file_id,
                    **result,
                    "content_sha256": content_digest,
                    "scoring_method": "deterministic_regex_heuristic",
                    "cache_key": cache_key,
                }
            )
            if processed % 1000 == 0:
                cache.flush()
            completed[file_id] = {
                **{key: str(value) for key, value in result.items()},
                "content_sha256": content_digest,
                "cache_key": cache_key,
            }
            processed += 1
            if args.spec_score_limit and processed >= args.spec_score_limit:
                break
        cache.flush()

    score_frame = pd.read_csv(cache_path, dtype={"file_url_sha16": str})
    score_frame = score_frame.drop_duplicates("file_url_sha16", keep="last")
    base = pd.read_csv(
        dataset_path, low_memory=False, dtype={"file_url_sha16": str}
    )
    base = base.drop(
        columns=[name for name in SPEC_SCORE_TARGETS if name in base], errors="ignore"
    )
    merged = base.merge(
        score_frame[["file_url_sha16", *SPEC_SCORE_TARGETS]],
        on="file_url_sha16",
        how="left",
        validate="one_to_one",
    )
    merged.to_csv(dataset_path, index=False)
    write_rq2_dataset(dataset_path, required=True)
    write_spec_score_source("deterministic_regex_heuristic", cache_path)
    print(
        f"Heuristically scored {processed:,} new specs; cached scores for "
        f"{score_frame['file_url_sha16'].nunique():,} specs in {cache_path}"
    )
    return dataset_path

def run_spec_scores(args: argparse.Namespace) -> Path:
    import pandas as pd

    model = args.llm_model or os.getenv("LLM_MODEL")
    if not model:
        raise AnalysisError("Specify --llm-model for local LLM spec scoring.")
    endpoint = args.llm_endpoint or os.getenv(
        "LLM_BASE_URL", "http://127.0.0.1:11434/v1/chat/completions"
    )
    source_path = WORKSPACE_ROOT / "specmine-zenodo" / "specs.jsonl" / "specs.jsonl"
    dataset_path = OUTPUT_DIR / "modeling_dataset.csv"
    if not source_path.exists():
        raise AnalysisError(f"Spec text source is missing: {source_path}")
    if not dataset_path.exists():
        raise AnalysisError("Run the extract phase before scoring spec outcomes.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / "spec_scores.csv"
    cache_key = f"{model}|{SPEC_SCORE_PROMPT_VERSION}"
    completed: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("cache_key") != cache_key:
                    raise AnalysisError(
                        "Spec score cache model/prompt differs from this run. "
                        "Move output/spec_scores.csv aside to rescore."
                    )
                completed[row["file_url_sha16"]] = row

    processed = 0
    mode = "a" if cache_path.exists() else "w"
    with source_path.open(encoding="utf-8") as source, cache_path.open(
        mode, encoding="utf-8", newline=""
    ) as cache:
        fields = [
            "file_url_sha16",
            *SPEC_SCORE_TARGETS,
            "completeness_evidence",
            "quality_evidence",
            "content_sha256",
            "cache_key",
        ]
        writer = csv.DictWriter(cache, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(
                    f"Invalid JSON in spec text source at line {line_number}: {exc}"
                ) from exc
            file_id = record.get("file_url_sha16")
            content = record.get("content")
            if not isinstance(file_id, str) or not file_id:
                raise AnalysisError(
                    f"Missing file_url_sha16 in spec text source at line {line_number}."
                )
            if not isinstance(content, str) or not content.strip():
                continue
            content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            cached = completed.get(file_id)
            if cached and cached.get("content_sha256") == content_digest:
                continue
            result = score_spec_content(endpoint, model, content, args.llm_timeout)
            writer.writerow(
                {
                    "file_url_sha16": file_id,
                    **result,
                    "content_sha256": content_digest,
                    "cache_key": cache_key,
                }
            )
            cache.flush()
            completed[file_id] = {
                **{key: str(value) for key, value in result.items()},
                "content_sha256": content_digest,
                "cache_key": cache_key,
            }
            processed += 1
            if args.spec_score_limit and processed >= args.spec_score_limit:
                break

    cache_frame = pd.read_csv(cache_path, dtype={"file_url_sha16": str})
    cache_frame = cache_frame.drop_duplicates("file_url_sha16", keep="last")
    base = pd.read_csv(
        dataset_path, low_memory=False, dtype={"file_url_sha16": str}
    )
    base = base.drop(
        columns=[name for name in SPEC_SCORE_TARGETS if name in base], errors="ignore"
    )
    score_columns = ["file_url_sha16", *SPEC_SCORE_TARGETS]
    merged = base.merge(
        cache_frame[score_columns],
        on="file_url_sha16",
        how="left",
        validate="one_to_one",
    )
    merged.to_csv(dataset_path, index=False)
    write_rq2_dataset(dataset_path, required=True)
    write_spec_score_source("local_llm", cache_path)
    print(
        f"Scored {processed:,} new specs; cached scores for "
        f"{cache_frame['file_url_sha16'].nunique():,} specs in {cache_path}"
    )
    return dataset_path
