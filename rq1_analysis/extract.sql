WITH link_features AS (
    SELECT
        spec_url_sha16,
        SUM(rel = 'code') AS links_code,
        SUM(rel = 'pr') AS links_pr,
        SUM(rel = 'issue') AS links_issue,
        SUM(rel = 'sibling') AS links_sibling,
        SUM(rel = 'branch') AS links_branch,
        SUM(rel = 'ref') AS links_ref,
        SUM(rel = 'anchor') AS links_anchor
    FROM spec_links
    GROUP BY spec_url_sha16
),
matched_prs AS (
    SELECT
        sf.file_url_sha16,
        p.tool,
        p.repo_name,
        p.pr_number,
        MAX(p.state) AS state,
        MAX(p.is_merged) AS is_merged,
        MAX(p.merged_at) AS merged_at,
        MAX(p.created_at) AS created_at
    FROM spec_files sf
    JOIN pr_files pf
      ON pf.repo_name = sf.repo_name
     AND pf.file_path = sf.file_path
    JOIN pull_requests p
      ON p.tool = pf.tool
     AND p.repo_name = pf.repo_name
     AND p.pr_number = pf.pr_number
    GROUP BY sf.file_url_sha16, p.tool, p.repo_name, p.pr_number
),
pr_features AS (
    SELECT
        file_url_sha16,
        COUNT(*) AS associated_pr_count,
        SUM(CASE WHEN state = 'closed' THEN 1 ELSE 0 END)
          AS target_pr_closed_count,
        SUM(CASE WHEN state = 'closed' AND is_merged = 1 THEN 1 ELSE 0 END)
          AS target_pr_closed_merged_count,
        SUM(CASE WHEN state = 'closed' AND is_merged = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN state = 'closed' THEN 1 ELSE 0 END), 0)
          AS target_pr_closed_merge_proportion,
        MAX(CASE WHEN is_merged = 1 THEN 1 ELSE 0 END)
          AS target_pr_any_merged,
        AVG(
            CASE WHEN is_merged = 1 AND merged_at IS NOT NULL AND created_at IS NOT NULL
                 THEN TIMESTAMPDIFF(
                     SECOND,
                     STR_TO_DATE(LEFT(created_at, 19), '%Y-%m-%dT%H:%i:%s'),
                     STR_TO_DATE(LEFT(merged_at, 19), '%Y-%m-%dT%H:%i:%s')
                 ) / 3600.0
            END
        ) AS target_avg_pr_merge_hours
    FROM matched_prs
    GROUP BY file_url_sha16
),
artifact_scope AS (
    SELECT
        af.repo_name,
        af.file_path,
        af.kind,
        af.n_tasks,
        af.n_tasks_done,
        CASE
          WHEN SUBSTRING_INDEX(SUBSTRING_INDEX(af.file_path, '/changes/', -1), '/', 1) = 'archive'
          THEN CONCAT(
            SUBSTRING_INDEX(af.file_path, '/changes/', 1), '/changes/',
            SUBSTRING_INDEX(SUBSTRING_INDEX(af.file_path, '/changes/', -1), '/', 2)
          )
          ELSE CONCAT(
            SUBSTRING_INDEX(af.file_path, '/changes/', 1), '/changes/',
            SUBSTRING_INDEX(SUBSTRING_INDEX(af.file_path, '/changes/', -1), '/', 1)
          )
        END AS change_root
    FROM openspec_artifact_files af
    WHERE LOCATE('/changes/', af.file_path) > 0
),
refs_by_artifact AS (
    SELECT
        repo_name,
        artifact_path,
        COUNT(DISTINCT CASE WHEN task_done IN (0, 1) THEN id END) AS refs_with_task_status,
        COUNT(DISTINCT CASE WHEN task_done = 1 THEN id END) AS refs_done
    FROM openspec_code_refs
    GROUP BY repo_name, artifact_path
),
artifact_metrics AS (
    SELECT
        a.repo_name,
        a.file_path,
        a.change_root,
        a.kind,
        a.n_tasks,
        a.n_tasks_done,
        CASE WHEN a.n_tasks IS NOT NULL AND a.n_tasks_done IS NOT NULL
             THEN GREATEST(a.n_tasks - a.n_tasks_done, 0)
        END AS n_tasks_remaining,
        CASE WHEN a.n_tasks > 0
             THEN a.n_tasks_done / NULLIF(a.n_tasks, 0)
        END AS completion_proportion,
        CASE WHEN a.n_tasks > a.n_tasks_done
             THEN a.n_tasks_done / NULLIF(a.n_tasks - a.n_tasks_done, 0)
        END AS done_incomplete_ratio,
        CASE WHEN r.refs_with_task_status > 0
             THEN r.refs_done / r.refs_with_task_status
        END AS code_ref_task_done_proportion
    FROM artifact_scope a
    LEFT JOIN refs_by_artifact r
      ON r.repo_name = a.repo_name
     AND r.artifact_path = a.file_path
),
openspec_features AS (
    SELECT
        sf.file_url_sha16,
        SUM(a.n_tasks_done) AS target_all_tasks_done_count,
        SUM(a.n_tasks_remaining) AS target_all_tasks_remaining_count,
        AVG(a.done_incomplete_ratio)
          AS target_all_tasks_done_incomplete_ratio_avg,
        AVG(a.completion_proportion) AS target_all_tasks_completion_proportion_avg,
        AVG(a.code_ref_task_done_proportion) AS target_code_refs_task_done_proportion_avg,
        SUM(CASE WHEN a.kind = 'tasks.md' THEN a.n_tasks_done END)
          AS target_tasks_md_done_count,
        SUM(CASE WHEN a.kind = 'tasks.md' THEN a.n_tasks_remaining END)
          AS target_tasks_md_remaining_count,
        AVG(CASE WHEN a.kind = 'tasks.md' THEN a.done_incomplete_ratio END)
          AS target_tasks_md_done_incomplete_ratio_avg
    FROM spec_files sf
    LEFT JOIN artifact_metrics a
      ON a.repo_name = sf.repo_name
     AND a.change_root = SUBSTRING_INDEX(sf.file_path, '/specs/', 1)
     AND LOCATE('/changes/', sf.file_path) > 0
    GROUP BY sf.file_url_sha16
)
SELECT
    sf.file_url_sha16,
    sf.repo_name,
    {FEATURE_SELECT},
    COALESCE(p.associated_pr_count, 0) AS associated_pr_count,
    p.target_pr_closed_count,
    p.target_pr_closed_merged_count,
    p.target_pr_closed_merge_proportion,
    COALESCE(p.target_pr_any_merged, 0) AS target_pr_any_merged,
    p.target_avg_pr_merge_hours,
    COALESCE(o.target_all_tasks_done_count, 0) AS target_all_tasks_done_count,
    COALESCE(o.target_all_tasks_remaining_count, 0) AS target_all_tasks_remaining_count,
    o.target_all_tasks_done_incomplete_ratio_avg,
    o.target_all_tasks_completion_proportion_avg,
    o.target_code_refs_task_done_proportion_avg,
    COALESCE(o.target_tasks_md_done_count, 0) AS target_tasks_md_done_count,
    COALESCE(o.target_tasks_md_remaining_count, 0) AS target_tasks_md_remaining_count,
    o.target_tasks_md_done_incomplete_ratio_avg
FROM spec_files sf
LEFT JOIN spec_content_features cf
  ON cf.file_url_sha16 = sf.file_url_sha16
LEFT JOIN link_features l
  ON l.spec_url_sha16 = sf.file_url_sha16
LEFT JOIN pr_features p
  ON p.file_url_sha16 = sf.file_url_sha16
LEFT JOIN openspec_features o
  ON o.file_url_sha16 = sf.file_url_sha16
ORDER BY sf.file_url_sha16
