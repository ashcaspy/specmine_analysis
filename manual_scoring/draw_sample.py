"""Rebuild manual_scoring/sample_order.csv: the 40-spec manual scoring sample.

Rule:
- SpecMine v1.0 specs with lang = 'en' (spec_content_features).
- Two groups: spec_tool = 'openspec' and spec_tool <> 'openspec'.
- Within each group, order by md5(file_url_sha16 || 'ssw625') (DuckDB md5, lowercase hex
  string, ascending) and take the first 20.
- Interleave the groups, OpenSpec first.

EXCLUDED lists specs whose link no longer loaded when the sample was checked (2026-10-18).
Each one keeps its place in the list but is swapped for the next spec in its group's order
after the first 20.

Run from the project root:
    .venv/bin/python manual_scoring/draw_sample.py [output.csv]
"""
import csv
import sys

import duckdb

PER_GROUP = 20
EXCLUDED = {
    "971e3ad574a4c7a5": "repository no longer on GitHub",
}

QUERY = """
select f.file_url_sha16, f.file_url, f.spec_tool, f.repo_name, f.file_path
from 'data/spec_files.parquet' f
join 'data/spec_content_features.parquet' cf using (file_url_sha16)
where cf.lang = 'en' and {cond}
order by md5(f.file_url_sha16 || 'ssw625')
limit {limit}
"""
COLUMNS = ["file_url_sha16", "file_url", "spec_tool", "repo_name", "file_path"]


def draw(con, cond):
    rows = [dict(zip(COLUMNS, r)) for r in
            con.sql(QUERY.format(cond=cond, limit=PER_GROUP + len(EXCLUDED))).fetchall()]
    picked, spares = rows[:PER_GROUP], [r for r in rows[PER_GROUP:] if r["file_url_sha16"] not in EXCLUDED]
    for row in picked:
        row["note"] = ""
    for i, row in enumerate(picked):
        old = row["file_url_sha16"]
        if old in EXCLUDED:
            picked[i] = {**spares.pop(0), "note": f"replaces {old} ({EXCLUDED[old]})"}
    return picked


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "manual_scoring/sample_order.csv"
    con = duckdb.connect()
    openspec = draw(con, "f.spec_tool = 'openspec'")
    others = draw(con, "f.spec_tool <> 'openspec'")

    rows = [row for pair in zip(openspec, others) for row in pair]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["order", *COLUMNS, "note"])
        w.writeheader()
        for i, row in enumerate(rows, 1):
            w.writerow({"order": i, **row})
    print(f"wrote {len(rows)} specs to {out}")


if __name__ == "__main__":
    main()
