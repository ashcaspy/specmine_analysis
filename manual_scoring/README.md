# Manual Scoring Process

## Purpose

Check the computer-generated completeness and quality scores (0–5) by having a person score a sample of specs with the same rubric, then comparing the two. This is the manual validation step the professor asked for (Response #1). It is also a first step toward the independent quality assessment promised in the Phase 1 proposal.

## Process

### 1. Pick the sample

- Draw 30–40 specs at random from the computer-scored set.
- Ideally spread them across tools (OpenSpec, speckit, etc.), but with greater weight to OpenSpec as it represents the majority.
- Keep the computer's scores in a separate file, and don't open it until step 4.

### 2. Set up the scoring sheet

One row per spec, with these columns:

| Column | Contents |
|---|---|
| Spec ID | `file_url_sha16` |
| Link | link to the spec |
| Tool | e.g., openspec, speckit |
| Completeness | your score, 0–5 |
| Completeness reason | one line |
| Quality | your score, 0–5 |
| Quality reason | one line |
| Notes | e.g., "no reason for a Security section in a spec like this" |
| (Optional) Complete / Clear / Consistent / Testable | yes/no each: the Phase 1 proposal's quality concepts |

### 3. Score the specs

For each spec: read it, score it, write the reasons.

Use the Manual scoring rubric. Score what it says, even where it seems unfair for a particular spec, and note the problem in the Notes column. That is itself a finding about the rubric.

### 4. Compare with the computer's scores

Only now open the computer's scores. For each measure, record:

- How often you agree exactly
- How often you're within 1 point
- Which way the computer is usually off (too high or too low)
- 2–3 examples of clear mistakes, explained using the computer's recorded evidence for that spec

## Manual scoring rubric

### Both scores

Score only the spec file itself, as written. Don't assume missing sections or requirements.

Placeholder text counts as absent: "TBD", "TODO", unfilled template text, lorem ipsum.

0: empty, only placeholder text, or not actually a spec (e.g., an unrelated README).

### Completeness

Core functionality is what the feature or system is primarily designed to do: its main behaviors, inputs and outputs, and main flows. It's detailed when an engineer could start building from it, meaning specific behaviors rather than just goals.

Required sections. Each one counts if the content is in the spec anywhere, even without its own heading, and says at least one concrete thing. A heading with only "TBD" or "N/A" under it doesn't count. An explicit "not applicable because…" does count.

| Section | Counts if it describes… | Typical heading words |
|---|---|---|
| Architecture | the parts of the system and how they fit together | architecture, system/technical design, components |
| Error Handling | what happens when things fail: errors, retries, messages | error handling, failure handling, exceptions |
| Security | access control, permissions, data protection, privacy, input validation | security, authentication, authorization, privacy |
| Edge Cases | unusual or boundary situations: empty, maximum, duplicate, concurrent, failure | edge cases, boundary conditions, failure cases |
| Data Models | data structures, fields, schemas, entities | data model, schema, entities |

Actionable technical detail is anything an engineer could act on directly: named endpoints, fields or files, concrete behaviors, numbers, examples, acceptance criteria, code or schemas.

| Score | Rule |
|---:|---|
| 5 | Core functionality detailed, and all 5 sections covered |
| 4 | Core functionality detailed, and 3–4 sections covered |
| 3 | Core functionality detailed, and 0–2 sections covered |
| 2 | Some actionable detail, but core functionality isn't fully detailed (however many sections are covered) |
| 1 | Only a high-level summary or goals; nothing actionable |
| 0 | Empty or unusable |

### Quality

Requirement: a statement of something the system must do or a condition it must meet. That includes must/shall/should statements, acceptance criteria, given/when/then scenarios, and checklist items that describe behavior. It doesn't include background, goals, or implementation to-dos like "write tests" or "update docs".

Testable: you could write a pass/fail check for it, with a clear condition and an expected result. Numbers help but aren't required: "returns an error when the ID is missing" is testable. A given/when/then scenario is testable only if its "then" names a concrete result.

Vague: words like fast, quick, robust, appropriate, user-friendly, intuitive, seamless, scalable, efficient, reasonable, small, large, or "as needed", used without anything that pins them down. "Responds within 200 ms" isn't vague. Hedges like "etc.", "if possible" or "ideally" also count as vague.

Contradictory: two statements that can't both be true.

Poorly structured: the requirements are hard to find or follow, e.g., a wall of text with no headings or lists, or requirements mixed in with unrelated discussion.

| Score | Rule |
|---:|---|
| 5 | Nearly all requirements testable, no vague wording, clearly structured |
| 4 | Most requirements (about 3 in 4) testable, at most a couple of vague terms, clearly structured |
| 3 | You can tell what to build, but there's noticeable vague wording or only about half the requirements are testable |
| 2 | Few clear requirements; mostly vague or hard to test, but not contradictory |
| 1 | Contradictory, very poorly structured, or no identifiable requirements |
| 0 | Empty or unusable |

If a spec has no requirements by design (e.g., a pure design document), score it 1 and note the spec type in the Notes column.
