# Claude Working Notes for this Repository

This file documents conventions Claude should follow when working in this repo. Especially around what gets committed and pushed to the public GitHub remote, since the course professor will see the repository.

## Git Push Policy

**Goal:** the public repo should contain only what a grader or external reader would benefit from. Internal team scratch notes, presentation transcripts, and machine-specific cache files do not belong on the remote.

### Always push (core deliverables)

| File | Reason |
|---|---|
| `Part2_Models_and_Results.ipynb` | Part 2 analysis notebook (sections 12 Binary, 13 Reframing) |
| `Part2_Report/CSE40467 - Project Part 2 Report.docx` | Final Part 2 report |
| `Part2_Report/generate_final_report.py` | Reproducible report-generation script |
| `Part2_Report/CSE40467 - Project Part 2 Report Template.docx` | Template (already tracked) |
| `Part2_Report/PP2 Rubric.pdf` | Rubric reference (already tracked) |
| `Data Science Report.ipynb` | Part 1 EDA notebook |
| `docs/project_part2.md` | Markdown source of the Part 2 report |
| `docs/DS Project Presentation.pdf` | Presentation slides |
| `docs/code_analysis.md`, `docs/summary.md`, `docs/expanded_training_results.md`, `docs/feature_experiment_results.md`, `docs/worldcup_subset_codebook.csv` | Project documentation |
| `scripts/clean_worldcup.py`, `scripts/feature_engineering.py`, `scripts/feature_engineering_expanded.py` | Data pipeline |
| `scripts/binary_no_draw_model.py` | Comment B response code |
| `scripts/test_set_breakdown.py` | Comment C response code |
| `data_clean/features_train.csv`, `data_clean/features_test.csv`, `data_clean/matches_train.csv`, `data_clean/matches_test.csv`, `data_clean/features_expanded_test.csv`, `data_clean/fifa_rankings.csv`, `data_clean/international_results.csv`, `data_clean/statsbomb_wc_stats.csv`, `data_clean/wc_squads.csv` | Cleaned data |
| `data_clean/binary_no_draw_summary.csv`, `data_clean/test_predictions_2022_grouped.csv` | Result CSVs the notebook loads |
| `files_needed/matches.csv`, `files_needed/tournaments.csv` | Raw data |
| `figures/01_*` through `figures/11_*` | Original visualizations |
| `figures/12_test_grouped_metrics.png`, `figures/13_favorite_vs_upset_cm.png`, `figures/14_knockout_betting_card.png` | Comment C figures |
| `figures/binary_no_draw_test_metrics.png`, `figures/binary_no_draw_roc.png`, `figures/binary_no_draw_knockout_confusion.png` | Comment B figures |
| `README.md`, `requirements.txt`, `assignment_desc.txt` | Repo metadata |

### Never push (internal team notes)

These belong in `.gitignore`. Reason for each:

| File | Reason |
|---|---|
| `docs/current.md` | Team work-split memo with personal context (e.g., "Drew left for grad photos") |
| `docs/plan.md` | Post-presentation action plan and accuracy-drift internal diagnostic. Awkward for the professor to see their own comments labeled "Comment A/B/C" with our internal triage |
| `docs/presentation_and_comments.txt` | Verbatim presentation transcript with the professor's casual remarks ("Bro, you are a sport guy"). Highly awkward if the professor finds their own transcript on our repo |
| `docs/draft_CSE40467 - Project Part 2 Report.pdf` | Outdated draft of the report. Confusing alongside the final docx |
| `scripts/fix_model_comparison_figure.py` | Slide-figure script that hardcodes the published 0.625 number rather than recomputing from data. Invites the question "why hardcode instead of re-running the model?" |

### Never commit (machine-generated cache)

Always ignore. These have no business in version control:

| Path | What it is |
|---|---|
| `.cache/` | matplotlib/fontconfig cache, machine-specific, auto-regenerated |
| `.mplconfig/` | matplotlib font list cache, machine-specific, auto-regenerated |
| `__pycache__/`, `*.pyc` | Python bytecode |
| `.DS_Store` | macOS folder metadata |

These caches end up in the repo because `scripts/binary_no_draw_model.py` and `scripts/test_set_breakdown.py` redirect them into the project tree via `MPLCONFIGDIR` and `XDG_CACHE_HOME`. The `.gitignore` rules above keep them out of git anyway.

## Pre-push checklist

Before any `git push` to the public remote, run through this:

1. **No Korean text** in code, notebooks, markdown, or scripts that get pushed (Korean summaries are personal notes, not deliverables). A quick check: `grep -rP '[\x{AC00}-\x{D7A3}]' --include='*.py' --include='*.md' --include='*.ipynb' .`
2. **No machine-specific caches** in `git status`. If `.cache/` or `.mplconfig/` shows up, remove with `git rm -r --cached <path>` and confirm the `.gitignore` covers them.
3. **Internal docs are gitignored**, not staged. Confirm `docs/current.md`, `docs/plan.md`, `docs/presentation_and_comments.txt` are not in the commit.
4. **No em-dashes (—) or en-dashes (–)** in any text the team did not personally write that style way; the professor prefers a plain-prose style for this course.
5. **Notebook outputs**: keep them when the figures or numbers are part of the grading evidence; clear them when they only contain stack traces or noisy progress bars.

## Style conventions for written content

- No em-dashes (`—`) or en-dashes (`–`). Use commas, parentheses, colons, or sentence breaks instead.
- Aim for first-year-PhD-student prose: specific numbers, hedged claims where appropriate, mix of prose and tables, no AI-style filler ("Furthermore," "Moreover," "It is worth noting that...").
- Always cite when claiming results from prior literature.
- Honest about limitations: small test set (n=64), seed sensitivity, etc.

## Reproducibility commands (for the README and report)

```bash
pip3 install -r requirements.txt
python3 scripts/clean_worldcup.py
python3 scripts/feature_engineering.py
python3 scripts/binary_no_draw_model.py        # Comment B variant
python3 scripts/test_set_breakdown.py          # Comment C breakdown
# Then run Part2_Models_and_Results.ipynb
```
