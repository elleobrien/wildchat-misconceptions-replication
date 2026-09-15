# analysis/

Scripts run from the repository root. *parquet* marks scripts that need the
rebuilt `data/wildchat_1m_english_python.parquet`.

## Labels

`data/labels_long.csv` has one row per (conversation, code, source, annotator).
`source` is `opus` (the claude-opus-4-7 annotation run), `verify` (human review
of the Opus labels), `round1_majority` (majority vote of the three-annotator
round on 30 conversations) or `anchor` (codebook anchor conversation).
`decision` on verify rows is `accept` / `reject` of the Opus label, or `extra`
for a code the human added. `note` is Opus's reasoning or the annotator's
comment, verbatim. A conversation can have several human rows per code (Elle's
full pass plus reviewer checks by Grischa, Sebastian and Antonio); all are kept.

`data/conversations.csv` has one row per conversation that entered the
pipeline. `labels_final` is the label set analysed in the paper and
`final_source` where it came from (`round1_majority` > `elle` > `anchor`);
`in_corpus = 1` marks the 754 analysed conversations.

`data/conv_id_map.csv` maps paper IDs to hashes. C954–C963 are
preliminary-study conversations quoted in the paper that never entered the
pipeline.

Other tables: `conversation_ips.csv` (hashed IP, model, timestamp per
conversation), `ip_activity.csv` (conversations per hashed IP over the corpus),
`first_turn_clusters.csv` (same-IP first-turn repeat clusters collapsed during
verification), `processing_stats.csv` (the processing funnel),
`kappa_verify_reviewers.csv` (peer-review agreement on the verification queue).

## Lexical measures

| module | measure | output |
|---|---|---|
| `regex_homework.py` | possibly pasted coursework | `data/homework_regex.csv` (*parquet*) |
| `regex_corrections.py` | assistant corrections / refusals | `data/corrections.csv` (*parquet*) |
| `regex_scraping.py` | web-scraping requests | `data/scraping_regex.csv` (*parquet*) |
| `regex_python_intro.py` | Python-course phrasing | `data/python_intro_regex.csv` (*parquet*) |

The patterns are the constants at the top of each module. The `evidence` /
`snippet` columns hold only the matched span.

## Tests and figures

| script | what | outputs |
|---|---|---|
| `kappa_human_opus.R` | human-vs-Opus agreement on the final corpus | `data/kappa_human_opus.csv` |
| `correction_omnibus_tests.R` | chi-square tests of refusal rate by model family and by code | `data/correction_omnibus_stats.csv`, `refusal_rates.csv` |
| `code_recurrence_by_ip.R` | within-IP recurrence per code, omnibus chi-square; main-text figure | `data/code_recurrence_by_ip.csv`, `code_recurrence_stats.csv`, `figures/code_recurrence_by_ip.*` |
| `sensitivity_clustering.R` | refusal tests refit with an IP random intercept and on single-label conversations; recurrence adjusted for IP activity | `data/sensitivity_stats.csv`, `sensitivity_activity_by_code.csv` |
| `fig_code_cooccurrence.R` | codes per conversation and pairwise co-occurrence | `figures/code_cooccurrence.*`, `data/code_combinations.csv`, `code_pair_counts.csv` |
| `fig_hw_by_code.R` | coursework-flag rate by code | `figures/hw_by_code.*`, `data/hw_by_code.csv`, `hw_by_code_stats.csv` |
