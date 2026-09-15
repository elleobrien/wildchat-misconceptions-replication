# Replication package — WildChat misconceptions study

Materials for *User Misconceptions of LLM-Based Conversational Programming
Assistants* (O'Brien, Santos Alves, Baltes, Liebel, Lungu, Kalinowski; under
review at TOSEM).

No conversation text is redistributed. Every table is keyed on WildChat's
`conversation_hash`; transcripts come from the WildChat-1M release (see
*Input data*). Scripts marked *parquet* in `analysis/README.md` need that
rebuilt corpus; everything else runs on the files here.

## Layout

```
refined_codebook.csv          the codebook: per-code definition, examples, boundary cases, screening prompt
requirements.txt

data/
  download_wildchat.py        fetch allenai/WildChat-1M from Hugging Face
  filter_wildchat.py          -> data/wildchat_1m_english_python.parquet (the study corpus)
  validation_round_labels.csv preliminary-study validation round

pipeline/                     the LLM labeling stages
  system_prompt_screening.txt, prompts_screening/    stages 1-2 (screening), one prompt per code
  system_prompt_annotation.txt, prompts_annotation/  stage 3 (annotation), one prompt per code
  run_labeling_ollama.py      stage 1: gpt-oss:120b over every conversation x code
  run_labeling_targeted.py    stage 2: claude-sonnet-4-6 on stage-1 positives
  run_annotation_targeted.py  stage 3: claude-opus-4-7 on stage-2 positives
  _labeling_core.py, preview_prompts.py, keys.json.example
  results/                    production outputs of stages 2-3 (screening verdicts, final annotation
                              verdicts, model-vs-panel comparison), with .meta.json run records

analysis/                     labels, lexical measures, statistical tests, figures
  README.md                   what each script and table is
  data/conv_id_map.csv        paper IDs (C001-C963) -> conversation_hash
  data/labels_long.csv        every Opus verdict with reasoning and every human verdict with notes
  data/conversations.csv      one row per conversation: final labels, provenance, exclusion flags
  data/                       all other tables the scripts read and write
  figures/                    the figures as they appear in the paper
```

In `refined_codebook.csv`, the `LLM Filter Prompt` column is the screening
prompt and is deliberately broader than the definition, so that candidates are
over-retrieved for adjudication; the annotation-track prompts are built from
the other columns.

## Input data

WildChat-1M is not redistributed. Rebuild the study corpus (English
conversations containing a Python code fence):

```bash
pip install -r requirements.txt
python data/download_wildchat.py
python data/filter_wildchat.py
```

## Analysis

Scripts run from the repository root (`Rscript analysis/foo.R`,
`python analysis/foo.py`). Every table in `analysis/data/` is committed, so any
script can be run on its own; each one's header says what it reads and writes.
R packages: `dplyr`, `tidyr`, `ggplot2`, `lme4`.
