#!/usr/bin/env Rscript
# Omnibus tests for the "Refuse - counts" bullet in the qualitative writeup:
# does the rate of first-person capability refusals ("I can't ...", regex in
# analysis/regex_corrections.py) differ (1) by model family (GPT-3.5 vs
# GPT-4) and (2) by misconception label?
#
# Outcome is binary per conversation (refusal occurred / not), so the omnibus
# test is chi-square -- the ANOVA analog for a yes/no outcome. Label cells are
# small, so the label test uses a simulated p-value; conversations with more
# than one label contribute one row per label (same convention as
# fig_hw_by_code.R), with a single-label-only sensitivity check.
# If an omnibus test is significant, the highest- and lowest-rate categories
# are printed.
#
# Requires analysis/data/{conversations,corrections,conversation_ips}.csv.
#   Rscript analysis/correction_omnibus_tests.R
suppressPackageStartupMessages({library(dplyr); library(tidyr)})
set.seed(42)

conv <- read.csv("analysis/data/conversations.csv", stringsAsFactors = FALSE,
                 check.names = FALSE, na.strings = character(0)) %>%
  filter(in_corpus == 1, labels_final != "")
cor <- read.csv("analysis/data/corrections.csv", stringsAsFactors = FALSE) %>%
  select(conversation_hash, corrected = any_correction)
meta <- read.csv("analysis/data/conversation_ips.csv", stringsAsFactors = FALSE) %>%
  select(conversation_hash, model)
d <- conv %>% inner_join(cor, by = "conversation_hash") %>%
  inner_join(meta, by = "conversation_hash") %>%
  mutate(family = ifelse(grepl("^gpt-4", model), "GPT-4", "GPT-3.5"))
stopifnot(nrow(d) == nrow(conv))

cat(sprintf("Labeled conversations: %d; with refusal statement: %d (%.1f%%)\n\n",
            nrow(d), sum(d$corrected), 100 * mean(d$corrected)))

rate_tbl <- function(df, g) df %>% group_by(group = .data[[g]]) %>%
  summarise(n = n(), k = sum(corrected), pct = 100 * k / n, .groups = "drop") %>%
  arrange(desc(pct))

report_extremes <- function(tab, p) {
  if (p >= 0.05) { cat("  Not significant at 0.05; most/least not reported.\n"); return(invisible()) }
  hi <- tab[1, ]; lo <- tab[nrow(tab), ]
  cat(sprintf("  Most:  %s  %.1f%% (%d/%d)\n", hi$group, hi$pct, hi$k, hi$n))
  cat(sprintf("  Least: %s  %.1f%% (%d/%d)\n", lo$group, lo$pct, lo$k, lo$n))
}

## 1. Model family --------------------------------------------------------
cat("== Refusal rate by model family ==\n")
fam_tab <- rate_tbl(d, "family")
print(as.data.frame(fam_tab %>% mutate(pct = round(pct, 1))), row.names = FALSE)
fam_test <- chisq.test(table(d$family, d$corrected), correct = FALSE)
cat(sprintf("Omnibus chi-square: X2 = %.2f, df = %d, p = %.4g\n",
            fam_test$statistic, fam_test$parameter, fam_test$p.value))
report_extremes(fam_tab, fam_test$p.value)

## 2. Misconception label -------------------------------------------------
cat("\n== Refusal rate by misconception label ==\n")
dl <- d %>% separate_rows(labels_final, sep = ";\\s*") %>% rename(label = labels_final)
lab_tab <- rate_tbl(dl, "label")
print(as.data.frame(lab_tab %>% mutate(pct = round(pct, 1))), row.names = FALSE)
lab_test <- chisq.test(table(dl$label, dl$corrected),
                       simulate.p.value = TRUE, B = 1e5)
cat(sprintf("Omnibus chi-square (simulated p, B = 1e5): X2 = %.2f, p = %.4g\n",
            lab_test$statistic, lab_test$p.value))
report_extremes(lab_tab, lab_test$p.value)

# Sensitivity: single-label conversations only (independent units).
ds <- d %>% filter(n_labels_final == 1) %>% mutate(label = labels_final)
ss_test <- chisq.test(table(ds$label, ds$corrected),
                      simulate.p.value = TRUE, B = 1e5)
cat(sprintf("Sensitivity, single-label conversations only (n = %d): X2 = %.2f, p = %.4g\n",
            nrow(ds), ss_test$statistic, ss_test$p.value))

## Machine-readable exports --
write.csv(data.frame(
  key = c("n_labeled", "n_refusals",
          "fam_chisq", "fam_df", "fam_p",
          "label_chisq", "label_p", "label_B"),
  value = c(nrow(d), sum(d$corrected),
            unname(fam_test$statistic), unname(fam_test$parameter), fam_test$p.value,
            unname(lab_test$statistic), lab_test$p.value, 1e5)),
  "analysis/data/correction_omnibus_stats.csv", row.names = FALSE)
write.csv(bind_rows(fam_tab %>% mutate(kind = "family"),
                    lab_tab %>% mutate(kind = "label")),
          "analysis/data/refusal_rates.csv", row.names = FALSE)
cat("wrote analysis/data/correction_omnibus_stats.csv and refusal_rates.csv\n")
