#!/usr/bin/env Rscript
# Human-vs-Opus agreement on the final in-corpus conversations.
#
# Same method as kappa_verify_targeted.R: each conversation x code cell is a
# 0/1 "code present" judgment; Cohen's kappa is computed on the pooled
# conversation x code grid (N_conv * 8 cells), plus per-code kappa.
# Human = labels_final from analysis/data/conversations.csv (Elle's labels,
# except round-1 majority vote on the 30 round-1 conversations, and codebook
# anchors). Opus = labels_opus (verdict=1 labels from the iarev1 run).
#
#   Rscript analysis/kappa_human_opus.R
suppressPackageStartupMessages(library(dplyr))

CONV <- "analysis/data/conversations.csv"
CODES <- c("Algebra", "Clear session", "Code Execution", "Continuous training",
           "Internet Access", "Local machine access", "Non-text output", "Session memory")

conv <- read.csv(CONV, stringsAsFactors = FALSE, check.names = FALSE, na.strings = character(0))
conv <- conv[conv$in_corpus == 1, ]

presence <- function(cells) {
  cells[is.na(cells)] <- ""
  m <- sapply(CODES, function(code) as.integer(grepl(code, cells, fixed = TRUE)))
  matrix(m, nrow = length(cells), dimnames = list(conv$conversation_hash, CODES))
}
H <- presence(conv$labels_final)
O <- presence(conv$labels_opus)

cohen_kappa <- function(x, y) {
  x <- as.integer(x); y <- as.integer(y); n <- length(x)
  tab <- table(factor(x, 0:1), factor(y, 0:1))
  po <- sum(diag(tab)) / n
  pe <- sum(rowSums(tab) * colSums(tab)) / n^2
  c(kappa = (po - pe) / (1 - pe), agreement = po, n = n,
    both = tab[2, 2], human_only = tab[2, 1], opus_only = tab[1, 2], neither = tab[1, 1])
}

cat(sprintf("conversations: %d   codes: %d   pooled cells: %d\n\n",
            nrow(conv), length(CODES), nrow(conv) * length(CODES)))
pooled <- cohen_kappa(H, O)
cat(sprintf("Pooled Cohen's kappa (human vs Opus) = %.3f   raw agreement = %.3f\n",
            pooled["kappa"], pooled["agreement"]))
cat(sprintf("  both=%d  human-only=%d  opus-only=%d  neither=%d\n\n",
            pooled["both"], pooled["human_only"], pooled["opus_only"], pooled["neither"]))

per_code <- t(sapply(CODES, function(k) cohen_kappa(H[, k], O[, k])))
per_code <- as.data.frame(per_code) %>%
  mutate(code = CODES, n_human = both + human_only, n_opus = both + opus_only,
         precision_opus = both / n_opus, recall_opus = both / n_human) %>%
  select(code, n_human, n_opus, both, human_only, opus_only, kappa, agreement, precision_opus, recall_opus)
cat("Per-code (Opus as classifier, human as reference):\n")
print(per_code %>% mutate(across(where(is.numeric), ~ round(.x, 3))), row.names = FALSE)

out <- bind_rows(
  per_code,
  data.frame(code = "POOLED", n_human = sum(H), n_opus = sum(O), both = pooled["both"],
             human_only = pooled["human_only"], opus_only = pooled["opus_only"],
             kappa = pooled["kappa"], agreement = pooled["agreement"],
             precision_opus = pooled["both"] / sum(O), recall_opus = pooled["both"] / sum(H)))
write.csv(out, "analysis/data/kappa_human_opus.csv", row.names = FALSE)
cat("\nwrote analysis/data/kappa_human_opus.csv\n")
