#!/usr/bin/env Rscript
# Code co-occurrence in the final human labels (in-corpus conversations):
#   (a) bar of conversations with 0 / 1 / 2 / 3+ codes
#   (b) heatmap of pairwise co-occurrence counts (diagonal = code total)
#   (c) table of the exact label combinations, most common first
# Saves analysis/figures/code_cooccurrence.{png,pdf} and
#       analysis/data/code_combinations.csv, analysis/data/code_pair_counts.csv
#
#   Rscript analysis/fig_code_cooccurrence.R
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2); library(patchwork)})

conv <- read.csv("analysis/data/conversations.csv", stringsAsFactors = FALSE,
                 check.names = FALSE, na.strings = character(0)) %>% filter(in_corpus == 1)
n_conv <- nrow(conv)

long <- conv %>% filter(labels_final != "") %>%
  select(conversation_hash, labels_final) %>%
  separate_rows(labels_final, sep = ";\\s*") %>% rename(code = labels_final) %>% distinct()

code_order <- long %>% count(code, sort = TRUE) %>% pull(code)

## (a) number of codes per conversation -----------------------------------
per_conv <- long %>% count(conversation_hash, name = "k") %>%
  right_join(conv %>% select(conversation_hash), by = "conversation_hash") %>%
  mutate(k = replace_na(k, 0L), k_bin = factor(ifelse(k >= 3, "3+", as.character(k)),
                                                levels = c("0", "1", "2", "3+")))
k_counts <- per_conv %>% count(k_bin, .drop = FALSE) %>% mutate(pct = 100 * n / n_conv)

p_k <- ggplot(k_counts, aes(k_bin, n)) +
  geom_col(fill = "#4C6A92", width = 0.7) +
  geom_text(aes(label = ifelse(n > 0 & round(pct) == 0, sprintf("%d\n(<1%%)", n), sprintf("%d\n(%.0f%%)", n, pct))), vjust = -0.2, size = 3.1, lineheight = 0.9) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(x = "Codes per conversation", y = "Conversations", title = "(a) Codes per conversation") +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank())

## (b) pairwise co-occurrence ---------------------------------------------
m <- long %>% mutate(v = 1L) %>%
  pivot_wider(names_from = code, values_from = v, values_fill = 0L) %>%
  select(all_of(code_order)) %>% as.matrix()
co <- crossprod(m)                       # diagonal = per-code n, off-diag = shared convs
pairs <- as.data.frame(as.table(co)) %>% setNames(c("a", "b", "n")) %>%
  mutate(a = factor(a, levels = code_order), b = factor(b, levels = rev(code_order)),
         is_diag = a == b)
write.csv(pairs %>% filter(!is_diag, as.integer(a) < as.integer(factor(b, levels = code_order))) %>%
            select(code_a = a, code_b = b, n_shared = n) %>% arrange(desc(n_shared)),
          "analysis/data/code_pair_counts.csv", row.names = FALSE)

p_heat <- ggplot(pairs, aes(a, b)) +
  geom_tile(aes(fill = ifelse(is_diag, NA, n)), colour = "white") +
  geom_text(aes(label = n, fontface = ifelse(is_diag, "bold", "plain"),
                colour = ifelse(!is_diag & n > max(n[!is_diag]) * 0.6, "white", "grey15")), size = 3) +
  scale_colour_identity() +
  scale_fill_gradient(low = "#F3F5F8", high = "#4C6A92", na.value = "grey88", name = "Shared\nconvs") +
  coord_fixed() +
  labs(x = NULL, y = NULL, title = "(b) Pairwise co-occurrence (diagonal = code total)") +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 35, hjust = 1),
        legend.position = "right")

p <- p_k + p_heat + plot_layout(widths = c(1, 2))
ggsave("analysis/figures/code_cooccurrence.png", p, width = 11, height = 5, dpi = 200, bg = "white")
ggsave("analysis/figures/code_cooccurrence.pdf", p, width = 11, height = 5)

## (c) exact combinations --------------------------------------------------
combos <- long %>% group_by(conversation_hash) %>%
  summarise(combo = paste(sort(code), collapse = " + "), k = n(), .groups = "drop") %>%
  count(combo, k, sort = TRUE, name = "n") %>% mutate(pct_of_labeled = 100 * n / sum(n))
write.csv(combos, "analysis/data/code_combinations.csv", row.names = FALSE)

cat("Codes per conversation (n =", n_conv, "in-corpus):\n"); print(as.data.frame(k_counts), row.names = FALSE)
cat("\nMulti-code combinations:\n"); print(as.data.frame(combos %>% filter(k > 1)), row.names = FALSE)
cat("\nTop pairs:\n"); print(read.csv("analysis/data/code_pair_counts.csv") %>% head(8), row.names = FALSE)
