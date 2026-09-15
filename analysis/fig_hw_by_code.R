#!/usr/bin/env Rscript
# Share of labeled conversations flagged "possibly pasted coursework" (regex category A,
# see analysis/regex_homework.py) by misconception code. Error bars = Wilson 95% intervals of the
# proportion; significance reported only as an omnibus chi-square across codes.
# Codes overlap (multi-label), so a conversation can appear under several codes;
# the omnibus test treats code membership as independent samples — rough screen only.
# Requires analysis/data/homework_regex.csv (python3 analysis/regex_homework.py --input <labeled hashes>).
# Saves analysis/figures/hw_by_code.{png,pdf}, analysis/data/hw_by_code.csv
#
#   Rscript analysis/fig_hw_by_code.R
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2)})
set.seed(42)  # simulated omnibus p must be reproducible; exported for the manuscript

conv <- read.csv("analysis/data/conversations.csv", stringsAsFactors = FALSE,
                 check.names = FALSE, na.strings = character(0)) %>% filter(in_corpus == 1, labels_final != "")
hw <- read.csv("analysis/data/homework_regex.csv", stringsAsFactors = FALSE) %>%
  transmute(conversation_hash, hw = as.integer(hw_flag == 1))
stopifnot(all(conv$conversation_hash %in% hw$conversation_hash))
d <- conv %>% inner_join(hw, by = "conversation_hash") %>%
  separate_rows(labels_final, sep = ";\\s*") %>% rename(code = labels_final)
uni <- d %>% distinct(conversation_hash, hw)

tot <- d %>% group_by(code) %>%
  summarise(n = n(), k = sum(hw), .groups = "drop") %>%
  bind_rows(uni %>% summarise(code = "All labeled conversations", n = n(), k = sum(hw))) %>%
  mutate(pct = 100 * k / n,
         # Wilson 95% interval
         z = qnorm(0.975), pp = k / n, dd = 1 + z^2 / n,
         c0 = (pp + z^2 / (2 * n)) / dd, hh = z * sqrt(pp * (1 - pp) / n + z^2 / (4 * n^2)) / dd,
         lo = pmax(0, 100 * (c0 - hh)), hi = 100 * (c0 + hh)) %>%
  dplyr::select(-z, -pp, -dd, -c0, -hh) %>%
  arrange(code == "All labeled conversations", pct) %>% mutate(code = factor(code, levels = code))
write.csv(tot, "analysis/data/hw_by_code.csv", row.names = FALSE)

omni <- chisq.test(as.matrix(d %>% count(code, hw) %>% pivot_wider(names_from = hw, values_from = n, values_fill = 0) %>%
                               select(-code)), simulate.p.value = TRUE, B = 1e5)
write.csv(data.frame(key = c("hw_omnibus_chisq", "hw_omnibus_p", "hw_omnibus_B"),
                     value = c(unname(omni$statistic), omni$p.value, 1e5)),
          "analysis/data/hw_by_code_stats.csv", row.names = FALSE)

p <- ggplot(tot, aes(x = pct, y = code)) +
  geom_col(width = 0.7, fill = "#4C6A92") +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.25, colour = "grey30", orientation = "y") +
  geom_text(aes(x = hi, label = sprintf("%.0f%%  (%d/%d)", pct, k, n)),
            hjust = -0.12, size = 3.1, colour = "grey20") +
  scale_x_continuous(limits = c(0, max(tot$hi) * 1.35), expand = expansion(mult = c(0, 0.02)),
                     labels = function(x) paste0(x, "%")) +
  labs(x = "Conversations flagged as possibly pasted coursework", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        plot.title.position = "plot",
        axis.text.y = element_text(colour = "grey10",
                                   face = ifelse(levels(tot$code) == "All labeled conversations", "bold", "plain")))
ggsave("analysis/figures/hw_by_code.png", p, width = 8.5, height = 4.4, dpi = 200, bg = "white")
ggsave("analysis/figures/hw_by_code.pdf", p, width = 8.5, height = 4.4)

cat(sprintf("Omnibus chi-square across codes (simulated p, B=1e5): X2 = %.2f, p = %.4f\n", omni$statistic, omni$p.value))
print(tot %>% arrange(desc(pct)) %>% select(code, n, k, pct, lo, hi) %>%
        mutate(across(c(pct, lo, hi), \(x) round(x, 1))) %>%
        as.data.frame(), row.names = FALSE)
