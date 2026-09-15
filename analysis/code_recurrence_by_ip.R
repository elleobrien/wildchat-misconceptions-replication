#!/usr/bin/env Rscript
# How common is it for a misconception code to recur within the same hashed IP?
#
# For each code we report:
#   n_conv        conversations carrying the code
#   n_ip          distinct IPs with >=1 such conversation
#   n_ip_recur    IPs with >=2 distinct conversations carrying the code
#   pct_ip_recur  n_ip_recur / n_ip
#   pct_conv_from_recur  share of the code's conversations that come from
#                        recurring IPs (how much volume is repeat behavior)
# Also reports, over the full hand-reviewed corpus (in_corpus == 1): the
# number of unique hashed IPs and the median/min/max conversations per IP.
#
# Inputs:  analysis/data/conversations.csv      (labels_final, in_corpus)
#          analysis/data/conversation_ips.csv   (conversation_ips.py)
# Output:  analysis/data/code_recurrence_by_ip.csv
#          analysis/figures/code_recurrence_by_ip.{png,pdf}
#
#   Rscript analysis/code_recurrence_by_ip.R

suppressPackageStartupMessages(library(ggplot2))

set.seed(1)

corpus <- read.csv("analysis/data/conversations.csv", stringsAsFactors = FALSE,
                   check.names = FALSE, na.strings = character(0))
corpus <- corpus[corpus$in_corpus == 1, ]

ips <- read.csv("analysis/data/conversation_ips.csv", stringsAsFactors = FALSE)
corpus$hashed_ip <- ips$hashed_ip[match(corpus$conversation_hash, ips$conversation_hash)]
stopifnot(!anyNA(corpus$hashed_ip))

# Conversations-per-IP over the full hand-reviewed corpus.
per_ip_convs <- table(corpus$hashed_ip)
cat(sprintf("Hand-reviewed corpus: %d conversations from %d unique hashed IPs\n",
            nrow(corpus), length(per_ip_convs)))
cat(sprintf("Conversations per IP: median %g, min %d, max %d\n\n",
            median(per_ip_convs), min(per_ip_convs), max(per_ip_convs)))

conv <- corpus[corpus$labels_final != "", ]

# Long format: one row per (conversation, code); multi-label split on ";".
label_sets <- strsplit(conv$labels_final, ";\\s*")
long <- data.frame(hashed_ip = rep(conv$hashed_ip, lengths(label_sets)),
                   code      = trimws(unlist(label_sets)),
                   stringsAsFactors = FALSE)

# Per-code recurrence summary from a long (ip, code) table.
summarize_recurrence <- function(long) {
  per_ip <- aggregate(list(n = rep(1L, nrow(long))), long[c("code", "hashed_ip")], sum)
  do.call(rbind, lapply(split(per_ip, per_ip$code), function(d) data.frame(
    code                = d$code[[1]],
    n_conv              = sum(d$n),
    n_ip                = nrow(d),
    n_ip_recur          = sum(d$n >= 2),
    pct_ip_recur        = 100 * mean(d$n >= 2),
    pct_conv_from_recur = 100 * sum(d$n[d$n >= 2]) / sum(d$n),
    stringsAsFactors = FALSE)))
}

obs <- summarize_recurrence(long)
obs <- obs[order(-obs$n_ip_recur, -obs$n_conv), ]
write.csv(obs, "analysis/data/code_recurrence_by_ip.csv", row.names = FALSE)

cat(sprintf("%d conversations, %d IPs, %d IPs with >=2 labeled conversations\n\n",
            nrow(conv), length(unique(conv$hashed_ip)),
            sum(table(conv$hashed_ip) >= 2)))
print(obs, row.names = FALSE, digits = 3)

# Omnibus test: does the recurrence rate differ by code? Unit is the
# (IP, code) pair, outcome is binary (recurred / not), so the omnibus test is
# chi-square -- the ANOVA analog for a yes/no outcome (same convention as
# correction_omnibus_tests.R). Codes seen from fewer than 10 IPs are
# excluded: their zero-recurrence cells are uninformative (can't distinguish
# "does not recur" from "too rare to observe"). Zero cells remain among the
# kept codes, so the p-value is simulated rather than asymptotic.
MIN_IPS <- 10
kept <- obs$code[obs$n_ip >= MIN_IPS]
per_ip <- aggregate(list(n = rep(1L, nrow(long))), long[c("code", "hashed_ip")], sum)
per_ip <- per_ip[per_ip$code %in% kept, ]
per_ip$recurred <- per_ip$n >= 2
omni <- chisq.test(table(per_ip$code, per_ip$recurred),
                   simulate.p.value = TRUE, B = 10000)
cat(sprintf("\nOmnibus recurrence-by-code (%d codes with >=%d IPs): chi-square = %.2f, simulated p = %.4f (B = 10000)\n",
            length(kept), MIN_IPS, omni$statistic, omni$p.value))

# Corpus-level and omnibus stats.
write.csv(data.frame(
  key   = c("corpus_convs", "corpus_ips", "labeled_ips", "min_ips",
            "omnibus_chisq", "omnibus_p"),
  value = c(nrow(corpus), length(per_ip_convs), length(unique(conv$hashed_ip)),
            MIN_IPS, round(unname(omni$statistic), 2), round(omni$p.value, 3))),
  "analysis/data/code_recurrence_stats.csv", row.names = FALSE)

# Figure: recurrence rate by code (style matches fig_hw_by_code.R).
# Unit is the IP: of the IPs that produced a code at all, the share that
# produced it in >=2 distinct conversations. Error bars = Wilson 95% intervals
# of the proportion.
fig <- obs[obs$n_ip >= MIN_IPS, ]
# Wilson 95% interval (well-defined at 0/n, unlike the normal-approximation SE)
wilson <- function(k, n, z = qnorm(0.975)) {
  p <- k / n; d <- 1 + z^2 / n
  c0 <- (p + z^2 / (2 * n)) / d; h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
  cbind(lo = c0 - h, hi = c0 + h)
}
w <- wilson(fig$n_ip_recur, fig$n_ip)
fig$lo <- pmax(0, 100 * w[, "lo"])
fig$hi <- 100 * w[, "hi"]
fig <- fig[order(fig$pct_ip_recur, fig$n_ip), ]
fig$code <- factor(fig$code, levels = fig$code)

p <- ggplot(fig, aes(x = pct_ip_recur, y = code)) +
  geom_col(width = 0.7, fill = "#4C6A92") +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.25, colour = "grey30", orientation = "y") +
  geom_text(aes(x = hi, label = sprintf("%.0f%%  (%d/%d)", pct_ip_recur, n_ip_recur, n_ip)),
            hjust = -0.12, size = 3.1, colour = "grey20") +
  scale_x_continuous(limits = c(0, max(fig$hi) * 1.35), expand = expansion(mult = c(0, 0.02)),
                     labels = function(x) paste0(x, "%")) +
  labs(x = "IPs with the code in two or more distinct conversations, of IPs ever producing it",
       y = NULL) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        axis.text.y = element_text(colour = "grey10"))
ggsave("analysis/figures/code_recurrence_by_ip.png", p, width = 8.5, height = 2.8, dpi = 200, bg = "white")
ggsave("analysis/figures/code_recurrence_by_ip.pdf", p, width = 8.5, height = 2.8)
