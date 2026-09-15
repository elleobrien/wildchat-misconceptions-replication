#!/usr/bin/env Rscript
# Sensitivity analyses for the manuscript's chi-square tests (Appendix
# "Sensitivity analyses"). The headline tests treat every conversation, or
# every (IP, code) pair, as an independent observation. Reviewers flagged
# three ways that can fail, and this script re-runs each test with the
# corresponding adjustment:
#
#  (a) IP clustering: several conversations come from the same hashed IP.
#      -> mixed-effects logistic regression (lme4::glmer, the binomial sibling
#         of lmer) with a random intercept per hashed_ip; plus an IP-level
#         aggregation check (any refusal per IP x family).
#  (b) Label multiplicity: a multi-label conversation contributes one row per
#      label to the refusal-by-label table.
#      -> refit on single-label conversations only, with the IP random
#         intercept.
#  (c) User activity: an IP with many conversations has more chances to
#      recur, so recurrence-by-code could be an activity artefact.
#      -> logistic regression of recurrence on code with log(total
#         conversations per IP over the full Python corpus) as a covariate.
#         The IP random intercept is fitted but its variance is estimated at
#         zero (few IPs appear under more than one code), so inference uses
#         ordinary logistic regression. Only the omnibus code effect (model
#         with code vs. activity-only null, LRT) is reported; Non-text output
#         has zero recurrences, so per-code coefficients would separate.
#
# Headline numbers are recomputed here (same conventions as
# correction_omnibus_tests.R and code_recurrence_by_ip.R) so the appendix
# table is self-contained.
#
# Inputs:  analysis/data/conversations.csv, corrections.csv, conversation_ips.csv,
#          ip_activity.csv (conversation_ips.py)
# Outputs: analysis/data/sensitivity_stats.csv        (key/value)
#          analysis/data/sensitivity_activity_by_code.csv
#   Rscript analysis/sensitivity_clustering.R
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(lme4)})
set.seed(42)
B <- 1e5
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

conv <- read.csv("analysis/data/conversations.csv", stringsAsFactors = FALSE,
                 check.names = FALSE, na.strings = character(0)) %>%
  filter(in_corpus == 1, labels_final != "")
cor  <- read.csv("analysis/data/corrections.csv", stringsAsFactors = FALSE) %>%
  select(conversation_hash, corrected = any_correction)
meta <- read.csv("analysis/data/conversation_ips.csv", stringsAsFactors = FALSE) %>%
  select(conversation_hash, hashed_ip, model)
act  <- read.csv("analysis/data/ip_activity.csv", stringsAsFactors = FALSE)
d <- conv %>% inner_join(cor, by = "conversation_hash") %>%
  inner_join(meta, by = "conversation_hash") %>%
  mutate(family = factor(ifelse(grepl("^gpt-4", model), "GPT-4", "GPT-3.5")))
stopifnot(nrow(d) == nrow(conv), !anyNA(d$hashed_ip))

ip_tab <- table(d$hashed_ip)
stats <- c(n_conv = nrow(d), n_ip = length(ip_tab), n_ip_multi = sum(ip_tab > 1),
           n_conv_from_multi = sum(ip_tab[ip_tab > 1]), max_conv_per_ip = max(ip_tab))
lrt <- function(m0, m1) { a <- anova(m0, m1); c(chisq = a$Chisq[2], df = a$Df[2], p = a$`Pr(>Chisq)`[2]) }
# Two labels (Continuous training, Clear session) have zero refusals, so the
# label fixed effects separate and glmer reports a singular/degenerate Hessian.
# The LRT statistic is identical across bobyqa, Nelder-Mead and nlminb (38.75
# at nAGQ = 1) and dropping the two labels gives the same conclusion (see
# six_label_* keys), so those Hessian warnings are muffled here.
quiet <- function(expr) withCallingHandlers(expr, warning = function(w) {
  if (grepl("Hessian|scaled gradient", conditionMessage(w))) invokeRestart("muffleWarning") })

## (a) Refusal by model family --------------------------------------------
cat("== Refusal by model family ==\n")
fam_naive <- chisq.test(table(d$family, d$corrected), correct = FALSE)
f0 <- glmer(corrected ~ 1 + (1 | hashed_ip), data = d, family = binomial, control = ctrl)
f1 <- update(f0, . ~ . + family)
fam_lrt <- lrt(f0, f1)
fam_or  <- exp(c(fixef(f1)["familyGPT-4"], confint(f1, parm = "familyGPT-4", method = "Wald")))
fam_sd  <- as.data.frame(VarCorr(f1))$sdcor[1]
# IP-level aggregation: did the IP produce any refusal, within each family.
u <- d %>% group_by(hashed_ip, family) %>%
  summarise(any = as.integer(any(corrected == 1)), .groups = "drop")
fam_agg <- chisq.test(table(u$family, u$any), correct = FALSE)
cat(sprintf("naive chi-square p = %.4f; glmer LRT p = %.4f; OR(GPT-4) = %.2f [%.2f, %.2f]; IP-level p = %.3f\n",
            fam_naive$p.value, fam_lrt["p"], fam_or[1], fam_or[2], fam_or[3], fam_agg$p.value))
stats <- c(stats,
  fam_naive_chisq = unname(fam_naive$statistic), fam_naive_p = fam_naive$p.value,
  fam_glmer_chisq = fam_lrt[["chisq"]], fam_glmer_df = fam_lrt[["df"]], fam_glmer_p = fam_lrt[["p"]],
  fam_or = unname(fam_or[1]), fam_or_lo = unname(fam_or[2]), fam_or_hi = unname(fam_or[3]),
  fam_ip_sd = fam_sd,
  fam_agg_n = nrow(u), fam_agg_chisq = unname(fam_agg$statistic), fam_agg_p = fam_agg$p.value)

## (a)+(b) Refusal by misconception label ----------------------------------
cat("\n== Refusal by label ==\n")
dl <- d %>% separate_rows(labels_final, sep = ";[[:space:]]*") %>% rename(label = labels_final)
lab_naive <- chisq.test(table(dl$label, dl$corrected), simulate.p.value = TRUE, B = B)
l0 <- glmer(corrected ~ 1 + (1 | hashed_ip), data = dl, family = binomial, control = ctrl)
l1 <- quiet(update(l0, . ~ . + label))
lab_lrt <- lrt(l0, l1)
dl6 <- dl %>% filter(!label %in% c("Continuous training", "Clear session"))
x0 <- glmer(corrected ~ 1 + (1 | hashed_ip), data = dl6, family = binomial, control = ctrl)
x1 <- update(x0, . ~ . + label)
six_lrt <- lrt(x0, x1)
ds <- d %>% filter(n_labels_final == 1) %>% mutate(label = labels_final)
single_naive <- chisq.test(table(ds$label, ds$corrected), simulate.p.value = TRUE, B = B)
s0 <- glmer(corrected ~ 1 + (1 | hashed_ip), data = ds, family = binomial, control = ctrl)
s1 <- quiet(update(s0, . ~ . + label))
single_lrt <- lrt(s0, s1)
cat(sprintf("rows = %d; naive simulated p = %.2g; glmer LRT p = %.2g\n", nrow(dl), lab_naive$p.value, lab_lrt["p"]))
cat(sprintf("six labels with >=1 refusal: rows = %d; glmer LRT p = %.2g\n", nrow(dl6), six_lrt["p"]))
cat(sprintf("single-label convs = %d; naive p = %.2g; glmer LRT p = %.2g\n", nrow(ds), single_naive$p.value, single_lrt["p"]))
stats <- c(stats,
  lab_rows = nrow(dl), lab_naive_chisq = unname(lab_naive$statistic), lab_naive_p = lab_naive$p.value,
  lab_glmer_chisq = lab_lrt[["chisq"]], lab_glmer_df = lab_lrt[["df"]], lab_glmer_p = lab_lrt[["p"]],
  lab_ip_sd = as.data.frame(VarCorr(l1))$sdcor[1],
  six_label_rows = nrow(dl6), six_label_chisq = six_lrt[["chisq"]], six_label_df = six_lrt[["df"]], six_label_p = six_lrt[["p"]],
  single_n = nrow(ds), single_naive_chisq = unname(single_naive$statistic), single_naive_p = single_naive$p.value,
  single_glmer_chisq = single_lrt[["chisq"]], single_glmer_df = single_lrt[["df"]], single_glmer_p = single_lrt[["p"]],
  B = B)

## (c) Recurrence by code, adjusting for activity --------------------------
cat("\n== Recurrence by code ==\n")
MIN_IPS <- 10
per <- dl %>% rename(code = label) %>% count(code, hashed_ip) %>%
  mutate(recur = as.integer(n >= 2)) %>% left_join(act, by = "hashed_ip")
stopifnot(!anyNA(per$total_convs))
kept <- per %>% count(code) %>% filter(n >= MIN_IPS) %>% pull(code)
per <- per %>% filter(code %in% kept) %>% mutate(log_act = log(total_convs))
per$code <- relevel(factor(per$code), ref = "Internet Access")
rec_naive <- chisq.test(table(per$code, per$recur), simulate.p.value = TRUE, B = B)
act_by_code <- per %>% group_by(code) %>%
  summarise(n_ip = n(), n_recur = sum(recur), pct_recur = 100 * mean(recur),
            median_convs = median(total_convs), mean_convs = mean(total_convs),
            pct_ip_multi = 100 * mean(total_convs >= 2), .groups = "drop") %>%
  arrange(desc(pct_recur))
print(as.data.frame(act_by_code), row.names = FALSE, digits = 3)
write.csv(act_by_code, "analysis/data/sensitivity_activity_by_code.csv", row.names = FALSE)
act_kw <- kruskal.test(total_convs ~ code, data = per)
# Mixed model: IP random intercept (an IP can appear under several codes).
rm1 <- glmer(recur ~ log_act + code + (1 | hashed_ip), data = per, family = binomial, control = ctrl)
rec_ip_sd <- as.data.frame(VarCorr(rm1))$sdcor[1]
cat(sprintf("glmer IP random-intercept SD = %.2g (singular = %s) -> inference from glm below\n",
            rec_ip_sd, isSingular(rm1)))
# Ordinary logistic regression with activity covariate.
g0 <- glm(recur ~ log_act, data = per, family = binomial)
g1 <- glm(recur ~ log_act + code, data = per, family = binomial)
rec_lrt <- anova(g0, g1, test = "Chisq")
act_or  <- exp(c(coef(g1)["log_act"], confint.default(g1, "log_act")))
cat(sprintf("naive p = %.2g; activity-adjusted LRT p = %.2g; activity OR per log-conv = %.2f\n",
            rec_naive$p.value, rec_lrt$`Pr(>Chi)`[2], act_or[1]))
cat(sprintf("activity by code Kruskal p = %.2g\n", act_kw$p.value))
stats <- c(stats,
  rec_pairs = nrow(per), rec_ips = n_distinct(per$hashed_ip),
  rec_ip_multi_code = sum(table(per$hashed_ip) > 1),
  rec_naive_chisq = unname(rec_naive$statistic), rec_naive_p = rec_naive$p.value,
  rec_ip_sd = rec_ip_sd,
  rec_adj_chisq = rec_lrt$Deviance[2], rec_adj_df = rec_lrt$Df[2], rec_adj_p = rec_lrt$`Pr(>Chi)`[2],
  act_or = unname(act_or[1]), act_or_lo = unname(act_or[2]), act_or_hi = unname(act_or[3]),
  act_kw_chisq = unname(act_kw$statistic), act_kw_p = act_kw$p.value)

write.csv(data.frame(key = names(stats), value = unname(stats)),
          "analysis/data/sensitivity_stats.csv", row.names = FALSE)
cat("\nwrote analysis/data/sensitivity_stats.csv and sensitivity_activity_by_code.csv\n")
