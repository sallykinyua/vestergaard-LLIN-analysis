## =============================================================================
## Vestergaard Data Science Intern Case Study 2026
## Mosquito Net Durability Study - Complete R Analysis
## Author: Sally Karimi Kinyua | March 2026
## Run: source("analysis.R")  OR  Rscript analysis.R
## =============================================================================

# -- 0. Install & Load Packages ------------------------------------------------
required_packages <- c(
  "tidyverse", "ggplot2", "lme4", "broom",
  "survival", "survminer", "pROC",
  "RColorBrewer", "scales", "patchwork"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

cat("All packages loaded!\n")

dir.create("charts/R", showWarnings = FALSE, recursive = TRUE)

# -- Vestergaard Palette -------------------------------------------------------
NAVY   <- "#1C3F6E"
TEAL   <- "#4A8FA8"
ORANGE <- "#E87722"
RED    <- "#C0392B"
GRAY   <- "#666666"
LGRAY  <- "#F7F7F7"
GREEN  <- "#2E7D32"

theme_vestergaard <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(colour = NAVY, face = "bold", size = 14),
      plot.subtitle    = element_text(colour = GRAY, size = 10),
      axis.title       = element_text(colour = GRAY, size = 10),
      axis.text        = element_text(colour = NAVY, size = 9),
      panel.grid.major = element_line(colour = "#EEEEEE"),
      panel.grid.minor = element_blank(),
      strip.text       = element_text(colour = NAVY, face = "bold"),
      legend.position  = "bottom",
      plot.background  = element_rect(fill = "white", colour = NA)
    )
}

# -- 1. Load & Clean Data ------------------------------------------------------
df <- read.csv(
  "data/data_for_case_study_data_science_intern_2026.csv",
  stringsAsFactors = FALSE
)

cat("Dataset loaded:", nrow(df), "rows x", ncol(df), "cols\n")

# Replace 9 (don't know / refused) with NA
for (col in c("detergent", "mattress", "bed", "matground", "hang", "fold")) {
  df[[col]] <- ifelse(df[[col]] == 9, NA, df[[col]])
}
df$still <- ifelse(df$still == 9, NA, df$still)

# Create derived variables
df <- df %>%
  mutate(
    functional = case_when(
      still == 1 & surv == 1 ~ 1,
      still == 0             ~ 0,
      surv  == 0             ~ 0,
      TRUE                   ~ NA_real_
    ),
    country_B  = ifelse(country == "B", 1, 0),
    district_f = as.factor(paste0(country, "-D", district)),
    clus_f     = as.factor(clus),
    crattgr_f  = factor(crattgr, levels = 0:2,
                         labels = c("Low", "Medium", "High")),
    nattgr_f   = factor(nattgr,  levels = 0:2,
                         labels = c("Low", "Medium", "High"))
  )

cat("Functional net rate:",
    round(mean(df$functional, na.rm = TRUE) * 100, 1), "%\n")

# Subset: only nets still present
surv_data <- df %>% filter(still == 1, !is.na(surv))

# -- 2. Headline Metrics -------------------------------------------------------
cat("\n=== HEADLINE METRICS ===\n")
cat("Net retention:   ",
    round(mean(df$still, na.rm = TRUE) * 100, 1), "%\n")
cat("Serviceability:  ",
    round(mean(surv_data$surv, na.rm = TRUE) * 100, 1), "%\n")
cat("Functional nets: ",
    round(mean(df$functional, na.rm = TRUE) * 100, 1), "%\n")

# -- 3. Country Comparison -----------------------------------------------------
cat("\n=== BY COUNTRY ===\n")

country_summary <- df %>%
  group_by(country) %>%
  summarise(
    n               = n(),
    retention_pct   = round(mean(still,      na.rm = TRUE) * 100, 1),
    serviceable_pct = round(mean(surv[still == 1], na.rm = TRUE) * 100, 1),
    functional_pct  = round(mean(functional, na.rm = TRUE) * 100, 1),
    care_att_mean   = round(mean(crattgr), 2),
    net_att_mean    = round(mean(nattgr),  2),
    detergent_pct   = round(mean(detergent, na.rm = TRUE) * 100, 1),
    matground_pct   = round(mean(matground, na.rm = TRUE) * 100, 1),
    .groups = "drop"
  )

print(country_summary)

# Chi-square test
ct_country    <- table(df$country, df$functional)
chi_country   <- chisq.test(ct_country)
cat(sprintf("\nChi-square (country vs functional): X2=%.1f, p=%.6f ***\n",
            chi_country$statistic, chi_country$p.value))

# -- 4. District Breakdown -----------------------------------------------------
cat("\n=== BY DISTRICT ===\n")

district_summary <- df %>%
  group_by(country, district) %>%
  summarise(
    n              = n(),
    retention_pct  = round(mean(still,      na.rm = TRUE) * 100, 1),
    functional_pct = round(mean(functional, na.rm = TRUE) * 100, 1),
    care_att_mean  = round(mean(crattgr), 2),
    .groups = "drop"
  )

print(district_summary)

# -- 5. Attitude Distribution by Country --------------------------------------
cat("\n=== CARE ATTITUDE DISTRIBUTION BY COUNTRY ===\n")

att_dist <- df %>%
  group_by(country, crattgr) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(country) %>%
  mutate(pct = round(n / sum(n) * 100, 1))

print(att_dist)

# -- 6. Statistical Tests ------------------------------------------------------
cat("\n=== CHI-SQUARE: Country vs Functional Nets ===\n")
print(chisq.test(table(df$country, df$functional)))

cat("\n=== MANN-WHITNEY: Care Attitude Country A vs B ===\n")
mw_result <- wilcox.test(crattgr ~ country, data = df)
print(mw_result)
cat("Median Country A:", median(df$crattgr[df$country == "A"]), "\n")
cat("Median Country B:", median(df$crattgr[df$country == "B"]), "\n")

# -- 7. Spearman Correlations --------------------------------------------------
cat("\n=== SPEARMAN CORRELATIONS with Serviceability ===\n")

for (var in c("crattgr", "nattgr", "numwgr", "numdogr")) {
  result <- cor.test(surv_data[[var]], surv_data$surv,
                     method = "spearman", exact = FALSE)
  sig <- ifelse(result$p.value < 0.001, "***",
         ifelse(result$p.value < 0.01,  "**",
         ifelse(result$p.value < 0.05,  "*", "ns")))
  cat(sprintf("  %-12s  r = %6.3f   p = %.4f  %s\n",
              var, result$estimate, result$p.value, sig))
}

# -- 8. Logistic Regression ----------------------------------------------------
cat("\n=== LOGISTIC REGRESSION: Predictors of Serviceability ===\n")

model_lr <- glm(
  surv ~ crattgr + nattgr + numwgr + numdogr + country_B,
  data   = surv_data,
  family = binomial()
)

or_table <- cbind(
  OR    = round(exp(coef(model_lr)), 3),
  Lower = round(exp(confint(model_lr))[, 1], 3),
  Upper = round(exp(confint(model_lr))[, 2], 3),
  pval  = round(summary(model_lr)$coefficients[, 4], 4)
)

cat("\nOdds Ratios (95% CI):\n")
print(or_table)
cat(sprintf("AIC: %.1f\n", AIC(model_lr)))

# -- 9. Mixed-Effects Logistic Regression -------------------------------------
cat("\n=== MIXED-EFFECTS LOGISTIC REGRESSION ===\n")
cat("(Random intercept per cluster - accounts for within-cluster correlation)\n")

model_mixed <- glmer(
  surv ~ crattgr + nattgr + country_B + (1 | clus_f),
  data    = surv_data,
  family  = binomial(),
  control = glmerControl(optimizer = "bobyqa")
)

print(summary(model_mixed)$coefficients)

# -- 10. ROC / AUC -------------------------------------------------------------
cat("\n=== MODEL DISCRIMINATION (AUC) ===\n")

surv_data$pred_prob <- predict(model_lr, type = "response")
roc_obj <- roc(surv_data$surv, surv_data$pred_prob, quiet = TRUE)
cat(sprintf("AUC: %.3f  (95%% CI: %.3f - %.3f)\n",
            auc(roc_obj),
            ci.auc(roc_obj)[1],
            ci.auc(roc_obj)[3]))

# -- 11. Survival Analysis -----------------------------------------------------
cat("\n=== SURVIVAL ANALYSIS ===\n")
cat("Event = net loss (still == 0). Censored at 24 months.\n")

surv_obj <- Surv(
  time  = rep(24, nrow(df)),
  event = ifelse(!is.na(df$still) & df$still == 0, 1, 0)
)

km_fit   <- survfit(surv_obj ~ country, data = df)
logrank  <- survdiff(surv_obj ~ country, data = df)

cat("\nKaplan-Meier table:\n")
print(summary(km_fit)$table)

cat("\nLog-rank test (Country A vs B):\n")
print(logrank)

cat("\n=== COX PROPORTIONAL HAZARDS MODEL ===\n")
cox_model <- coxph(
  surv_obj ~ crattgr + nattgr + country_B + matground,
  data = df
)

hr_table <- cbind(
  HR    = round(exp(coef(cox_model)), 3),
  Lower = round(exp(confint(cox_model))[, 1], 3),
  Upper = round(exp(confint(cox_model))[, 2], 3),
  pval  = round(summary(cox_model)$coefficients[, 5], 4)
)

cat("\nHazard Ratios (risk of net loss):\n")
print(hr_table)

# -- 12. Counterfactual Simulation ---------------------------------------------
cat("\n=== COUNTERFACTUAL SIMULATION ===\n")
cat("Q: What if Country B had Country A's care attitude distribution?\n\n")

a_att_dist  <- df %>%
  filter(country == "A") %>%
  count(crattgr) %>%
  mutate(prop = n / sum(n))

serv_by_att <- surv_data %>%
  group_by(crattgr) %>%
  summarise(serv_rate = mean(surv, na.rm = TRUE), .groups = "drop")

b_retention        <- mean(df$still[df$country == "B"], na.rm = TRUE)
b_serv_actual      <- mean(surv_data$surv[surv_data$country == "B"], na.rm = TRUE)
b_functional_now   <- b_retention * b_serv_actual

sim_serv_rate <- sum(
  a_att_dist$prop * serv_by_att$serv_rate[
    match(a_att_dist$crattgr, serv_by_att$crattgr)
  ],
  na.rm = TRUE
)

sim_functional <- b_retention * sim_serv_rate

cat(sprintf("Country B CURRENT functional rate:    %.1f%%\n",
            b_functional_now   * 100))
cat(sprintf("Country B SIMULATED functional rate:  %.1f%%\n",
            sim_functional     * 100))
cat(sprintf("Improvement:                         +%.1f pp\n",
            (sim_functional - b_functional_now) * 100))
cat(sprintf("Extra nets protected per 1,000:      +%.0f nets\n",
            (sim_functional - b_functional_now) * 1000))

# -- 13. Visualisations --------------------------------------------------------
cat("\n=== GENERATING CHARTS ===\n")

## Plot 1 - Country comparison
p1 <- country_summary %>%
  select(country, retention_pct, serviceable_pct, functional_pct) %>%
  pivot_longer(-country, names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric,
    levels = c("retention_pct", "serviceable_pct", "functional_pct"),
    labels = c("Retention", "Serviceability", "Functional")
  )) %>%
  ggplot(aes(x = metric, y = value, fill = country)) +
  geom_col(position = "dodge", width = 0.65,
           colour = "white", linewidth = 0.8) +
  geom_text(aes(label = paste0(value, "%")),
            position = position_dodge(0.65),
            vjust = -0.5, fontface = "bold", size = 3.8) +
  scale_fill_manual(values = c("A" = NAVY, "B" = ORANGE),
                    labels = c("Country A", "Country B")) +
  scale_y_continuous(limits = c(0, 105),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Net Outcomes at 24 Months: Country A vs Country B",
    subtitle = "Chi-square p<0.001 - significant gap across all three metrics",
    x = NULL, y = "Percentage (%)", fill = "Country"
  ) +
  theme_vestergaard()

ggsave("charts/R/r_plot1_country_comparison.png",
       p1, width = 9, height = 5.5, dpi = 150)
cat("  Plot 1 saved\n")

## Plot 2 - Care attitude vs serviceability
att_serv <- surv_data %>%
  group_by(crattgr) %>%
  summarise(
    serviceability = round(mean(surv, na.rm = TRUE) * 100, 1),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(att_label = factor(crattgr,
    labels = c("Low\n(score 0)", "Medium\n(score 1)", "High\n(score 2)")))

p2 <- ggplot(att_serv, aes(x = att_label, y = serviceability)) +
  geom_col(fill = c(RED, ORANGE, GREEN),
           width = 0.6, colour = "white", linewidth = 0.8) +
  geom_text(aes(label = paste0(serviceability, "%\n(n=", n, ")")),
            vjust = -0.4, fontface = "bold", size = 4, colour = NAVY) +
  annotate("segment", x = 1, xend = 3, y = 88, yend = 88,
           arrow = arrow(ends = "both", length = unit(0.2, "cm")),
           colour = GREEN, linewidth = 1.2) +
  annotate("text", x = 2, y = 91.5,
           label = "+24.5 percentage points",
           colour = GREEN, fontface = "bold", size = 4) +
  scale_y_continuous(limits = c(0, 100),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Care & Repair Attitude vs Net Serviceability",
    subtitle = "OR=1.52 (95% CI: 1.27-1.82), Spearman r=0.246, p<0.001",
    x = "Care & Repair Attitude Score",
    y = "Serviceability Rate (%)"
  ) +
  theme_vestergaard()

ggsave("charts/R/r_plot2_attitude_serviceability.png",
       p2, width = 8, height = 5.5, dpi = 150)
cat("  Plot 2 saved\n")

## Plot 3 - District functional nets
p3 <- district_summary %>%
  mutate(dist_label = paste0(country, "-District ", district)) %>%
  ggplot(aes(x = reorder(dist_label, functional_pct),
             y = functional_pct, fill = country)) +
  geom_col(width = 0.65, colour = "white", linewidth = 0.8) +
  geom_text(aes(label = paste0(functional_pct, "%")),
            hjust = -0.2, fontface = "bold", size = 4) +
  geom_hline(yintercept = 66.2, colour = NAVY,
             linetype = "dashed", linewidth = 1, alpha = 0.7) +
  annotate("text", x = 0.65, y = 68, label = "Country A avg: 66.2%",
           colour = NAVY, size = 3.5, fontface = "italic") +
  scale_fill_manual(values = c("A" = NAVY, "B" = ORANGE)) +
  scale_y_continuous(limits = c(0, 95),
                     labels = function(x) paste0(x, "%")) +
  coord_flip() +
  labs(
    title    = "Functional Nets by District at 24 Months",
    subtitle = "A-District 4 outlier: 15.7pp below Country A average",
    x = NULL, y = "Functional Net Rate (%)", fill = "Country"
  ) +
  theme_vestergaard()

ggsave("charts/R/r_plot3_district_functional.png",
       p3, width = 9, height = 5.5, dpi = 150)
cat("  Plot 3 saved\n")

## Plot 4 - Attitude distribution stacked bar
p4 <- att_dist %>%
  mutate(att_label = factor(crattgr,
    labels = c("Low (0)", "Medium (1)", "High (2+)"))) %>%
  ggplot(aes(x = country, y = pct, fill = att_label)) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.8) +
  geom_text(aes(label = paste0(pct, "%")),
            position = position_stack(vjust = 0.5),
            colour = "white", fontface = "bold", size = 4.5) +
  scale_fill_manual(
    values = c("Low (0)" = RED, "Medium (1)" = ORANGE, "High (2+)" = TEAL),
    name   = "Care Attitude"
  ) +
  scale_x_discrete(labels = c("Country A", "Country B")) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Care & Repair Attitude Distribution by Country",
    subtitle = "Country B: only 13.3% high attitude vs 78.4% in Country A",
    x = NULL, y = "Percentage of Households (%)"
  ) +
  theme_vestergaard()

ggsave("charts/R/r_plot4_attitude_distribution.png",
       p4, width = 8, height = 5.5, dpi = 150)
cat("  Plot 4 saved\n")

## Plot 5 - Counterfactual simulation
sim_data <- data.frame(
  scenario = factor(
    c("Country A\n(actual)",
      "Country B\n(actual)",
      "Country B\n(simulated BCC)"),
    levels = c("Country A\n(actual)",
               "Country B\n(actual)",
               "Country B\n(simulated BCC)")
  ),
  functional = c(66.2, 37.3, round(sim_functional * 100, 1)),
  bar_type   = c("actual_a", "actual_b", "simulated")
)

p5 <- ggplot(sim_data,
             aes(x = scenario, y = functional, fill = bar_type)) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.8) +
  geom_text(aes(label = paste0(functional, "%")),
            vjust = -0.5, fontface = "bold", size = 5) +
  geom_hline(yintercept = 66.2,
             linetype = "dashed", colour = NAVY,
             linewidth = 1, alpha = 0.6) +
  annotate("text",
           x     = 3,
           y     = sim_functional * 100 + 9,
           label = paste0("+", round((sim_functional - b_functional_now)*100,1),
                          "pp\n+",
                          round((sim_functional - b_functional_now)*1000, 0),
                          " nets/1,000"),
           colour = GREEN, fontface = "bold", size = 4) +
  scale_fill_manual(
    values = c("actual_a" = NAVY, "actual_b" = ORANGE, "simulated" = GREEN)
  ) +
  scale_y_continuous(limits = c(0, 85),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Counterfactual Simulation: Impact of Behaviour Change",
    subtitle = "If Country B households adopted Country A care attitudes",
    x = NULL, y = "Functional Net Rate (%)"
  ) +
  theme_vestergaard() +
  theme(legend.position = "none")

ggsave("charts/R/r_plot5_simulation.png",
       p5, width = 9, height = 5.5, dpi = 150)
cat("  Plot 5 saved\n")

## Plot 6 - ROC curve
roc_df <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

p6 <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(colour = NAVY, linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = GRAY) +
  annotate("text", x = 0.65, y = 0.15,
           label = sprintf("AUC = %.3f", auc(roc_obj)),
           colour = NAVY, fontface = "bold", size = 5) +
  labs(
    title    = "ROC Curve - Logistic Regression",
    subtitle = "Predicting net serviceability at 24 months",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_vestergaard()

ggsave("charts/R/r_plot6_roc_curve.png",
       p6, width = 7, height = 6, dpi = 150)
cat("  Plot 6 saved\n")

## Plot 7 - KM survival curve
km_plot <- ggsurvplot(
  km_fit,
  data        = df,
  palette     = c(NAVY, ORANGE),
  conf.int    = TRUE,
  pval        = TRUE,
  legend.labs = c("Country A", "Country B"),
  legend.title = "Country",
  title       = "Net Survival Curve: Country A vs Country B",
  subtitle    = "Probability of net remaining in household over 24 months",
  xlab        = "Time (months)",
  ylab        = "Net Survival Probability",
  ggtheme     = theme_vestergaard()
)

ggsave("charts/R/r_plot7_survival_km.png",
       plot   = km_plot$plot,
       width  = 9,
       height = 5.5,
       dpi    = 150,
       bg     = "white")
cat("  Plot 7 (KM survival) saved\n")

cat("\n All R charts saved to charts/R/\n")
cat("\n=== SUGGESTED FOLLOW-UP ANALYSES ===\n")
cat("1. Kaplan-Meier with multiple timepoints (12m, 24m, 36m)\n")
cat("2. Mixed-effects model with district-level random slopes\n")
cat("3. Spatial autocorrelation - Moran's I (needs GPS per cluster)\n")
cat("4. Random Forest for at-risk HH prediction at distribution\n")
cat("5. Net class subgroup analysis if product codes available\n")
cat("\nAnalysis complete.\n")
