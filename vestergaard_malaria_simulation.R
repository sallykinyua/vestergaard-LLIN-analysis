library(malariasimulation)
params_base <- get_parameters(list(human_population = 1000))
params_base <- set_equilibrium(params_base, init_EIR = 50)
cat("Parameters set!\n")


run_scenario <- function(net_coverage, label) {
  params <- params_base
  if (net_coverage > 0) {
    params <- set_bednets(
      params,
      timesteps = 1,
      coverages = net_coverage,
      retention = 625,
      dn0    = matrix(0.40, nrow = 1, ncol = 1),
      rn     = matrix(0.30, nrow = 1, ncol = 1),
      rnm    = matrix(0.24, nrow = 1, ncol = 1),
      gamman = 2.64
    )
  }
  sim <- run_simulation(timesteps = 1095, parameters = params)
  total_cases <- sum(sim$D_count)
  cat(label, "- cases:", total_cases, "\n")
  return(list(cases = total_cases, sim = sim))
}
cat("Function updated!\n")

cat("Running scenario 1 - No nets...\n")
s0 <- run_scenario(0.000, "No nets")
cat("Running scenario 2 - Country B now...\n")
sB <- run_scenario(0.373, "Country B now")
cat("Running scenario 3 - Country B BCC...\n")
sS <- run_scenario(0.534, "Country B BCC")
cat(sprintf("No nets:               %d cases\n", s0$cases))
cat(sprintf("Country B (37.3%%):     %d cases\n", sB$cases))
cat(sprintf("Country B BCC (53.4%%): %d cases\n", sS$cases))
cat(sprintf("Cases prevented by BCC: %d\n", sB$cases - sS$cases))
            
results <- data.frame(
  scenario = factor(
    c("No Nets", "Country B\nActual 37.3%", "Country B\nBCC 53.4%"),
    levels = c("No Nets", "Country B\nActual 37.3%", "Country B\nBCC 53.4%")
  ),
  cases = c(s0$cases, sB$cases, sS$cases)
)

library(ggplot2)
p <- ggplot(results, aes(x = scenario, y = cases, fill = scenario)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = format(cases, big.mark = ",")),
            vjust = -0.5, fontface = "bold", size = 5) +
  scale_fill_manual(values = c("#C0392B", "#E87722", "#2E7D32")) +
  scale_y_continuous(labels = scales::comma, limits = c(0, 14000)) +
  annotate("segment", x = 2, xend = 3, y = 13000, yend = 13000,
           arrow = arrow(ends = "both", length = unit(0.2, "cm")),
           colour = "#2E7D32", linewidth = 1.2) +
  annotate("text", x = 2.5, y = 13400,
           label = "449 cases prevented",
           colour = "#2E7D32", fontface = "bold", size = 4.5) +
  labs(
    title    = "Malaria Simulation: Clinical Cases Over 3 Years",
    subtitle = "BCC intervention improving functional net coverage in Country B (malariasimulation, EIR=50)",
    x        = NULL,
    y        = "Projected Clinical Cases (D_count)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(colour = "#1C3F6E", face = "bold"),
    plot.subtitle   = element_text(colour = "#666666"),
    plot.background = element_rect(fill = "white", colour = NA)
  )

ggsave("charts/R/r_plot8_malaria_simulation.png",
       p, width = 9, height = 5.5, dpi = 150, bg = "white")

cat("Chart saved to charts/R/r_plot8_malaria_simulation.png\n")
cat("Open it with: \n")

shell.exec(file.path(getwd(), "charts/R/r_plot8_malaria_simulation.png"))
system('git add .')
system('git -C "C:/Users/Admin/OneDrive/Desktop/vestergaard_malaria_modelling" add .')
system('git config --global --add safe.directory C:/Users/Admin/OneDrive/Desktop/vestergaard_malaria_modelling')
system('git -C "C:/Users/Admin/OneDrive/Desktop/vestergaard_malaria_modelling" add .')
system('git -C "C:/Users/Admin/OneDrive/Desktop/vestergaard_malaria_modelling" commit -m "Add malariasimulation 449 cases prevented by BCC"')
system('git config --global user.email "karimisally905@gmail.com"')
system('git config --global user.name "sallykinyua"')
system('git -C "C:/Users/Admin/OneDrive/Desktop/vestergaard_malaria_modelling" commit -m "Add malariasimulation 449 cases prevented by BCC"')
system('git -C "C:/Users/Admin/OneDrive/Desktop/vestergaard_malaria_modelling" push')

