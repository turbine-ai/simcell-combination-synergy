# Perturbation test function
# combi_result = combiomarker result table
# output_path = output folder to save histograms

pert_test = function(combi_result, output_path = output_folder) {
  mut = combi_result %>% select(Cell_line, Combiname, Max_Bliss_IC50 = Max_Bliss_IC50_mut) %>%
    mutate(mutation_status = "1")
  nomut = combi_result %>% select(Cell_line, Combiname, Max_Bliss_IC50 = Max_Bliss_IC50_nomut) %>%
    mutate(mutation_status = "0")
  hist = rbind(mut, nomut)
  
  # Overlaid histograms
  hist_plot = ggplot(hist, aes(x = Max_Bliss_IC50, fill = mutation_status)) +
    geom_histogram(alpha = 0.5, position = "identity") +
    theme_classic() +
    ylab("Count") +
    xlab("Max Bliss IC50") +
    labs(fill = 'Perturbation status') +
    theme_minimal() +
    labs(title = unique(combi_result$Combiname)) +
    theme(
      axis.text.x = element_text(
        angle = 0,
        hjust = 1,
        color = "black",
        size = 20
      ),
      legend.title = element_text(size = 20),
      legend.text = element_text(size = 15),
      axis.text.y = element_text(
        hjust = 1,
        size = 20,
        color = "black"
      ),
      axis.title.x = element_text(size = 22, color = "black"),
      axis.title.y = element_text(size = 22, color = "black")
    )
  print(hist_plot)
  
  filename = str_replace(unique(combi_result$Combiname), ":", "-")
  
  ggsave(
    paste0(output_folder, filename,
           "_Bliss_histogram",
           ".png"),
    plot = hist_plot,
    height = 5,
    width = 7,
    units = 'in',
    dpi = 500
  )
  
  
  qqplot = ggqqplot(hist$Max_Bliss_IC50, theme = theme_minimal())
  print(qqplot)
  
  return(
    wilcox.test(
      combi_result$Max_Bliss_IC50_mut,
      combi_result$Max_Bliss_IC50_nomut
    )$p.value
  )
  
}

# Mann-Whitney analysis at gene level
# combi_result = combiomarker result table
# group_var = gene name for aggregation
aggr_mannwhitney = function(combi_result, group_var = "Gene_name") {
  group_var = rlang::sym(group_var)
  
  mut = combi_result %>% select(Cell_line, Combiname, Gene_name, Max_Bliss_IC50 = Max_Bliss_IC50_mut) %>%
    mutate(mutation_status = "1")
  nomut = combi_result %>% select(Cell_line, Combiname, Gene_name, Max_Bliss_IC50 = Max_Bliss_IC50_nomut) %>%
    mutate(mutation_status = "0")
  hist = rbind(mut, nomut)
  
  
  mutation_aggr_mw = hist %>%
    group_by(!!group_var) %>%
    do(tidy(wilcox.test(Max_Bliss_IC50 ~ mutation_status, data = .)))
  
  mutation_aggr_mw = mutation_aggr_mw %>%
    ungroup() %>%
    arrange(p.value) %>%
    select(!!group_var, "Mann-Whitney_p.value" = p.value)
  
  
  return(mutation_aggr_mw)
  
}

# collect significant genes based on Mann-Whitney
# aggr_pvalues = aggregated p-values
# p_thr = p-value threshold
# col = column with Mann-Whitney p-values
# prefix = prefix for file name
sign_genes = function(aggr_pvalues,
                      p_thr = 0.001,
                      col = "Mann-Whitney_p.value") {
  col = rlang::sym(col)
  filt = aggr_pvalues %>% filter(p_thr > !!col)
  filt = filt %>% separate(Gene_name,
                           sep = ":",
                           into = c("Symbol", "Perturbation"))
  
  return(filt)
  
  
}
