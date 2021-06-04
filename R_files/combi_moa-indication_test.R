library("tidyverse")
library("data.table")
library("ggpubr")
library("broom")

# for heatmap visualization
library("ComplexHeatmap")
library("circlize")

input_folder = "data/processed/"
output_folder = "results/"

# data
ind = fread(paste0(input_folder, "combi_metrics_syn_astra-indication_and_MoA.txt"))
ind = ind %>% select(MoA_1, MoA_2, Indication, Max_syn_ic50) %>%
  unite("CombiMoA", MoA_1:MoA_2, remove = TRUE, sep = "_")

# Kruskal-Wallis test
kruskal = kruskal.test(Max_syn_ic50 ~ CombiMoA, data = ind)
# pairwise comparisons between group levels with corrections for multiple testing by Wilcoxon
p_wilc = pairwise.wilcox.test(ind$Max_syn_ic50, ind$CombiMoA, p.adjust.method = "BH")
p_wilc = as.array(p_wilc$p.value)

# export p-values
write.table(p_wilc, paste0(output_folder, "combi_Moa-pairwise_wilcoxon_pvalues.tsv"), sep = "\t", quote = FALSE)

# simple wilcoxon for DDR_DDR combimoa
ind_ddr = ind %>% mutate(status = ifelse(CombiMoA == "DDR_DDR", "1", "0"))
ddr_ddr = ind_ddr %>% filter(status == "1")
other = ind_ddr %>% filter(status == "0")

wilcox.test(ddr_ddr$Max_syn_ic50,
            other$Max_syn_ic50)

# level of indication

# Kruskal-Wallis test
kruskal_indi = kruskal.test(Max_syn_ic50 ~ Indication, data = ind)
# pairwise comparisons between group levels with corrections for multiple testing by Wilcoxon
wilc = pairwise.wilcox.test(ind$Max_syn_ic50, ind$Indication, p.adjust.method = "BH")
ps = wilc$p.value

# export p-values
write.table(ps, paste0(output_folder, "combi_indication-pairwise_wilcoxon_pvalues.tsv"), sep = "\t", quote = FALSE)

ps[is.na(ps)] = 1
col_fun = colorRamp2(c(0.045, 0.025, 0.001), c("white", "#FF6964", "#D7413C"))
png(paste0(output_folder, "combi_indication-pairwise_wilcoxon_pvalues.png"), units = "in", width = 5, height = 5, res = 500)

Heatmap(
  wilc$p.value,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  col = col_fun,
  na_col = "white",
  row_names_side = "left",
  heatmap_legend_param = list(
    title = "p-value",
    legend_height = unit(4, "cm"),
    title_position = "topcenter"
  ),
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (0.00001 > ps[i, j]) {
      grid.text(sprintf("***", ps[i, j]), x, y, gp = gpar(fontsize = 12))
    } else if(ps[i, j] > 0.05 & 1 > ps[i, j]) {
      grid.text(sprintf("n.s.", ps[i, j]), x, y, gp = gpar(fontsize = 8))
    }
  }
)

dev.off()
