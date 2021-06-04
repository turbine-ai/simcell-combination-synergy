## ROC for monotherapy
# input = input data with in silico monotherapy IC50s
# is_thr = in silico threshold for F1 score
# iv_thr = in vitro threshold
# filename = file name of the exported plot with extension
mono_roc = function(input,
                    is_thr = 1,
                    iv_thr = 2000,
                    filename = "Monotherapy_ROC.png") {
  input = input %>%
    mutate(IS_labels = ifelse(abs(log10(IV_DREAM) - log10(IS)) < is_thr,
                              1, 0)) %>%
    mutate(IV_DREAM_labels = ifelse(abs(IV_DREAM) < iv_thr,
                                    1, 0))
  
  
  roc = roc(input$IV_DREAM_labels, input$IS)
  f1_score = F1_Score(
    y_true = input$IV_DREAM_labels,
    y_pred = input$IS_labels,
    positive = 1
  )
  
  text1 = paste0("AUC: ",
                 round(roc$auc, digits = 2))
  
  text2 = paste0("F1 score: ",
                 round(f1_score, digits = 2))
  
  df <- data.frame(
    "sensitivity" = roc$sensitivities,
    "specificity" = roc$specificities
  )
  
  # sort sensitivity and 1 - spec.
  fpr = sort(1 - df$specificity)
  tpr = sort(df$sensitivity)
  
  ggplot(data = df, aes(x = fpr,
                        y = tpr,
                        group = 1)) +
    annotate(
      geom = "text",
      x = 0.75,
      y = 0.20,
      label = text1,
      color = "black",
      size = 5
    ) +
    annotate(
      geom = "text",
      x = 0.75,
      y = 0.10,
      label = text2,
      color = "black",
      size = 5
    ) +
    geom_line(color = "#46B4AA", size = 1.5) +
    geom_point(color = "#46B4AA", size = 2.5) +
    geom_abline(intercept = 0, slope = 1) +
    labs(title = "") +
    xlab("1 - specificity") +
    ylab("sensitivity") +
    theme(
      plot.title = element_text(
        size = 17,
        face = "bold",
        hjust = 0.5
      ),
      axis.text.x = element_text(
        size = 20,
        angle = 0,
        vjust = 0.5,
        color = "black"
      ),
      
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line = element_line(colour = "black"),
      
      axis.text.y = element_text(size = 20, color = "black"),
      axis.title.x = element_text(size = 20),
      axis.title.y = element_text(size = 20)
    )
  
  ggsave(
    paste0(output_folder, filename),
    height = 5,
    width = 7,
    units = 'in',
    dpi = 500
  )
  
  
}

# excluding classes from mono table
check_classes = function(classes) {
  if(length(unique(classes)) == 2) {
    x = "OK"
  } else {
    x = "not_for_roc"
  }
  return(x)
}
