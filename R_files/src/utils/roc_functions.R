## helper functions
per100 <- function(x) {
  x / 100
}

compA <- function(x) {
  comb <- str_split(x, "_")[[1]][2]
  compA <- str_split(comb, ":")[[1]][1]
}

compB <- function(x) {
  comb <- str_split(x, "_")[[1]][2]
  compB <- str_split(comb, ":")[[1]][2]
}

combination <- function(x) {
  comb <- str_split(x, "_")[[1]][2]
}

# ROC calculation
# scores. = simulated combination scores
# is_thr. = in silico threshold
# iv_thr. = in vitro threshold
roc_calc <-
  function(scores.,
           is_thr.,
           iv_thr.) {
    scores <- scores.
    scores <- scores %>%
      select(
        cl_comb_pairs,
        Bliss_max,
        Bliss_max_IC50,
        HSA_max,
        HSA_max_IC50,
        DREAM_synergy_score
      )
    scores <- scores %>%
      mutate(DREAM_synergy_score = str_replace(DREAM_synergy_score, ",", ".")) %>%
      mutate_at(vars(DREAM_synergy_score), as.numeric) %>%
      # divide by 100
      mutate_if(is.numeric, per100) %>%
      mutate(Bliss_max_is_syn = case_when(Bliss_max > is_thr. ~ 1,
                                          TRUE ~ 0)) %>%
      mutate(Bliss_maxIC50_is_syn = case_when(Bliss_max_IC50 > is_thr. ~ 1,
                                              TRUE ~ 0)) %>%
      mutate(HSA_max_is_syn = case_when(HSA_max > is_thr. ~ 1,
                                        TRUE ~ 0)) %>%
      mutate(HSA_maxIC50_is_syn = case_when(HSA_max_IC50 > is_thr. ~ 1,
                                            TRUE ~ 0)) %>%
      mutate(DREAM_iv_syn = case_when(DREAM_synergy_score > iv_thr. ~ 1,
                                      TRUE ~ 0))
    
  }

# ROC curve
# scores. = simulated combination scores (predictor)
# label = binary classes (response)
# is_label = in silico binary classes (for F1 calculation)
# plot_title = title of the ROC curve
roc_curve <-
  function(score,
           label,
           is_label,
           plot_title) {
    calc <- roc(label, score)
    auc <- calc$auc
    df <- data.frame(
      "sensitivity" = calc$sensitivities,
      "specificity" = calc$specificities
    )
    
    
    number_of_syn = sum(label)
    
    if (length(unique(is_label)) == 1) {
      f1_score = NA_integer_
    } else {
      f1_score = F1_Score(y_pred = is_label, y_true = label)
    }
    number_of_rows = length(label)
    
    text1 = paste0(
      "AUC: ",
      round(auc, digits = 2),
      ", ",
      "IV count/syn: ",
      number_of_rows,
      "/",
      number_of_syn
    )
    
    text2 = paste0("F1 score: ",
                   round(f1_score, digits = 2))
    
    # sort sensitivity and 1 - spec.
    fpr = sort(1 - df$specificity)
    tpr = sort(df$sensitivity)
    
    ggplot(data = df, aes(x = fpr,
                          y = tpr,
                          group = 1)) +
      xlab("1 - specificity") +
      ylab("sensitivity") +
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
      labs(title = plot_title) +
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
  }

## function for ROC curve computation
# input = simulated combination scores. by default: scores variable.
# is_thr = in silico threshold
# iv_thr = in vitro threshold
# export = logical. if TRUE the curve will be exported
performance_analysis = function(input = scores,
                                is_thr,
                                iv_thr,
                                export = TRUE) {
  input <-
    input %>% select(
      cl_comb_pairs,
      Bliss_max,
      Bliss_max_IC50,
      HSA_max,
      HSA_max_IC50,
      Grid_QC,
      DREAM_synergy_score
    )
  
  # roc calculation and plot
  roc = roc_calc(scores. = input,
                 is_thr. = is_thr,
                 iv_thr. = iv_thr)
  roc_curve = roc_curve(
    score = roc$Bliss_max,
    label = roc$DREAM_iv_syn,
    is_label = roc$Bliss_maxIC50_is_syn,
    plot_title = paste0(
      "Whole performance, ",
      "DREAM IV thr: ",
      as.character(iv_thr * 100)
    )
  )
  
  # exporting
  if (export == TRUE) {
    ggsave(
      paste0(
        output_folder,
        "Whole performance, ",
        " DREAM_IV_thr_",
        as.character(iv_thr * 100),
        ".png"
      ),
      plot = roc_curve,
      height = 5,
      width = 7,
      units = 'in',
      dpi = 500
    )
  }
  
}

# compound = compound of interest (e.g. AZ12618466)
# input = simulated combination scores. by default: scores variable.
# is_thr = in silico threshold
# iv_thr = in vitro threshold
# export = logical. if TRUE the curve will be exported
perf_anal_compound_level = function(input = scores,
                                    compound = "AZ12618466",
                                    is_thr = 0.2,
                                    iv_thr = 0.1,
                                    export = TRUE) {

  input <- input %>% filter(str_detect(cl_comb_pairs, compound))
  
  if (dim(input)[1] == 0) {
    print(paste0(compound, " is not found."))
  }
  
  compA_list <- sapply(input$cl_comb_pairs, compA, simplify = TRUE)
  compB_list <- sapply(input$cl_comb_pairs, compB, simplify = TRUE)
  comb_list <-
    sapply(input$cl_comb_pairs, combination, simplify = TRUE)
  
  input <-
    input %>% mutate(compA = unname(compA_list)) %>% mutate(compB = unname(compB_list))
  input <-
    inner_join(input, comb_filtering, by = c("compA" = "turbine_name"))
  input <- input %>% rename(moa_cat_compA = MoA_category)
  input <-
    inner_join(input, comb_filtering, by = c("compB" = "turbine_name"))
  input <-
    input %>% select(
      cl_comb_pairs,
      Bliss_max,
      Bliss_max_IC50,
      HSA_max,
      HSA_max_IC50,
      Grid_QC,
      DREAM_synergy_score,
      compA,
      compB,
      moa_cat_compA,
      MoA_category
    )
  input <- input %>% rename(moa_cat_compB = MoA_category)
  input <- input %>% mutate(combination = unname(comb_list))
  input <-
    inner_join(input, feas, by = c("combination" = "Combiname"))
  input <- input %>% rename(feasibility = `Combi MoA_feasibility`)
  
  # roc
  roc = roc_calc(scores. = input,
                 is_thr. = is_thr,
                 iv_thr. = iv_thr)
  roc_curve = roc_curve(
    score = roc$Bliss_max_IC50,
    label = roc$DREAM_iv_syn,
    is_label = roc$Bliss_maxIC50_is_syn,
    plot_title = paste0(compound, " DREAM IV thr: ", as.character(iv_thr * 100))
  )
  
  if (export == TRUE) {
    ggsave(
      paste0(
        output_folder,
        compound,
        " DREAM_IV_thr_",
        as.character(iv_thr * 100),
        ".png"
      ),
      plot = roc_curve,
      height = 5,
      width = 7,
      units = 'in',
      dpi = 500
    )
  }
  
}
