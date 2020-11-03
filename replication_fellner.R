#' Load data from Fellner et all (2013)
#'
#' Loading libraries
library(sandwich)
library(fixest)
library(paramtest)
library(dplyr)
library(tidyr)
library(knitr)
library(kableExtra)
library(stargazer)
library(texreg)

#' Loading data
data <- haven::read_dta("data_final.dta") %>% 
  as.data.frame()

# Describing the experiment's design --------------------------------------

#' Table counting the number of recipients in each treatment
#'
data$treatment <- as.factor(data$treatment)
t_sizes <- as.vector(table(data$treatment))

buckets <- data.frame(Buckets = c("T0", "T1", "T2", "T3", "T4", "T5", "T6"),
                      Description = c("No-mailing", "Baseline", "Threat", "Information",
                                      "Info&Threat", "Moral", "Moral&Threat"),
                      Size = t_sizes)
#' Replication of tables 1 and 2 of Fellner et al.
#' 
#' Table 1
gtab1 <- data %>% 
  group_by(treatment) %>% 
  summarise(across(gender:compliance, mean, na.rm = TRUE))
ngtab1 <- data %>% 
  summarise(Buckets = "Total", Description = "", Size = n(), 
            across(gender:compliance, mean, na.rm = TRUE))

anova_results <- data.frame(var = c("gender", "age", "pop2005", 
                                    "pop_density2005", "compliance")) %>% 
  rowwise() %>% 
  mutate(anov = list(anova(lm(paste0(var, "~treatment"), data = data))),
         pval = anov["treatment", "Pr(>F)"])
anov_row <- anova_results %>% 
  select(-anov) %>% 
  pivot_wider(names_from = var, values_from = pval) %>% 
  mutate(Buckets = "Anova: ",
         Description = "p-values",
         Size = NA_integer_) %>% 
  select(Buckets, Description, Size, everything())

tab1 <- bind_rows(ngtab1, 
                  bind_cols(buckets, gtab1) %>% 
                    select(-treatment),
                  anov_row) %>% 
  select(-Size, Size) %>% 
  kbl(digits = 2, booktabs = TRUE, format = "latex", label = "tab1",     
      col.names = c("Tratamento", "Descrição", "Gênero", "Idade",
                    "População", "Densidade pop.", "Compliance",
                    "Observações"),
      caption = "Balanceamento de características individuais e por município por tipo de tratamento.") %>% 
  kable_styling(latex_options = "HOLD_position", font_size = 10) %>% 
  kable_classic(full_width = FALSE) %>% 
  save_kable(file = "./Tables/table1.tex")

#' Regressions of Table 2
#' 
reg_21 <- feols(resp_A~mailing+threat+appeal+info, data = data)
reg_22 <- feols(resp_A~mailing+threat+appeal+info+i_tinf+i_tapp, data = data)

delivered <- data %>% 
  filter(delivered == 1)
reg_23 <- feols(resp_B~threat+appeal+info, data = delivered)
reg_24 <- feols(resp_B~threat+appeal+info+i_tinf+i_tapp, data = delivered)
reg_25 <- feols(resp_all~threat+appeal+info, data = delivered)
reg_26 <- feols(resp_all~threat+appeal+info+i_tinf+i_tapp, data = delivered)

fixest::setFixest_dict(c(resp_A = "Registro", 
                         resp_B = "Atual. Contratual", 
                         resp_all = "Resposta Geral",
                         mailing = "Correio",
                         threat = "Ameaça",
                         appeal = "Moral",
                         info = "Info",
                         i_tinf = "Ameaça x Info",
                         i_tapp = "Ameaça x Moral",
                         threat_evasion_D1 = "Ameaça x Evasão",
                         appeal_evasion_D1 = "Moral x Evasão",
                         info_evasion_D1 = "Info x Evasão",
                         evasion_1 = "Evasão",
                         threat_evasion_D2 = "Ameaça x Evasão",
                         appeal_evasion_D2 = "Moral x Evasão",
                         info_evasion_D2 = "Info x Evasão",
                         evasion_2 = "Evasão",
                         "(Intercept)" = "Constante"))
#' Create table2
esttex(reg_21, reg_22, reg_23, reg_24, reg_25, reg_26,
       file = "./Tables/table2.tex",
       label = "tab:tab2",
       replace = TRUE,
       se = "White",
       digits = 3,
       fitstat = "",
       order = c("Correio", "^Ameaça$", "^Moral$", "^Info$", "Ameaça x Moral",
                 "Ameaça x Info", "Constante"),
       title = "Efeito do tratamento nos registros, atualizações contratuais, and resposta geral para o modelo de regressão linear.")

#' Heterogeneous treatment effects
#' Replication of Table C1
#' Mediana da população dos municípios, da densidade, da renda e de votantes
#' a direita
med_pop <- median(data$pop2005, na.rm = TRUE)
med_den <- median(data$pop_density2005, na.rm = TRUE)
med_rend <- median(data$inc_aver, na.rm = TRUE)
med_vot <- median(data$vo_cr + data$vo_r, na.rm = TRUE)
#' Dataframe com indicadores de acima da mediana
efeito_het_df <- data %>% 
  mutate(pop_hi = pop2005 >= med_pop,
         popdens_hi = pop_density2005 >= med_den,
         rend_hi = inc_aver >= med_rend,
         vot_hi = (vo_cr + vo_r) >= med_vot)
#' Conjunto de variáveis de controle
Z_extended <- gsub("\\s+", "+",
                   "pop_density2005 pop2005 nat_EU nat_nonEU fam_marri fam_divor_widow
                   edu_hi edu_lo rel_evan rel_isla rel_orth_other rel_obk pers2 pers3 pers4
                   pers5more  vo_r vo_cl vo_l  j_unempl j_retire j_house j_studen
                   inc_aver age0_30 age30_60  bgld kaern noe ooe steierm tirol vlbg")
#' Regressões para efeitos heterogêneos
#' formulas
form_str <- paste0("resp_A~threat+appeal+info+gender+compliance_t+", Z_extended)
#' Default do erro padrão é ser robusto a heterocedasticidade
fixest::setFixest_se(no_FE = "white")
het_reg_pop <- efeito_het_df %>% 
  filter(delivered == 1) %>% 
  group_by(pop_hi) %>% 
  summarise(lm_model = list(feols(as.formula(form_str), data = cur_data())))

het_reg_den <- efeito_het_df %>% 
  filter(delivered == 1) %>% 
  group_by(popdens_hi) %>% 
  summarise(lm_model = list(feols(as.formula(form_str), data = cur_data())))

het_reg_rend <- efeito_het_df %>% 
  filter(delivered == 1) %>% 
  group_by(rend_hi) %>% 
  summarise(lm_model = list(feols(as.formula(form_str), data = cur_data())))

het_reg_vot <- efeito_het_df %>% 
  filter(delivered == 1) %>% 
  group_by(vot_hi) %>% 
  summarise(lm_model = list(feols(as.formula(form_str), data = cur_data())))

esttex(het_reg_pop$lm_model, het_reg_den$lm_model, het_reg_rend$lm_model, 
       het_reg_vot$lm_model,
       file = "./Tables/tablec1.tex",
       label = "tab:tabc1",
       replace = TRUE,
       se = "White",
       digits = 3,
       fitstat = "",
       keep = c("^Ameaça$", "^Moral$", "^Info$"),
       order = c("^Ameaça$", "^Moral$", "^Info$"),
       title = "Efeito heterogêneo do tratamento. Modelo de regressão linear.")

# Minimum detectable effect -----------------------------------------------
#'
# Proportion of treated
P <- 1 - t_sizes["0"]/sum(t_sizes)
# Total individuals
N <- sum(t_sizes)

#' Ideally one should NOT first collect the data, then run regressions to estimate
#' error variance, but we are not in a perfect world, thus ...
#' to replicate the results from reghdfe you need to provide the argument
#' dof = dof(cluster.df = "min", t.df = "min").
setFixest_dof(dof(cluster.df = "min", t.df = "min"))
setFixest_se(no_FE = "white")
reg_fix <- feols(resp_A~mailing, data = data)
se_beta <- coeftable(reg_fix)["mailing", 2]
k <- 0.8
a <- 0.5
#' By default, the degrees of freedom used to find the p-value from the Student t
#' distribution is equal to the number of observations minus the number of
#' estimated coefficients. N-2 in this case
tk <- qt(k, N - 2)
ta2 <- qt(1 - a/2, N - 2)
MDE <- (tk + ta2)*se_beta

#' Power simulation to detect MDE.
#' Use paramtest package that implements parallelized process to greatly 
#' improve computation speed
#' 
#' Create a function that is ONE randomization of our dataset, compute the regression
#' and infers over the treatment effect
#' model should be a character string with model formula. ex: "resp_A ~ mailing"
pt_func <- function(simNum, df, eff, model, eff_name, alpha = 0.05) {
  # No spaces allowed in model formula
  model <- gsub("\\s+", "", model)
  depvar <- sub("~.+", "", model)
  stopifnot(depvar %in% names(df))
  #' Drawing a sample, with replacement, from our data. 
  #' Since this is the real data from experiment, the sampling scheme is supposed
  #' to mimic our assignment mechanism. All variables will be sampled the same way.
  sample.draw <- sample(seq_len(nrow(df)), replace = TRUE)
  
  data.artificial <- df[sample.draw, ]
  
  #' Create outcome. 
  #' Y ~ a + beta*Treatment + error
  #' beta is treatment effect, thus beta = eff
  #' a will be the mean of control group T0
  #' We can understand E[Y] as the probability of having Y=1|T. When T=0, E[Y] = a
  #' and T = 1 then E[Y] = a + eff
  #' 
  a <- mean(df[df[, eff_name] == 0, depvar]) 
  treat <- data.artificial[, eff_name]
  ey <- a + eff*treat
  data.artificial[, depvar] <- sapply(ey, function(x) rbinom(n = 1, size = 1, prob = x))
    
  
  #' Running models and storing whether we reject the null that effect is 0
  #' feols did not play nice with parallel in paramtest
  #' get back to lm and sandwich
  # model1 <- feols(as.formula(model), data = data.artificial)
  # pval <- coeftable(model1)[eff_name, 4]
  # rej1 <- ifelse(pval < alpha, 1, 0)
  
  model2 <- lm(as.formula(model), data = data.artificial)
  se <- sqrt(diag(sandwich::vcovHC(model2, type = "HC1")))
  rej2 <- 1*(abs(model2$coefficients[eff_name]/se[eff_name]) > qnorm(1 - alpha/2))

  #' paramtest needs a named list as return
  return(c(M2 = rej2))
}

#' Compute the simulation for many possible treatment effects with 
#' paramtest::grid_search()
#' 
grid <- seq(0, 0.007, 0.001)
mail_sim <- grid_search(pt_func, n.iter = 1000, output = "data.frame",
                      df = data, model = "resp_A~mailing", eff_name = "mailing",
                      params = list(eff = grid),
                      parallel = "multicore", ncpus = 4)
mail_pwr <- results(mail_sim) %>%
  group_by(eff.test) %>%
  summarise(across(contains("mailing"), mean))

#' Full regression to estimate power over each parameter
#' Outcome: resp_A (binary)
#' Treatments: mailing, threat, appeal, info (all binary)
reg_full <- feols(resp_A~mailing+threat+appeal+info, data = data)

#' Power Tests for multiple regression.
#' "Você pode verificar isso por simulação, que para dois valores de efeitos para 
#' b2 distintos, o poder do teste para b1 não vai ser afetado; dependendo somente 
#' desse efeito. Isso se deve à propriedade de equivariância do estimador de 
#' mínimos quadrados
#' 
#' Thus, to test for the power of each parameter independently we just specify
#' the model argument as "outcome~var_to_test" and perform the simulation just as 
#' before.

#' specs for table with power calculations on Threat, Moral and Info
#' 
specs <- tibble::tibble(eff_name = c("threat", "appeal", "info"),
                        grid = list(seq(0.008, 0.012, 0.001),
                                    seq(-0.005, -0.001, 0.001),
                                    seq(-0.003, 0.001, 0.001)))

grid_search_fun <- function(grid, eff_name, n_iter = 100) {
  model <- paste0("resp_A~", eff_name)
  grid_search(pt_func, n.iter = n_iter, output = "data.frame",
              df = data, model = model, eff_name = eff_name,
              params = list(eff = grid),
              parallel = "multicore", ncpus = 4)
}

simulations <- specs %>% 
  rowwise() %>% 
  mutate(sims = list(grid_search_fun(grid, eff_name, n_iter = 1000)),
         results = list(results(sims) %>% 
                          group_by(eff.test) %>% 
                          summarise(across(contains(eff_name), mean))))

tab_powers <- do.call(cbind, simulations$results)
tab_powers <- cbind(seq_len(nrow(tab_powers)), tab_powers)

#' Function to make the assessment. Input arguments are the data frame, 
#' unconstrained formula to regress, variable name to make the assessment, 
#' coefficient under H0, number of simulations to run, significance level, an optional 
#' argument to set whether the residuals from model fit should be sampled with 
#' replacement to carry out all simulations. If there are clusters in the 
#' model they may be passed on through "cluster".
#' 
# ferman_assessment function ---------------------------------------
ferman_assessment <- function(df, model, assess_on, H0 = 0.0, nsim = 1000, 
                              alpha = 0.05, res.samp = FALSE, cluster = NULL, 
                              weights = NULL) {
  # Coercing df to data.frame ONLY (no tibble or data.table)
  df <- as.data.frame(df)
  # Thus to replicate the results from reghdfe you need to provide the argument 
  # dof = dof(cluster.df = "min", t.df = "min"). From fixest "On standard-errors"
  # https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html
  fixest::setFixest_dof(fixest::dof(cluster.df = "min", t.df = "min"))
  fixest::setFixest_se(no_FE = "white")
  # No spaces allowed in model formula
  model <- gsub("\\s+", "", model)
  depvar <- sub("~.+", "", model)
  # Simulations sequence
  sim <- seq_len(nsim)
  # Rejections vector (of 0s and 1s)
  rejections <- c()
  
  # Step 1: estimate H0 model
  pattern <- paste0(assess_on, "\\+")
  null_text <- gsub(pattern, "", model)
  off <- H0*df[, assess_on]
  weight <- df[, weights]
  H0fit <- fixest::feols(as.formula(null_text), data = df, offset = off, 
                         warn = FALSE, weights = weight)
  # H0fit <- lm(as.formula(null_text), data = df)
  # Step 2: store the predicted values
  y_pred <- predict(H0fit)
  res <- residuals(H0fit)
  
  # Step 3: nsim iterations
  # Step 3.1: draw simulation errors and generate y_sim
  for (i in sim) {
    if (res.samp)
      sim_error <- sample(res, replace = TRUE)
    else
      sim_error <- rnorm(length(res))
    
    y_sim <- y_pred + sim_error
    df[, depvar] <- y_sim
    
    # Step 3.2: Estimate the unrestricted model for each sim
    sim_reg <- fixest::feols(as.formula(model), data = df, weights = weight,
                             warn = FALSE)
    
    # Step 3.3: Test the null hypothesis for each sim
    # Extract all p-values from desired coefficient
    if (is.null(cluster)) {
      beta <- summary(sim_reg)$coeftable[assess_on, 1]
      se_beta <- summary(sim_reg)$coeftable[assess_on, 2]
      tstat <- abs((beta - H0)/se_beta)
    }
    else {
      beta <- summary(sim_reg, cluster = cluster)$coeftable[assess_on, 1]
      se_beta <- summary(sim_reg, cluster = cluster)$coeftable[assess_on, 2]
      tstat <- abs((beta - H0)/se_beta)
    }
    # Test whether pvals < alpha and store in rejections
    rejections[i] <- ifelse(tstat > qnorm(1 - alpha/2), 1, 0)
  }
  
  # Return the mean of rejections
  return(mean(rejections))
}

#' Testing the LPM regression on all four treatments
#' 
#' Unconstrained model. H0 will be imposed on this formula
model <- "resp_A~mailing+threat+appeal+info" 
assess <- list(beta0 = "mailing", 
               beta1 = "threat", 
               beta2 = "appeal", 
               beta3 = "info")
tab_assess05 <- lapply(assess, function(x){
  ferman_assessment(data, model, x)
}) %>% as.data.frame()

tab_assess10 <- lapply(assess, function(x){
  ferman_assessment(data, model, x, alpha = 0.10)
})  %>% as.data.frame()

tab_assess <- rbind(tab_assess05, tab_assess10)
tab_assess <- cbind(level = c("5%", "10%"), tab_assess)


# Randomization inference -------------------------------------------------
#' 
#' Number of permutations
n_per <- 1000
# Difference in means for control and treatments
m0 <- mean(data[data[, "mailing"] == 0, "resp_A"])
m1 <- mean(data[data[, "mailing"] == 1, "resp_A"])
m2 <- mean(data[data[, "threat"] == 1, "resp_A"])
m3 <- mean(data[data[, "appeal"] == 1, "resp_A"])
m4 <- mean(data[data[, "info"] == 1, "resp_A"])

t1 <- abs(m1 - m0)
t2 <- abs(m2 - m1) # All other statistics are partial effects!
t3 <- abs(m3 - m1)
t4 <- abs(m4 - m1)

# Hold the flags (rejections)
rej <- as.data.frame(matrix(nrow = n_per, ncol = 4))
names(rej) <- c("mailing", "threat", "appeal", "info")

for (i in seq_len(n_per)) {
  idx <- sample(N)
  # We shuffle the treatments but NOT the outcomes!
  # New positions for treatments
  pm1 <- mean(data[which(data[idx, "mailing"] == 1), "resp_A"])
  pm2 <- mean(data[which(data[idx, "threat"] == 1), "resp_A"])
  pm3 <- mean(data[which(data[idx, "appeal"] == 1), "resp_A"])
  pm4 <- mean(data[which(data[idx, "info"] == 1), "resp_A"])
  
  pt1 <- abs(pm1 - m0)
  pt2 <- abs(pm2 - m1)
  pt3 <- abs(pm3 - m1)
  pt4 <- abs(pm4 - m1)
  
  # Flag if permutation statistic is greater than baseline
  rej[i, "mailing"] <- ifelse(pt1 > t1, 1, 0)
  rej[i, "threat"] <- ifelse(pt2 > t2, 1, 0)
  rej[i, "appeal"] <- ifelse(pt3 > t3, 1, 0)
  rej[i, "info"] <- ifelse(pt4 > t4, 1, 0)
}

# P-values. Don't forget to include the baseline!! n_per + 1
ppvals <- colSums(rej) / (n_per + 1)
ppvals_df <- data.frame(treatment = c("Mailing", "Threat", "Moral", "Info"), 
                        pval = ppvals, row.names = NULL)

# Replication of results --------------------------------------------------


#' Regressions of Table 3
#' 
#' Way too much controls :-)
Z <- gsub("\\s+", "+", 
          "schober pop_density2005 pop2005  nat_EU nat_nonEU  fam_marri fam_divor_widow 
          rel_evan rel_isla rel_orth_other rel_obk pers2-pers4 pers5more  vo_r vo_cl 
          vo_l  j_unempl j_retire j_house j_studen inc_aver edu_aver age_aver   
          bgld kaern noe ooe salzbg steierm tirol vlbg")
# Z_extended <- gsub("\\s+", "+",
#                    "pop_density2005 pop2005 nat_EU nat_nonEU fam_marri fam_divor_widow 
#                    edu_hi edu_lo rel_evan rel_isla rel_orth_other rel_obk pers2 pers3 pers4 
#                    pers5more  vo_r vo_cl vo_l  j_unempl j_retire j_house j_studen 
#                    inc_aver age0_30 age30_60  bgld kaern noe ooe steierm tirol vlbg")
rhs1 <- paste0(gsub("\\s+", "+",
                    "threat appeal info threat_evasion_D1 appeal_evasion_D1 
                    info_evasion_D1 evasion_1 gender+"),
               Z)
rhs2 <- paste0(gsub("\\s+", "+",
                    "threat appeal info threat_evasion_D2 appeal_evasion_D2 
                    info_evasion_D2 evasion_2 gender+"),
               Z)
reg_31 <- feols(as.formula(paste0("resp_A~", rhs1)), data = delivered)
reg_32 <- feols(as.formula(paste0("resp_A~", rhs2)), data = delivered)
#' 
#' Save workspace
#' 
save.image("I/input/workspace.RData")
