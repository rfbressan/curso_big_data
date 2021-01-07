#' Analise e replicacao de alguns resultados em Fellner et all (2013)
#' Autor: Rafael Felipe Bressan
#' Arquivo: script_R.R
#' Script R para o trabalho monografico: Inferência Causal com Machine Learning
#' uma aplicacao para evasao fiscal
#' Pós-Graduacao Lato Sensu em Ciencia de Dados e Big Data PUC-MG
#' Ano: 2021
#' 
#' Carrega as bibliotecas
library(sandwich)
library(fixest)
library(tidyverse)
library(glue)
library(skimr)
library(knitr)
library(kableExtra)
library(stargazer)
library(texreg)

#' Tabela 1 com a descricao das variaveis
desc_tbl <- read_csv("descricao.csv")

kbl(desc_tbl, booktabs = TRUE, longtable = TRUE, format = "latex",
    col.names = c("Variável", "Descrição"),
    caption = "Variáveis e descrições",
    label = "descricao") %>%
  kable_styling(full_width = FALSE, 
                latex_options = c("repeat_header"),
                repeat_header_text = "(continuação)") %>%
  column_spec(2, width = "30em") %>% 
  save_kable(file = "./Tables/table_descricao.tex")

#' Carregando os dados
data <- haven::read_dta("data_final.dta") %>% 
  as.data.frame()

# Descrevendo o desenho do experimento --------------------------------

#' Tabela contando o numero de recipientes de cada tratamento
#'
data$treatment <- factor(data$treatment)
t_sizes <- as.vector(table(data$treatment))

buckets <- data.frame(treatment = sort(unique(data$treatment)),
                      Buckets = c("T0", "T1", "T2", "T3", "T4", "T5", "T6"),
                      Description = c("Sem Correio", "Correio", "Ameaça", "Info",
                                      "Info&Ameaça", "Moral", "Moral&Ameaça"),
                      Size = t_sizes) %>% 
  mutate(Prop = Size / nrow(data))

#' Tabela 2 recipientes de cada tratamento
kbl(buckets[-1], format = "latex", booktabs = TRUE, label = "descritivas1",
    col.names = c("Tratamento", "Descrição", "Observações", "Proporção"),
    caption = "Distribuição dos tratamentos na amostra.") %>% 
  kable_styling(latex_options = c("HOLD_position")) %>% 
  kable_classic(full_width = FALSE) %>% 
  save_kable("./Tables/table_descritivas1.tex")

#' Junta de volta as inforamacoes de bucket e descricao para os dados
data <- data %>% 
  left_join(buckets[, c("treatment", "Buckets", "Description")], by = "treatment")
#' Estatisticas descritivas da base
#' Variaveis com valor NA
skim_df <- data %>% 
  skim_without_charts()

na_df <- skim_df %>% 
  filter(n_missing > 0) %>% 
  select(skim_variable, n_missing, complete_rate)

#' Tabela 3 dados faltantes
kbl(na_df, digits = 2, format = "latex", booktabs = TRUE, label = "missings",
    col.names = c("Variável", "No. Faltantes", "Completude"),
    caption = "Dados faltantes na amostra.") %>% 
  kable_styling(latex_options = c("HOLD_position")) %>% 
  kable_classic(full_width = FALSE) %>% 
  footnote(general_title = "Nota:",
           general = "Completude refere-se a proporção de linhas preenchidas contra faltantes, e varia de zero a um.",
           threeparttable = TRUE) %>% 
  save_kable("./Tables/table_missings.tex")

#' Analise Exploratoria
#' Problema de atrito
attrition_level <- data %>% 
  filter(mailing == 1) %>% 
  group_by(treatment, Buckets, Description) %>% 
  summarise(mail_count = n(),
            deliv_na_count = sum(is.na(delivered)),
            deliv_0_count = sum(delivered == 0),
            attr_rate = (deliv_na_count + deliv_0_count) / nrow(cur_data()))
chi_test <- chisq.test(attrition_level$deliv_0_count, 
                       p = attrition_level$mail_count, rescale.p = TRUE)
atrito_foot <- glue("Na média total a taxa de atrito foi de {format(mean(attrition_level$attr_rate), digits = 4)} e não houve diferença entre tratamentos, como aponta o teste qui-quadrado de Pearson para dados de contagem, com p-valor de {format(chi_test$p.value, digits = 4)}.")
#' Tabela 5 detalhando o nivel de atrito por tratamento
kbl(attrition_level[-1], digits = 4, format = "latex", booktabs = TRUE, 
    label = "atrito-level", 
    col.names = c("Tratamento", "Descrição", "Cartas", "Entregues NA", 
                  "Não Entregues", "Taxa Atrito"),
    caption = "Taxa de atrito por tratamento.") %>% 
  kable_styling(latex_options = c("HOLD_position")) %>% 
  kable_classic(full_width = FALSE) %>% 
  footnote(general_title = "Nota:", general = atrito_foot,
           threeparttable = TRUE) %>% 
  save_kable("./Tables/table_atrito_level.tex")
#' Todos os NAs se referem ao grupo de controle, mas houve um pouco de atrito.
#' Verificar balanceamento de variaveis para aqueles que atritaram
atrito_controle <- data %>% 
  filter(mailing == 0 | (mailing == 1 & delivered == 0))

attr_bal <- atrito_controle %>% 
  group_by(treatment, Buckets) %>% 
  summarise(across(c(gender, age_aver, inc_aver, pop2005, pop_density2005, 
                     compliance), 
                   mean, na.rm = TRUE))
#' Teste anova para diferenca de medias
attr_anova <- tibble(var = c("gender", "age_aver", "inc_aver", "pop2005", 
                                    "pop_density2005", "compliance")) %>% 
  rowwise() %>% 
  mutate(anov = list(anova(lm(paste0(var, "~treatment"), data = atrito_controle))),
         pval = anov["treatment", "Pr(>F)"]) %>% 
  select(var,pval) %>% 
  pivot_wider(names_from = var, values_from = pval) %>% 
  add_column(Buckets = "Anova p-valor", .before = 1)

atr_bal_foot <- "Gênero igual a zero para mulher. Demais variáveis são denominadas em nível municipal, por exemplo Idade refere-se a idade média dos habitantes do município de residência do indivíduo."
#' Tabela 6 balanceamento com atrito
attr_bal[-1] %>% 
  bind_rows(attr_anova) %>% 
  kbl(digits = 4, format = "latex", booktabs = TRUE, label = "atrito-bal",
      col.names = c("Tratamento", "Gênero", "Idade", "Renda", "População", 
                    "Dens. pop.", "Compliance"),
      caption = "Análise de atrito. Balanceamento de variáveis selecionadas") %>% 
  kable_styling(latex_options = c("HOLD_position")) %>% 
  kable_classic(full_width = FALSE) %>% 
  footnote(general_title = "Nota:",
           threeparttable = TRUE,
           general = atr_bal_foot) %>% 
  save_kable("./Tables/table_atrito_bal.tex")
#' Histograma com designacao de tratamento e atrito
hist1 <- data %>% 
  filter(treatment %in% c("0", "6"), pop_density2005 < 30) %>% 
  select(treatment, pop_density2005) %>% 
  ggplot(aes(x = pop_density2005, y = ..density.., fill = treatment)) +
  geom_histogram(bins = 50, alpha = 0.8, position = "dodge") +
  labs(x = "",
       y = "Frequência relativa",
       title = "Tratamento designado") +
  guides(fill = guide_legend(title = "Tratamento")) +
  scale_fill_discrete(type = c("blue", "red")) +
  theme_classic()

hist2 <- atrito_controle %>% 
  filter(treatment %in% c("0", "6"), pop_density2005 < 30) %>% 
  select(treatment, pop_density2005) %>% 
  ggplot(aes(x = pop_density2005, y = ..density.., fill = treatment)) +
  geom_histogram(bins = 50, alpha = 0.8, position = "dodge") +
  labs(x = "Densidade Populacional (hab/km2)",
       y = "Frequência relativa",
       title = "Atrito") +
  guides(fill = guide_legend(title = "Tratamento")) +
  scale_fill_discrete(type = c("blue", "red")) +
  theme_classic()
png("./Figs/fig_atr_hist.png")
gridExtra::grid.arrange(hist1, hist2)
dev.off()

#' Replicacao das tabelas 1 e 2 de Fellner et al.
#' 
#' Tabela 1
gtab1 <- data %>% 
  group_by(treatment, Buckets, Description) %>% 
  summarise(across(c(gender, age_aver, inc_aver, pop2005, 
                     pop_density2005, compliance),
                   mean, na.rm = TRUE))

anova_results <- data.frame(var = c("gender", "age_aver", "inc_aver", "pop2005", 
                                    "pop_density2005", "compliance")) %>% 
  rowwise() %>% 
  mutate(anov = list(anova(lm(paste0(var, "~treatment"), data = data))),
         pval = anov["treatment", "Pr(>F)"])
anov_row <- anova_results %>% 
  select(-anov) %>% 
  pivot_wider(names_from = var, values_from = pval) %>% 
  mutate(Buckets = "Anova: ",
         Description = "p-values") %>% 
  select(Buckets, Description, everything())

#' Tabela 4 balanceamento. Table 1 de Fellner et. al
tbl1_cap <- "Balanceamento de características individuais e por município por tipo de tratamento."
gtab1[-1] %>% 
  bind_rows(anov_row) %>% 
  kbl(digits = 4, booktabs = TRUE, format = "latex", label = "tab1",     
      col.names = c("Tratamento", "Descrição", "Gênero", "Idade", "Renda",
                    "População", "Dens. pop.", "Compliance"),
      caption = tbl1_cap) %>% 
  kable_styling(latex_options = "HOLD_position", font_size = 10) %>% 
  kable_classic(full_width = FALSE) %>% 
  footnote(general_title = "Nota:",
           threeparttable = TRUE,
           general = atr_bal_foot) %>% 
  save_kable(file = "./Tables/table1.tex")

#' Regressoes da Tabela 7
#' 
reg_21 <- feols(resp_A~mailing+threat+appeal+info, data = data)
reg_22 <- feols(resp_A~mailing+threat+appeal+info+i_tinf+i_tapp, data = data)

delivered <- data %>% 
  filter(delivered == 1)
reg_23 <- feols(resp_B~threat+appeal+info, data = delivered)
reg_24 <- feols(resp_B~threat+appeal+info+i_tinf+i_tapp, data = delivered)
reg_25 <- feols(resp_all~threat+appeal+info, data = delivered)
reg_26 <- feols(resp_all~threat+appeal+info+i_tinf+i_tapp, data = delivered)

#' Dicionário para o nome das variáveis nas tabelas
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
#' Ajusta o estilo das tabelas
est_style = list(depvar = "title:Dep. Var.",
                 model = "title:Modelo",
                 var = "title:\\emph{Variáveis}",
                 stats = "title:\\emph{Estatísticas de diagnóstico}",
                 notes = "title:\\emph{\\medskip Notas:}")
#' Cria a Tabela 7. Table 2 in Fellner et. al
esttex(reg_21, reg_22, reg_23, reg_24, reg_25, reg_26,
       file = "./Tables/table2.tex",
       label = "tab:tab2",
       style = est_style,
       replace = TRUE,
       se = "White",
       digits = 3,
       fitstat = "",
       order = c("Correio", "^Ameaça$", "^Moral$", "^Info$", "Ameaça x Moral",
                 "Ameaça x Info", "Constante"),
       title = "Efeito do tratamento nos registros, atualizações contratuais, and resposta geral para o modelo de regressão linear.")

#' Efeitos Heterogeneos
#' Replicacao da Table C1
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
                   inc_aver age0_30 age30_60  bgld kaern noe ooe salzbg steierm 
                   tirol vlbg wien schober")
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

#' Tabela 8
esttex(het_reg_pop$lm_model, het_reg_den$lm_model, het_reg_rend$lm_model, 
       het_reg_vot$lm_model,
       file = "./Tables/tablec1.tex",
       label = "tab:tabc1",
       style = est_style,
       replace = TRUE,
       se = "White",
       digits = 3,
       fitstat = "",
       keep = c("^Ameaça$", "^Moral$", "^Info$"),
       order = c("^Ameaça$", "^Moral$", "^Info$"),
       title = "Efeito heterogêneo do tratamento. Modelo de regressão linear.")