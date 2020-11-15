library(kableExtra)
library(readr)
library(dplyr)

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

