# Roda metodos de ML para inferencia causal
# Dados do paper: Fellner et. al. 2003

# importa bibliotecas necessarias
# import os
# import urllib.request
import pickle
import numpy as np
import pandas as pd 

# Regressao linear e IV
import statsmodels.api as sm
from linearmodels import IV2SLS

# Modelos de ML
import lightgbm as lgb 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

# EconML
from econml.ortho_iv import DMLATEIV, IntentToTreatDRIV
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import ForestDML, DML, SparseLinearDML
from econml.causal_forest import CausalForest

# Graficos
import matplotlib.pyplot as plt

# Carregando os dados
data = pd.read_stata("data_final.dta").sort_values("treatment")

################################################################
# Ignorando o atrito e estimando os efeitos apenas para 
# delivered == 1
################################################################
deliv = data[data["delivered"]!=0].fillna(0)
# Tratamentos
t1_deliv = deliv[deliv["treatment"].isin([0,1])]
t2_deliv = deliv[deliv["treatment"].isin([0,2])]
t3_deliv = deliv[deliv["treatment"].isin([0,3])]
t5_deliv = deliv[deliv["treatment"].isin([0,5])]

# Variaveis de interesse
Y1 = t1_deliv["resp_A"].values
T1 = t1_deliv["delivered"].values
X_cols = ["gender", "pop_density2005", 
    "compliance", "compliance_t", "vo_r", "vo_cr", "vo_cl", "vo_l", 
    "inc_aver", "edu_aver", "edu_lo", "edu_mi", "edu_hi", 
    "age_aver", "age0_30", "age30_60", "nat_A", "nat_EU", "nat_nonEU"]
X1 = t1_deliv[X_cols].values
X1_treat=X1[T1 == 1]
X1_eval = (t1_deliv[X_cols]
    .describe()
    .loc[["mean", "min", "25%", "50%", "75%", "max"]]
)

Y2 = t2_deliv["resp_A"].values
T2 = t2_deliv["delivered"].values
X2 = t2_deliv[X_cols].values
X2_treat=X2[T2 == 1]
X2_eval = (t2_deliv[X_cols]
    .describe()
    .loc[["mean", "min", "25%", "50%", "75%", "max"]]
)

Y3 = t3_deliv["resp_A"].values
T3 = t3_deliv["delivered"].values
X3 = t3_deliv[X_cols].values
X3_treat=X3[T3 == 1]
X3_eval = (t3_deliv[X_cols]
    .describe()
    .loc[["mean", "min", "25%", "50%", "75%", "max"]]
)

Y5 = t5_deliv["resp_A"].values
T5 = t5_deliv["delivered"].values
X5 = t5_deliv[X_cols].values
X5_treat=X5[T5 == 1]
X5_eval = (t5_deliv[X_cols]
    .describe()
    .loc[["mean", "min", "25%", "50%", "75%", "max"]]
)
# ForestDML() = CausalForest()??
# ForestDML eh muito mais rapido que CausalForest com resultados 
# semelhantes para a interpretacao via arvore

# DML com regressao logistica para E[T|X,W] e Floresta Aleatoria para
# E[Y|X,W]. Modelo final para Theta(X) eh escolhido por floresta
dml=ForestDML(
    model_t=LogisticRegression(),
    model_y=RandomForestRegressor(),
    discrete_treatment=True,
    n_estimators=1000,
    subsample_fr=0.7,
    min_samples_leaf=20,
    n_crossfit_splits=3,
    n_jobs=-1
)
# DML para T1
dml.fit(Y1, T1, X1, inference='auto')
dml1_eff=dml.effect(X1, T0=0, T1=1)
dml1_eff_treat=dml.effect(X1_treat, T0=0, T1=1)
print(f"ATE T1 por DML: {np.mean(dml1_eff)}")
dml1_inf=dml.effect_inference(X1_eval.values)
dml1_summary=dml1_inf.summary_frame(alpha=0.05)[["point_estimate", "stderr"]]
dml1_summary.index=X1_eval.index
dml1_summary=dml1_summary.stack()
dml1_summary.name="Correio"

# DML para T2
dml.fit(Y2, T2, X2, inference='auto')
dml2_eff=dml.effect(X2, T0=0, T1=1)
dml2_eff_treat=dml.effect(X2_treat, T0=0, T1=1)
print(f"ATE T2 por DML: {np.mean(dml2_eff)}")
dml2_inf=dml.effect_inference(X2_eval.values)
dml2_summary=dml2_inf.summary_frame(alpha=0.05)[["point_estimate", "stderr"]]
dml2_summary.index=X2_eval.index
dml2_summary=dml2_summary.stack()
dml2_summary.name="Ameaça"

# Interpretacao por arvore de decisao para T2
interp.interpret(dml, X)
fig, ax1 = plt.subplots(figsize=(25,6))
interp.plot(feature_names=X_cols, fontsize=12, ax=ax1)
fig.savefig("Figs/fig_tree_dml.png")

# DML para T3
dml.fit(Y3, T3, X3, inference='auto')
dml3_eff=dml.effect(X3, T0=0, T1=1)
dml3_eff_treat=dml.effect(X3_treat, T0=0, T1=1)
print(f"ATE T3 por DML: {np.mean(dml3_eff)}")
dml3_inf=dml.effect_inference(X3_eval.values)
dml3_summary=dml3_inf.summary_frame(alpha=0.05)[["point_estimate", "stderr"]]
dml3_summary.index=X3_eval.index
dml3_summary=dml3_summary.stack()
dml3_summary.name="Info"

# DML para T5Y5 = t5_deliv["resp_A"].values
dml.fit(Y5, T5, X5, inference='auto')
dml5_eff=dml.effect(X5, T0=0, T1=1)
dml5_eff_treat=dml.effect(X5_treat, T0=0, T1=1)
print(f"ATE T5 por DML: {np.mean(dml5_eff)}")
dml5_inf=dml.effect_inference(X5_eval.values)
dml5_summary=dml5_inf.summary_frame(alpha=0.05)[["point_estimate", "stderr"]]
dml5_summary.index=X5_eval.index
dml5_summary=dml5_summary.stack()
dml5_summary.name="Moral"

# ATE
dml_ate=[np.mean(x) for x in [dml1_eff, dml2_eff, dml3_eff, dml5_eff]]
# ATT
dml_att=[np.mean(x) for x in 
    [dml1_eff_treat, dml2_eff_treat, dml3_eff_treat, dml5_eff_treat]]

# ATE mais alto que ATT? Como explicar isso sem recorrer a spillover?
# Sumario com os resultados para os 4 tratamentos
dml_summary=(
    pd.concat(
        [dml1_summary, dml2_summary, dml3_summary, dml5_summary],
        axis=1)
    .reset_index()
    .rename(columns={"level_0": "X", "level_1": "Estatística"})
)

dml_summary.to_latex(
    buf="Tables/tab_dml_summary.tex",
    decimal=",",
    caption="Efeitos heterogêneos do tratamento estimados por Double Machine Learning.",
    label="tab:dml-summary",
    index=False
)
# Nota: os estágios de previsão foram floresta aleatória para E[Y|X]
# e regressão logística para E[T|X]. O modelo final para o efeito 
# condicional do tratamento, $\theta(X)$, é uma floresta aleatória.

################################################################
# Metodos com variaveis Instrumentais
################################################################
# Apenas tratamento T1
t1_data = data[data["treatment"].isin([0,1])]
# Controles nao recebem cartas
t1_data["delivered"] = (t1_data["delivered"]
    .fillna(0)
    .astype("int8"))

# Definindo as variaveis
Z = t1_data["treatment"]
T = t1_data["delivered"]
Y = t1_data["resp_A"]
X = t1_data[X_cols]

# Modelos para E[Y|X] e E[T|Z,X]
lgb_YX_par = {
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 30,
    "max_depth": 5
}

lgb_TXZ_par = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 30,
    "max_depth": 5
}

lgb_theta_par = {
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 30,
    "max_depth": 3
}

modelTXZ = lgb.LGBMClassifier(**lgb_TXZ_par)
modelYX = lgb.LGBMRegressor(**lgb_YX_par)
modelZX = lgb.LGBMClassifier(**lgb_TXZ_par)
# Modelo inicial para o efeito heterogeneo
# Sera melhorado pelo algoritmo Double Robust IV
pre_theta = lgb.LGBMRegressor(**lgb_theta_par)

## Modelo 2-stages Least Squares
T_sm = sm.add_constant(T)
model_2sls = IV2SLS(Y, exog=T_sm["const"], endog=T, instruments=Z)
iv_fit = model_2sls.fit()
iv_fit.summary

## Modelo DMLATEIV
model_dml = DMLATEIV(
    model_Y_W=modelYX,
    model_T_W=modelTXZ,
    model_Z_W=modelZX,
    discrete_treatment=True,
    discrete_instrument=True,
    n_splits=5
)
model_dml.fit(Y, T, Z, W=None, inference="bootstrap")
model_dml.const_marginal_effect_interval(alpha=0.05)

## Modelo DRIV
# Treina o modelo Linear DRIV. O proprio Theta(X) eh uma
# Projecao linear em X!
model_ldriv = LinearIntentToTreatDRIV(
    model_Y_X=modelYX,
    model_T_XZ=modelTXZ,
    flexible_model_effect=pre_theta,
    n_splits=5,
    featurizer=None #PolynomialFeatures(degree=1, include_bias=False)
)
model_ldriv.fit(Y, T, Z=Z, X=X, inference="statsmodels")
# Resultado do ajuste
model_ldriv.summary(feat_name=X.columns)

# Treina o modelo DRIV mais flexivel. Theta(X) pode ser um modelo
# flexivel (nao parametrico, ie. floresta aleatoria) de X
# ATENCAO: leva bastante tempo para rodar
model_driv = IntentToTreatDRIV(
    model_Y_X=modelYX,
    model_T_XZ=modelTXZ,
    flexible_model_effect=pre_theta,
    n_splits=5,
    featurizer=None #PolynomialFeatures(degree=1, include_bias=False)
)
model_driv.fit(Y, T, Z=Z, X=X, inference="bootstrap")
# Resultado do ajuste. X_mean deve ser um dataframe
X_eval = X.describe().loc[["mean", "min", "25%", "50%", "75%", "max"]]
driv_effect = model_driv.effect_interval(X=X_eval, alpha=0.05)
driv_dict = {"LB": driv_effect[0], "UB": driv_effect[1]}
driv_df=pd.DataFrame(driv_dict, index=X_eval.index)

# Interpretacao causal por arvore de decisao
interp = SingleTreeCateInterpreter(
    include_model_uncertainty=False, 
    max_depth=3, min_samples_leaf=10)
interp.interpret(model_driv, X)
fig, ax1 = plt.subplots(figsize=(25,6))
interp.plot(feature_names=X.columns, fontsize=12, ax=ax1)
fig.savefig("Figs/fig_tree_driv.png")


# Salva objetos 
with open("modelos.pkl", "wb") as f:
    pickle.dump([model_dml, model_ldriv, model_driv], f)

# Carrega objetos
# with open("modelos.pkl", "wb") as f:
#     model_dml, model_ldriv, model_driv = pickle.load(f)


# Floresta Causal
# cf=CausalForest(
#     n_trees=200,
#     min_leaf_size=15,
#     max_depth=8,
#     model_T=LogisticRegression(),
#     model_Y=RandomForestRegressor(),
#     discrete_treatment=True,
#     random_state=123
# )
# cf.fit(Y, T, X)
# cf_inf=cf.effect_inference(X_eval.values)
# cf_const_eff=cf.const_marginal_effect(X)
# print(f"ATE por Causal Forest {np.mean(cf_const_eff)}")
# # cf_eff=cf.effect(X_eval.values)
# # cf_lb, cf_ub=cf.effect_interval(X_eval.values)
# # cf_dict={"Estimate": cf_eff, "LB": cf_lb, "UB": cf_ub}
# # cf_df=pd.DataFrame(cf_dict, index=X_eval.index)
# cf_summary=cf_inf.summary_frame(alpha=0.05)
# cf_summary.index=X_eval.index
# cf_summary
# cf_summary.to_latex(
#     buf="Tables/tab_cf_summary.tex",
#     decimal=",",
#     caption="Efeitos heterogêneos do tratamento T1 estimados por Floresta Causal.",
#     label="tab:cf-summary"
# )
# # Floresta causal nao eh muito robusta a especificacao do modelo
# # E[Y|X,W] (Random Forest ou Lasso)

# # Interpretacao causal por arvore de decisao
# interp = SingleTreeCateInterpreter(
#     include_model_uncertainty=False, 
#     max_depth=3, min_samples_leaf=10)
# interp.interpret(cf, X)
# fig, ax1 = plt.subplots(figsize=(25,6))
# interp.plot(feature_names=X_cols, fontsize=12, ax=ax1)
# fig.savefig("Figs/fig_tree_cf.png")

