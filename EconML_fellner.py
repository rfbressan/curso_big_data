# Roda metodos de ML para inferencia causal
# Dados do paper: Fellner et. al. 2003

# importa bibliotecas necessarias
import os
import urllib.request
import numpy as np
import pandas as pd 

# Regressao linear e IV
import statsmodels.api as sm
from linearmodels import IV2SLS

# Modelos de ML
import lightgbm as lgb 
from sklearn.preprocessing import PolynomialFeatures

# EconML
from econml.ortho_iv import LinearIntentToTreatDRIV, \
    DMLATEIV
from econml.cate_interpreter import SingleTreeCateInterpreter

# Graficos
import matplotlib.pyplot as plt 
%matplotlib inline

# Carregando os dados
data = pd.read_stata("data_final.dta").sort_values("treatment")
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
X_cols = ["gender", "pop_density2005", 
    "compliance", "compliance_t", "vo_r", "vo_cr", "vo_cl", "vo_l", 
    "inc_aver", "edu_aver", "edu_lo", "edu_mi", "edu_hi", 
    "age_aver", "age0_30", "age30_60", "nat_A", "nat_EU", "nat_nonEU"]
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
    discrete_instrument=True
)
model_dml.fit(Y, T, Z, W=X, inference="bootstrap")
model_dml.const_marginal_effect_interval(alpha=0.05)

## Modelo DRIV
# Treina o modelo DRIV
model_driv = LinearIntentToTreatDRIV(
    model_Y_X=modelYX,
    model_T_XZ=modelTXZ,
    flexible_model_effect=pre_theta,
    featurizer=PolynomialFeatures(degree=1, include_bias=False)
)
model_driv.fit(Y, T, Z=Z, X=X, inference="statsmodels")
# Resultado do ajuste
model_driv.summary()

# Interpretacao causal por arvore de decisao
interp = SingleTreeCateInterpreter(
    include_model_uncertainty=False, 
    max_depth=3, min_samples_leaf=10)
interp.interpret(model_driv, X)
fig, ax1 = plt.subplots(figsize=(25,6))
interp.plot(feature_names=X.columns, fontsize=12, ax=ax1)
