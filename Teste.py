# DML use case with continuous treatment
# import econml
import warnings
warnings.filterwarnings('ignore')

# Main imports
from econml.dml import DMLCateEstimator, LinearDMLCateEstimator, SparseLinearDMLCateEstimator, ForestDMLCateEstimator
from econml.causal_forest import CausalForest
from econml.cate_interpreter import SingleTreeCateInterpreter

import numpy as np
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, LinearRegression, MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


# Treatment effect function
def exp_te(x):
    return np.exp(2*x[0])

# DGP constants
np.random.seed(123)
n = 2000            # number of samples
n_w = 30            # number of controls (confounders)
support_size = 5    # 
n_x = 1             # number of covariates (heterogeneous)

# Outcome support
support_Y = np.random.choice(np.arange(n_w), size=support_size, replace=False)
coefs_Y = np.random.uniform(0, 1, size = support_size)
epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n) # function
# Treatment support
support_T = support_Y
coefs_T = np.random.uniform(0, 1, size=support_size)
eta_sample = lambda n: np.random.uniform(-1, 1, size=n) # function

# Generate controls, covariates, treatments and outcomes
W = np.random.normal(0, 1, size=(n, n_w))
X = np.random.uniform(0, 1, size=(n, n_x))
# Heterogeneous treatment effects
TE = np.array([exp_te(x_i) for x_i in X])
T = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
Y = TE * T + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)
# Split data into train and test samples
Y_train, Y_val, T_train, T_val, X_train, X_val, W_train, W_val = train_test_split(Y, T, X, W, test_size = .2)
X_test = np.array(list(product(np.arange(0, 1, 0.1), repeat=n_x))) # to keep the array with 2-D?

# Train estimator.
# Default: linear heterogeneous effect 
est = LinearDMLCateEstimator(model_y=RandomForestRegressor(),
                             model_t=RandomForestRegressor(),
                             random_state=123)
est.fit(Y_train, T_train, X_train, W_train)
te_pred = est.effect(X_test)

# Sparse polynomial
est1 = SparseLinearDMLCateEstimator(model_y=RandomForestRegressor(),
                                    model_t=RandomForestRegressor(),
                                    featurizer=PolynomialFeatures(degree=3),
                                    random_state=123)
est1.fit(Y_train, T_train, X_train, W_train)
te_pred1 = est1.effect(X_test)

# Sparse polynomial with final Lasso
est2 = DMLCateEstimator(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        model_final=Lasso(alpha=0.1, fit_intercept=False),
                        featurizer=PolynomialFeatures(degree=10),
                        random_state=123)
est2.fit(Y_train, T_train, X_train, W_train)
te_pred2 = est2.effect(X_test)

# Random forest final stage
# One can replace model_y and model_t with any scikit-learn regressor and classifier correspondingly
# as long as it accepts the sample_weight keyword argument at fit time.
est3 = ForestDMLCateEstimator(model_y=RandomForestRegressor(),
                              model_t=RandomForestRegressor(),
                              discrete_treatment=False,
                              n_estimators=1000,
                              subsample_fr=0.8,
                              min_samples_leaf=10,
                              min_impurity_decrease=0.001,
                              verbose=0,
                              min_weight_fraction_leaf=0.01)
est3.fit(Y_train, T_train, X_train, W_train)
te_pred3 = est3.effect(X_test)
 
# Visualization
expected_te = np.array([exp_te(x_i) for x_i in X_test])
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(X_test, te_pred, label='DML default')
ax.plot(X_test, te_pred1, label='DML poly degree 3')
ax.plot(X_test, te_pred2, label='DML poly degree 10 Lasso')
ax.plot(X_test, te_pred3, label='DML Random Forest')
ax.plot(X_test, expected_te, 'b--', label='True Effect')
ax.set_ylabel('Treatment Effect')
ax.set_xlabel('X')
ax.legend()
fig.show()

# Model selection based on MSE of final stage Y residual (godness-of-fit)
score = {}
score['DML default'] = est.score(Y_val, T_val, X_val, W_val)
score['DML poly 3'] = est1.score(Y_val, T_val, X_val, W_val)
score['DML poly Lasso'] = est2.score(Y_val, T_val, X_val, W_val)
score['DML Random Forest'] = est3.score(Y_val, T_val, X_val, W_val)
print('Best model based on MSE of Y residuals is:', min(score, key=lambda x: score.get(x)))

# Model selection based on MSE of treatment effect (Unfeasible. Do not see the true effect)
score_te = {}
score_te['DML default'] = ((expected_te - te_pred)**2).mean()
score_te['DML poly 3'] = ((expected_te - te_pred1)**2).mean()
score_te['DML poly Lasso'] = ((expected_te - te_pred2)**2).mean()
score_te['DML Random Forest'] = ((expected_te - te_pred3)**2).mean()
print('Best model based on UNFEASIBLE MSE of true TE residuals is:', min(score_te, key=lambda x: score_te.get(x)))

## Causal Forest estimation
def get_test_train_data(n, n_w, support_size, n_x, te_func, n_treatments):
    # Outcome support
    support_Y = np.random.choice(range(n_w), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    # Treatment support 
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=(support_size, n_treatments))
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n) 
    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_x))
    # Heterogeneous treatment effects
    TE = np.array([te_func(x_i, n_treatments) for x_i in X])
    log_odds = np.dot(W[:, support_T], coefs_T)
    T_sigmoid = np.exp(log_odds)
    T_sigmoid = T_sigmoid/np.sum(T_sigmoid, axis=1, keepdims=True)
    T = np.array([np.random.choice(n_treatments, p=p) for p in T_sigmoid])
    TE = np.concatenate((np.zeros((n,1)), TE), axis=1)
    Y = TE[np.arange(n), T] + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)
    X_test = np.array(list(product(np.arange(0, 1, 0.01), repeat=n_x)))

    return (Y, T, X, W), (X_test, np.array([te_func(x, n_treatments) for x in X_test]))

import scipy.special
def te_func(x, n_treatments):
    return [np.exp(2*x[0]), 3*scipy.special.expit(100*(x[0] - .5)) - 1, -2*scipy.special.expit(100*(x[0] - .25))]

np.random.seed(123)
(Y, T, X, W), (X_test, te_test) = get_test_train_data(2000, 3, 3, 1, te_func, 4)

cf=CausalForest(
    n_trees=200,
    min_leaf_size=15,
    max_depth=8,
    model_T=LogisticRegression(),
    model_Y=Lasso(),
    discrete_treatment=True,
    random_state=123
)
cf.fit(Y, T, X)

# Interpretacao causal por arvore de decisao
# ERRO: so eh possivel para tratamentos de 1 dimensao
interp = SingleTreeCateInterpreter(
    include_model_uncertainty=False, 
    max_depth=3, min_samples_leaf=10)
interp.interpret(cf, X)
fig, ax1 = plt.subplots(figsize=(25,6))
interp.plot(feature_names=X.columns, fontsize=12, ax=ax1)