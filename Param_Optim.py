import pandas as pd
import numpy as np
import re
from Model_Sum import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


all_data = pd.read_csv(r"VeriSetleri/All_Data/import-exported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
all_data = all_data.rename(columns=lambda lm: re.sub('[^A-Za-z0-9_]+', '_', lm))
all_data.sort_values(by=['Exporter', 'Importers'], axis=0, inplace=True)

imp_exp_list = ['Importers', 'Exporter']
imp_exp_outer_list = [col for col in all_data.columns if col not in ['Importers', 'Exporter']]

# Test verileri seçimi
all_data1 = all_data.loc[(all_data['Exporter'] == 'Argentina') & (all_data['Importers'] == 'Costa Rica')]
all_data2 = all_data.loc[(all_data['Exporter'] == 'Turkey') & (all_data['Importers'] == 'Qatar')]
all_data3 = all_data.loc[(all_data['Exporter'] == 'China') & (all_data['Importers'] == 'Greece')]

row1 = all_data1.iloc[0]
row2 = all_data2.iloc[0]
row3 = all_data3.iloc[0]

imp_exp1 = row1[imp_exp_list]
xy1 = row1[imp_exp_outer_list].reset_index().rename(columns={'index': 'Year', 72: 'Value'})
xy1 = label_encoder(xy1, 'Year')
xy1['Year'] = xy1['Year'].apply(lambda lm: lm + 2009)
# 0 yılı 2009 olarak kabul ediyoruz ve bunları label encoder ile integer değerlere dönüştürüoyurz
x1 = xy1.iloc[:-1, :-1]
y1 = xy1.iloc[:-1, 1]

X1_values = x1.values
Y1_values = y1.values

imp_exp2 = row2[imp_exp_list]
xy2 = row2[imp_exp_outer_list].reset_index().rename(columns={'index': 'Year', 879: 'Value'})
xy2 = label_encoder(xy2, 'Year')
xy2['Year'] = xy2['Year'].apply(lambda lm: lm + 2009)
# 0 yılı 2009 olarak kabul ediyoruz ve bunları label encoder ile integer değerlere dönüştürüoyurz
x2 = xy2.iloc[:-1, :-1]
y2 = xy2.iloc[:-1, 1]

X2_values = x2.values
Y2_values = y2.values

imp_exp3 = row3[imp_exp_list]
xy3 = row3[imp_exp_outer_list].reset_index().rename(columns={'index': 'Year', 247: 'Value'})
xy3 = label_encoder(xy3, 'Year')
xy3['Year'] = xy3['Year'].apply(lambda lm: lm + 2009)
# 0 yılı 2009 olarak kabul ediyoruz ve bunları label encoder ile integer değerlere dönüştürüoyurz
x3 = xy3.iloc[:-1, :-1]
y3 = xy3.iloc[:-1, 1]
X3_values = x3.values
Y3_values = y3.values

############################# Random Forest ####################################
rf_param_grid = {
    'n_estimators': [15, 25, 50, 100, 200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'criterion': ['mse', 'mae']
}

rf_reg = RandomForestRegressor(random_state=1, n_jobs=-1)
CV_rfc = GridSearchCV(estimator=rf_reg, param_grid=rf_param_grid, cv=5, n_jobs=-1)

############################# DENEY 1

CV_rfc.fit(X1_values, Y1_values.ravel())

CV_rfc.best_params_
CV_rfc.best_score_

# {'criterion': 'mse',
#  'max_depth': 2,
#  'max_features': 'sqrt',
#  'n_estimators': 15}


############################# DENEY 2
CV_rfc.fit(X2_values, Y2_values.ravel())

CV_rfc.best_params_
CV_rfc.best_score_

# {'criterion': 'mse',
#  'max_depth': 2,
#  'max_features': 'auto',
#  'n_estimators': 1000}

############################# DENEY 3
CV_rfc.fit(X3_values, Y3_values.ravel())

CV_rfc.best_params_
CV_rfc.best_score_

# {'criterion': 'mse',
#  'max_depth': 2,
#  'max_features': 'sqrt',
#  'n_estimators': 25}

############################  Decision Tree ##############################

r_dt_param_grid = {'criterion': ["mse", "friedman_mse", "mae"],
                   'max_depth': np.arange(1, 21),
                   'splitter': ['best', 'random'],
                   'min_samples_split': np.arange(2, 10),
                   'min_samples_leaf': np.arange(2, 10),
                   'max_features': ['auto', 'sqrt', 'log2'],
                   'max_leaf_nodes': np.arange(2, 10)
                   }
r_dt = DecisionTreeRegressor(random_state=1)
CV_r_dt = GridSearchCV(estimator=r_dt, param_grid=r_dt_param_grid, cv=5, n_jobs=-1)

############################# DENEY 1
CV_r_dt.fit(X1_values, Y1_values.ravel())

CV_r_dt.best_params_
CV_r_dt.best_score_

# {'criterion': 'mse',
#  'max_depth': 2,
#  'max_features': 'auto',
#  'max_leaf_nodes': 3,
#  'min_samples_leaf': 2,
#  'min_samples_split': 2,
#  'splitter': 'best'}

############################# DENEY 2
CV_r_dt.fit(X2_values, Y2_values.ravel())

CV_r_dt.best_params_
CV_r_dt.best_score_

# {'criterion': 'mae',
#  'max_depth': 3,
#  'max_features': 'auto',
#  'max_leaf_nodes': 4,
#  'min_samples_leaf': 2,
#  'min_samples_split': 2,
#  'splitter': 'random'}

############################# DENEY 3
CV_r_dt.fit(X3_values, Y3_values.ravel())

CV_r_dt.best_params_
CV_r_dt.best_score_

# {'criterion': 'mae',
#  'max_depth': 1,
#  'max_features': 'auto',
#  'max_leaf_nodes': 2,
#  'min_samples_leaf': 2,
#  'min_samples_split': 2,
#  'splitter': 'random'}

############################  XG Boost ##############################

XGB_param_grid = {"learning_rate": (0.05, 0.10, 0.15, 0.20),
                  "max_depth": [3, 4, 5, 6, 8, 9, 10, 15],
                  "min_child_weight": [1, 3, 5, 7],
                  "gamma": [0.0, 0.1, 0.2],
                  "colsample_bytree": [0.3, 0.4],
                  "n_esminators": [20, 50, 100, 250]
                  }
XGB = XGBRegressor(n_jobs=-1)
CV_XGB = GridSearchCV(estimator=XGB, param_grid=XGB_param_grid, cv=5, n_jobs=-1)

############################# DENEY 1
CV_XGB.fit(X1_values, Y1_values.ravel())

CV_XGB.best_params_
CV_XGB.best_score_

# {'colsample_bytree': 0.3,
#  'gamma': 0.0,
#  'learning_rate': 0.05,
#  'max_depth': 3,
#  'min_child_weight': 3,
#  'n_esminators': 20}
############################# DENEY 2
CV_XGB.fit(X2_values, Y2_values.ravel())

CV_XGB.best_params_
CV_XGB.best_score_

# {'colsample_bytree': 0.3,
#  'gamma': 0.0,
#  'learning_rate': 0.2,
#  'max_depth': 3,
#  'min_child_weight': 3,
#  'n_esminators': 20}

############################# DENEY 3
CV_XGB.fit(X3_values, Y3_values.ravel())

CV_XGB.best_params_
CV_XGB.best_score_

# {'colsample_bytree': 0.3,
#  'gamma': 0.0,
#  'learning_rate': 0.05,
#  'max_depth': 3,
#  'min_child_weight': 1,
#  'n_esminators': 20}



