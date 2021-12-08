import time
import numpy as np
import pandas as pd
# pip install lightgbm
import lightgbm as lgb
import warnings
from eda import *
from data_prep import *
import pandas_profiling as pp

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##################################################
# Veri Seti
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/import-exported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)


df = all_data.melt(id_vars=["Exporter", 'Importers'],
                   var_name="Year",
                   value_name="Value")
df['id'] = np.nan
df.loc[(df["Year"] == 'Exported value in 2020'), 'id':] = np.arange(
    len(df.loc[(df["Year"] == 'Exported value in 2020'), :]))
# 2020 yıllarına id atama yapıldı

#####################################################
# Random Noise
#####################################################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


#####################################################
# Lag/Shifted Features
#####################################################

df.sort_values(by=['Exporter', 'Importers', 'Year'], axis=0, inplace=True)

# check_df(df)
df["Value"].head(10)
df["Value"].shift(1).values[0:10]

pd.DataFrame({"Value": df["Value"].values[0:10],
              "lag1": df["Value"].shift(1).values[0:10],
              "lag2": df["Value"].shift(2).values[0:10],
              "lag3": df["Value"].shift(3).values[0:10],
              "lag4": df["Value"].shift(4).values[0:10]})

df.groupby(["Exporter", "Importers"])['Value'].head()

df.groupby(["Exporter", "Importers"])['Value'].transform(lambda x: x.shift(1))


def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe['values_lag_' + str(lag)] = dataframe.groupby(["Exporter", "Importers"])['Value'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [1, 2, 3, 4, 5, 6, 8, 9, 10])

df.head()

#####################################################
# Rolling Mean Features
#####################################################

# Hareketli Ortalamalar

df["Value"].head(10)
df["Value"].rolling(window=2).mean().values[0:10]
df["Value"].rolling(window=3).mean().values[0:10]
df["Value"].rolling(window=5).mean().values[0:10]

a = pd.DataFrame({"sales": df["Value"].values[0:10],
                  "roll2": df["Value"].rolling(window=2).mean().values[0:10],
                  "roll3": df["Value"].rolling(window=3).mean().values[0:10],
                  "roll5": df["Value"].rolling(window=5).mean().values[0:10]})

a = pd.DataFrame({"sales": df["Value"].values[0:10],
                  "roll2": df["Value"].shift(1).rolling(window=2).mean().values[0:10],
                  "roll3": df["Value"].shift(1).rolling(window=3).mean().values[0:10],
                  "roll5": df["Value"].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_features(dataframe, windows):
    dataframe = dataframe.copy()
    for window in windows:
        dataframe['values_roll_mean_' + str(window)] = dataframe.groupby(["Exporter", "Importers"])['Value']. \
                                                           transform(
            lambda x: x.shift(1).rolling(window=window).mean()) + random_noise(dataframe)
    return dataframe


df = roll_mean_features(df, [1, 2, 3, 5, 10])

df.head()

#####################################################
# Exponentially Weighted Mean Features
#####################################################


pd.DataFrame({"Value": df["Value"].values[0:10],
              "roll2": df["Value"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["Value"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["Value"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["Value"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["Value"].shift(1).ewm(alpha=0.1).mean().values[0:10]})


def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['values_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["Exporter", "Importers"])['Value']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [1, 2, 3, 4, 5, 6, 8, 9, 10]

df = ewm_features(df, alphas, lags)

#####################################################
# One-Hot Encoding
#####################################################

df = pd.get_dummies(df, columns=['Exporter', 'Importers', 'Year'])


#####################################################
# Converting values to log(1+values)
#####################################################

# df['Value'] = np.log1p(df["Value"].values)


#####################################################
# Custom Cost Function
#####################################################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#####################################################
# MODEL VALIDATION
#####################################################

# Light GBM: optimizasyon 2 açıdan ele alınmalı.

#####################################################
# Time-Based Validation Sets
#####################################################

import re

df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
# Özel karakter dışındakileri kaldırdık


# 2018'e kadar train seti
train = df.loc[~((df["Year_Exported_value_in_2019"] == 1) | (df["Year_Exported_value_in_2020"] == 1)), :]

# 2019 yılı validation seti
val = df.loc[(df["Year_Exported_value_in_2019"] == 1), :]

df.columns

cols = [col for col in df.columns if col not in ["Value", 'id']]

Y_train = train['Value']
X_train = train[cols]

Y_val = val['Value']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

####################################################
# LightGBM Model
#####################################################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': -1,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error
# learning_rate: shrinkage_rate, eta
# num_boost_round: n_estimators, number of boosting iterations.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols, free_raw_data=False)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols,
                     free_raw_data=False)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))


##########################################
# Değişken önem düzeyleri
##########################################

def plot_lgb_importances(model, plot=False, num=10):
    from matplotlib import pyplot as plt
    import seaborn as sns
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)

plot_lgb_importances(model, plot=True, num=30)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

##########################################
# Final Model
##########################################


train = df.loc[~(df["Year_Exported_value_in_2020"] == 1), :]
Y_train = train['Value']
X_train = train[cols]

test = df.loc[(df["Year_Exported_value_in_2020"] == 1), :]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.2,
              'feature_fraction': 0.8,
              'max_depth': 6,
              'verbose': -1,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'Value']]
submission_df['Pred_Value'] = test_preds
submission_df['id'] = submission_df.id.astype(int)
# submission_df.to_csv('submission.csv', index=False)

