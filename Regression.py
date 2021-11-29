import re
import pandas as pd
import numpy as np
from eda import *
from data_prep import *
import warnings
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##################################################
# Veri Seti
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/import-exported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
all_data = all_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

# check_df(all_data)
# cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(all_data)
#
# for i in cat_cols:
#     cat_summary(all_data, i, plot=True)
#
# for i in num_cols:
#     num_summary(all_data, i, plot=False)

# plt.rcParams["figure.figsize"] = (20, 15)
# sns.heatmap(all_data.corr(), annot=True)
# plt.show()


imp_exp_list = ['Importers', 'Exporter']
imp_exp_outer_list = [col for col in all_data.columns if col not in ['Importers', 'Exporter']]

# Model oluşturma

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

for row in all_data.iterrows():  # 0 pandas index olmak üzere diğerleri sıra ile gitmektedir.
    index = row[0]
    row = row[1]  # pandas serieslere dönüştürmek için kullanıldı indexi ayrı tutuyoruz
    imp_exp = row[imp_exp_list]
    xy = row[imp_exp_outer_list].reset_index().rename(columns={'index': 'Year', index: 'Value'})
    xy = label_encoder(xy, 'Year')
    xy['Year'] = xy['Year'].apply(lambda x: x+2009)
    # 0 yılı 2009 olarak kabul ediyoruz ve bunları label encoder ile integer değerlere dönüştürüoyurz

    x = xy.iloc[:-1, :-1]
    y = xy.iloc[:-1, 1]  # 2020 verileri hiç eğitime katmıyoruz makine öğrenmesinde hiç görmemesi açısında

    x_2020 = xy.iloc[:, :-1].values
    y_2020 = xy.iloc[:, 1].values

    X_values = x.values
    Y_values = y.values

    lin_reg = LinearRegression(n_jobs=-1)
    lin_reg.fit(X_values, Y_values)
    predictedLinear = lin_reg.predict(X_values)

    plt.scatter(X_values, Y_values, color='red')
    plt.plot(X_values, predictedLinear, color='blue')
    plt.title(f'Year-Values Lin-reg \nImporter : {row["Importers"]} \n Exporter : {row["Exporter"]}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()
    print("Linear R2 degeri:")
    print(r2_score(Y_values, predictedLinear))

    predictedLinear2020 = lin_reg.predict(x_2020)
    plt.scatter(x_2020, y_2020, color='red')
    plt.plot(x_2020, predictedLinear2020, color='blue')
    plt.title(f'Year-Values Lin-reg \nImporter : {row["Importers"]} \n Exporter : {row["Exporter"]}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()
    print("Linear R2 degeri:")
    print(r2_score(y_2020, predictedLinear2020))

    lin_reg.predict(np.array(2020).reshape(-1, 1)) # SADECE 2020
    #TODO bunları ana pandas dataframe e ekle gözle karşılaştırma yapılsın
    #TODO eklendikten sonra her bir algoritma ile tek tek çıktılarını al

    print('------- IMPORTANCE -------')
    importance = lin_reg.coef_[0]
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    print('-------------------------')
