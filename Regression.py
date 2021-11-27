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

check_df(all_data)
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(all_data)

for i in cat_cols:
    cat_summary(all_data, i, plot=True)

for i in num_cols:
    num_summary(all_data, i, plot=False)

plt.rcParams["figure.figsize"] = (20, 15)
sns.heatmap(all_data.corr(), annot=True)
plt.show()


importers = all_data.iloc[:, :1]
x = all_data.iloc[:, 1:-2]
y = all_data.iloc[:,-2:-1]
exporter = all_data.iloc[:, -1:]

columns = x.columns

# Model oluşturma

X_values = x.values
Y_values = y.values

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lin_reg = LinearRegression()
lin_reg.fit(X_values, Y_values)
predictedLinear = lin_reg.predict((X_values))
print("Linear R2 degeri:")
print(r2_score(Y_values, predictedLinear))

print('------- IMPORTANCE -------')
importance = lin_reg.coef_[0]
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
print('-------------------------')