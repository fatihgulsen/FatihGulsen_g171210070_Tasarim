import time
import numpy as np
import pandas as pd
import warnings
from eda import *
from data_prep import *
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##################################################
# Veri Seti
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/import-exported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
col = all_data.columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(all_data)
check_df(all_data)

# profile = pp.ProfileReport(all_data)
# profile.to_file('profile.html')

# Corr
plt.rcParams["figure.figsize"] = (20, 15)
sns.heatmap(all_data.corr(), annot=True)
plt.show()

# Bütün yılları tek kolona aldık
df = all_data.melt(id_vars=["Exporter", 'Importers'],
                   var_name="Year",
                   value_name="Value")

df.sort_values(by=['Exporter', 'Importers', 'Year'], axis=0, inplace=True)
df = df.loc[df['Importers'] != 'World']  # Bütün verilerin toplamı bir daha olduğu için atıyoruz
df.reset_index(drop=True, inplace=True)

# profile = pp.ProfileReport(df)
# profile.to_file('profile2.html')

check_df(df)

# Ticaret value dağılımı nasıl ?
df[['Value']].describe().T

# Exporters ülke sayısı ve ülkeler
df[['Exporter']].nunique()
df['Exporter'].unique()

# Exporters ülke sayısı ve ülkeler
df[['Importers']].nunique()
df['Importers'].unique()

# Exporters yılların yoplam-ortalama-medyan ve standart sapma değerleri
df.groupby(["Exporter"]).agg({"Value": ["sum", "mean", "median", "std"]})
df.groupby(["Importers"]).agg({"Value": ["sum", "mean", "median", "std"]})

catplot_df = df.groupby(["Exporter", "Importers"], as_index=False).agg({"Value": ["sum", "mean", "median", "std"]})
cols = catplot_df.columns
new_col = [i[0] if i[1] == '' else i[0] + '_' + i[1] for i in cols]
catplot_df.columns = new_col

# Turket +20000000$
chart = sns.catplot(x='Importers', y='Value_sum', col='Exporter',
                    data=catplot_df.loc[((catplot_df['Exporter'] == 'Turkey') & (catplot_df['Value_sum'] > 20000000) & (
                            catplot_df['Importers'] != 'World'))], kind='bar')
chart.set_xticklabels(rotation=90, horizontalalignment='right')
plt.show()

# USA +150000000$
chart = sns.catplot(x='Importers', y='Value_sum', col='Exporter',
                    data=catplot_df.loc[((catplot_df['Exporter'] == 'USA') & (catplot_df['Value_sum'] > 150000000))],
                    kind='bar')
chart.set_xticklabels(rotation=90, horizontalalignment='right')
plt.show()

# CHINA +200000000$
chart = sns.catplot(x='Importers', y='Value_sum', col='Exporter',
                    data=catplot_df.loc[((catplot_df['Exporter'] == 'China') & (catplot_df['Value_sum'] > 200000000))],
                    kind='bar')
chart.set_xticklabels(rotation=90, horizontalalignment='right')
plt.show()

#######################
# Elimizdeki tüm verilerin ticaret hacmi oranı
colors = sns.color_palette('pastel')
all_trade_exporter = df.groupby(["Exporter"]).sum()
all_trade_exporter.plot(kind='pie', y='Value', x='Exporter', autopct='%1.2f%%',colors=colors,
                        startangle=270, fontsize=17, shadow=True, explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), legend=False, title='Ticaret Hacimleri Oranı')
plt.show()

#######################

