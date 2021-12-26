# %%
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
plt.rcParams["figure.figsize"] = [16.00, 16.00]
plt.rcParams["figure.autolayout"] = True
sns.set(rc={'figure.figsize': [16.00, 16.00]})


def bar_plot(df_catplot: pd.DataFrame, country: str, value: int, _x: str, _y: str, _col: str):
    chart = sns.catplot(x=_x, y=_y, col=_col,
                        data=df_catplot.loc[
                            ((df_catplot[_col] == country) & (df_catplot[_y] > value)
                             )], kind='bar', legend=True, legend_out=True)
    chart.set_xticklabels(rotation=90, horizontalalignment='center', fontsize=7)
    plt.tight_layout()
    plt.show()


def pie_plot(_all_data: pd.DataFrame, _x: str, _y: str, _title: str):
    colors = sns.color_palette('pastel')
    _all_data = _all_data.groupby([_x]).sum()
    _all_data.plot(kind='pie', y=_y, x=_x, autopct='%1.2f%%', colors=colors,
                   startangle=270, fontsize=17, shadow=True, explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                   legend=False,
                   title=_title)
    plt.tight_layout()
    plt.show()


def df_group_agg(df: pd.DataFrame, group: list, agg: dict):
    new_df = df.groupby(group, as_index=False).agg(agg)
    cols = new_df.columns
    new_col = [col[0] if col[1] == '' else col[0] + '_' + col[1] for col in cols]
    new_df.columns = new_col
    return new_df


# %%
##################################################
# Veri Seti import-exported-value
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

catplot_df = df_group_agg(df, ["Exporter", "Importers"], {"Value": ["sum", "mean", "median", "std"]})

exporter = df['Exporter'].unique()
sinir_Deger = [10000000, 200000000, 150000000, 500000, 15000000, 200000000]

for i, j in zip(exporter, sinir_Deger):
    bar_plot(df_catplot=catplot_df, country=i, value=j, _x='Importers', _y='Value_sum', _col='Exporter')

#######################
# Elimizdeki tüm verilerin ticaret hacmi oranı
pie_plot(_all_data=df, _x='Exporter', _y='Value', _title='Ticaret Hacimleri Oranı')
############################################################################################
############################################################################################
############################################################################################
# %%

##################################################
# Veri Seti export-imported-value
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/export-imported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
col = all_data.columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(all_data)
check_df(all_data)

# profile = pp.ProfileReport(all_data)
# profile.to_file('profile.html')

# Corr
sns.heatmap(all_data.corr(), annot=True)
plt.show()

# Bütün yılları tek kolona aldık
df = all_data.melt(id_vars=["Exporters", 'Importer'],
                   var_name="Year",
                   value_name="Value")

df.sort_values(by=['Exporters', 'Importer', 'Year'], axis=0, inplace=True)
df = df.loc[df['Exporters'] != 'World']  # Bütün verilerin toplamı bir daha olduğu için atıyoruz
df.reset_index(drop=True, inplace=True)

# profile = pp.ProfileReport(df)
# profile.to_file('profile2.html')

check_df(df)

# Ticaret value dağılımı nasıl ?
df[['Value']].describe().T

# Importer ülke sayısı ve ülkeler
df[['Importer']].nunique()
df['Importer'].unique()

# Exporters ülke sayısı ve ülkeler
df[['Exporters']].nunique()
df['Exporters'].unique()

# Exporters yılların yoplam-ortalama-medyan ve standart sapma değerleri
df.groupby(["Exporters"]).agg({"Value": ["sum", "mean", "median", "std"]})
df.groupby(["Importer"]).agg({"Value": ["sum", "mean", "median", "std"]})

catplot_df = df_group_agg(df, ["Importer", "Exporters"], {"Value": ["sum", "mean", "median", "std"]})

importer = df['Importer'].unique()
sinir_Deger = [200000000, 150000000, 20000000, 210000000, 10000000, 500000]

for i, j in zip(importer, sinir_Deger):
    bar_plot(df_catplot=catplot_df, country=i, value=j, _x='Exporters', _y='Value_sum', _col='Importer')

#######################
# Elimizdeki tüm verilerin ticaret hacmi oranı
pie_plot(_all_data=df, _x='Importer', _y='Value', _title='Ticaret Hacimleri Oranı')
############################################################################################
############################################################################################
############################################################################################
# %%
##################################################
# Veri Seti exporter-product
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/exporter-product.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
col = all_data.columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(all_data)
check_df(all_data)

# profile = pp.ProfileReport(all_data)
# profile.to_file('profile.html')

# Corr
sns.heatmap(all_data.corr(), annot=True)
plt.show()

# Bütün yılları tek kolona aldık
df = all_data.melt(id_vars=["Exporter", 'Product label', 'Code'],
                   var_name="Year",
                   value_name="Value")

df.sort_values(by=['Exporter', 'Product label', 'Code', 'Year'], axis=0, inplace=True)
df = df.loc[df['Code'] != 'TOTAL']  # Bütün verilerin toplamı bir daha olduğu için atıyoruz
df.reset_index(drop=True, inplace=True)

# profile = pp.ProfileReport(df)
# profile.to_file('profile2.html')

check_df(df)

# Ticaret value dağılımı nasıl ?
df[['Value']].describe().T

# Exporter ülke sayısı ve ülkeler
df[['Exporter']].nunique()
df['Exporter'].unique()

# Ürün sayısı ve çeşitleri
df[['Product label']].nunique()
df['Product label'].unique()

# Ürün sayısı ve kodlar
df[['Code']].nunique()
df['Code'].unique()

# Exporter ve ürünlerin   yoplam-ortalama-medyan ve standart sapma değerleri
df.groupby(["Exporter"]).agg({"Value": ["sum", "mean", "median", "std"]})
df.groupby(["Code"]).agg({"Value": ["sum", "mean", "median", "std"]})

catplot_df = df_group_agg(df, ["Exporter", 'Product label'], {"Value": ["sum", "mean", "median", "std"]})

exporter = df['Exporter'].unique()
sinir_Deger = [10000000, 200000000, 150000000, 500000, 15000000, 150000000]

for i, j in zip(exporter, sinir_Deger):
    bar_plot(df_catplot=catplot_df, country=i, value=j, _x='Product label', _y='Value_sum', _col='Exporter')

#######################
# Elimizdeki tüm verilerin ticaret hacmi oranı
pie_plot(_all_data=df, _x='Exporter', _y='Value', _title='Ticaret Hacimleri Oranı')


# %%
##################################################
# Veri Seti importer-product
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/importer-product.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
col = all_data.columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(all_data)
check_df(all_data)

# profile = pp.ProfileReport(all_data)
# profile.to_file('profile.html')

# Corr
sns.heatmap(all_data.corr(), annot=True)
plt.show()

# Bütün yılları tek kolona aldık
df = all_data.melt(id_vars=["Importer", 'Product label', 'Code'],
                   var_name="Year",
                   value_name="Value")

df.sort_values(by=['Importer', 'Product label', 'Code', 'Year'], axis=0, inplace=True)
df = df.loc[df['Code'] != 'TOTAL']  # Bütün verilerin toplamı bir daha olduğu için atıyoruz
df.reset_index(drop=True, inplace=True)

# profile = pp.ProfileReport(df)
# profile.to_file('profile2.html')

check_df(df)

# Ticaret value dağılımı nasıl ?
df[['Value']].describe().T

# Exporter ülke sayısı ve ülkeler
df[['Importer']].nunique()
df['Importer'].unique()

# Ürün sayısı ve çeşitleri
df[['Product label']].nunique()
df['Product label'].unique()

# Ürün sayısı ve kodlar
df[['Code']].nunique()
df['Code'].unique()

# Exporter ve ürünlerin   yoplam-ortalama-medyan ve standart sapma değerleri
df.groupby(["Importer"]).agg({"Value": ["sum", "mean", "median", "std"]})
df.groupby(["Code"]).agg({"Value": ["sum", "mean", "median", "std"]})

catplot_df = df_group_agg(df, ["Exporter", 'Product label'], {"Value": ["sum", "mean", "median", "std"]})

exporter = df['Exporter'].unique()
sinir_Deger = [10000000, 200000000, 150000000, 500000, 15000000, 150000000]

for i, j in zip(exporter, sinir_Deger):
    bar_plot(df_catplot=catplot_df, country=i, value=j, _x='Product label', _y='Value_sum', _col='Exporter')

#######################
# Elimizdeki tüm verilerin ticaret hacmi oranı
pie_plot(_all_data=df, _x='Exporter', _y='Value', _title='Ticaret Hacimleri Oranı')
