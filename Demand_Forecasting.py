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
world_data = read_data(r"VeriSetleri\World\Trade_Map_-_List_of_exporters_for_the_selected_product_(All_products).txt")
world_data = world_data.iloc[:, 0:-1]
world_data = world_data.dropna(axis=0)

df = world_data.melt(id_vars=["Exporters"],
                             var_name="Year",
                             value_name="Value")

#################################
# EDA
#################################
profile = pp.ProfileReport(df)
profile.to_file('profile.html')
check_df(df)

world_data.corr()

# Ticaret value dağılımı nasıl ?
df[['Value']].describe().T

# Exporters ülke sayısı ve ülkeler
df[['Exporters']].nunique()
df['Exporters'].unique()

# Exporters yılların yoplam-ortalama-medyan ve standart sapma değerleri
df.groupby(["Exporters"]).agg({"Value": ["sum", "mean", "median", "std"]})

#####################################################
# Lag/Shifted Features
#####################################################

pd.DataFrame({"Value": df["Value"].values[0:10],
              "lag1": df["Value"].shift(1).values[0:10],
              "lag2": df["Value"].shift(2).values[0:10],
              "lag3": df["Value"].shift(3).values[0:10],
              "lag4": df["Value"].shift(4).values[0:10]})