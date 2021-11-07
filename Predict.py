import pandas as pd
import matplotlib.pyplot as plt
from data_prep import *
from eda import *
import seaborn as sns
from sklearn.model_selection import train_test_split

world_data = read_data(r"VeriSetleri\World\Trade_Map_-_List_of_exporters_for_the_selected_product_(All_products).txt")
# check_df(world_data)
world_data = world_data.iloc[:, 0:-1]

world_data = world_data.melt(id_vars=["Exporters"],
                             var_name="Year",
                             value_name="Value")
# check_df(world_data)
Test_Data = world_data.dropna(axis=0)
# check_df(Test_Data)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(Test_Data)

# for num in num_cols:
#     num_summary(Test_Data, num)


for i in num_cols:
    grab_outliers(Test_Data, i, index=True)

low, up = outlier_thresholds(Test_Data, 'Value')

df_outlier_remove = remove_outlier(Test_Data, 'Value')

df_outlier_remove = pd.get_dummies(df_outlier_remove, columns=['Exporters','Year'])
df_outlier_remove['Value'] = np.log1p(df_outlier_remove["Value"].values)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False