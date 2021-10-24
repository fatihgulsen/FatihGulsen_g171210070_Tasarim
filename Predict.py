import pandas as pd
import matplotlib.pyplot as plt
from data_prep import *
from eda import *
import seaborn as sns

world_data = pd.read_csv(r"VeriSetleri\World\Trade_Map_-_List_of_exporters_for_the_selected_product_(All_products).txt",sep='\t')
# check_df(world_data)
world_data.rename(columns={
    'Exported value in 2009': '2009',
    'Exported value in 2010': '2010',
    'Exported value in 2011': '2011',
    'Exported value in 2012': '2012',
    'Exported value in 2013': '2013',
    'Exported value in 2014': '2014',
    'Exported value in 2015': '2015',
    'Exported value in 2016': '2016',
    'Exported value in 2017': '2017',
    'Exported value in 2018': '2018',
    'Exported value in 2019': '2019',
    'Exported value in 2020': '2020'}, inplace=True)
world_data = world_data.iloc[:, 0:-1]
# check_df(world_data)
Test_Data = world_data.dropna(axis=0)
# check_df(Test_Data)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(Test_Data)

# for num in num_cols:
#     num_summary(Test_Data, num)

data_visual(Test_Data, 'World')
line_plt_compare_country(Test_Data, ['Turkey', 'China', 'Germany', 'Argentina', 'Senegal'],
                         ['r', 'g', 'm', 'b', 'y'])

for i in num_cols:
    grab_outliers(Test_Data, i, index=True)


low, up = outlier_thresholds(Test_Data, '2019')

df_outlier_remove = remove_outlier(Test_Data, '2019')
