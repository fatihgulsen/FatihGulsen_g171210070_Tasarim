import pandas as pd
import matplotlib.pyplot as plt
from data_prep import  *
from eda import *


world_data = pd.read_csv(r"VeriSetleri\World\Trade_Map_-_List_of_exporters_for_the_selected_product_(All_products).txt",sep='\t',index_col='Exporters')
check_df(world_data)
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
    'Exported value in 2020': '2020'},inplace=True)
world_data = world_data.iloc[:,0:-1] 
check_df(world_data)
world_data = world_data.T
Test_Data = world_data.dropna(axis=1)


check_df(Test_Data)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(Test_Data)

for num in num_cols:
    num_summary(Test_Data,num)


Test_Data.World.plot(color='b',label='World',legend=True) #plot World column

plt.show()


Test_Data.Turkey.plot(color='r',label='Turkey') #plot Turkey column
Test_Data.China.plot(color='g',label='China') #plot China Column
Test_Data.Germany.plot(color='m',label='Germany') #plot Germany Column
Test_Data.Argentina.plot(color='b',label='Argentina') #plot Argentina Column
Test_Data.Senegal.plot(color='y',label='Argentina') #plot Argentina Column

plt.xlabel("Year")
plt.ylabel("Export Value")
plt.legend()
plt.show()

