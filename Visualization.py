import pandas as pd
import matplotlib.pyplot as plt
from data_prep import *
from eda import *
import seaborn as sns
from sklearn.model_selection import train_test_split

world_data = pd.read_csv(r"VeriSetleri\World\Trade_Map_-_List_of_exporters_for_the_selected_product_(All_products).txt",
                         sep='\t')

world_data = world_data.iloc[:, 0:-1]
data_visual(world_data, 'World')
line_plt_compare_country(world_data, ['Turkey', 'China', 'Germany', 'Argentina', 'Senegal'],
                         ['r', 'g', 'm', 'b', 'y'])