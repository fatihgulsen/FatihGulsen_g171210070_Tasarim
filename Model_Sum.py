import pandas as pd
import matplotlib.pyplot as plt
from data_prep import *
from eda import *
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import tree


def SMAPE(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape


def algo_scatter(row, predictions, x_values, y_values, text=''):
    plt.scatter(x_values, y_values, color='red')
    plt.grid(True)
    plt.plot(x_values, predictions, color='blue')
    plt.title(f'Year - Values {text} \nImporter : {row["Importers"]} \n Exporter : {row["Exporter"]}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()


def tree_plot(dtr_model, row, fontsize=3.8, text=''):
    plt.title(f'Year - Values {text} \nImporter : {row["Importers"]} \n Exporter : {row["Exporter"]}')
    tree.plot_tree(dtr_model, fontsize=fontsize)
    plt.show()


def metrics_score(text, predictions, y_values):
    print('\n---------------------- Score Table -------------------------')
    print(f'{text} R^2 Score: %.3f' % r2_score(y_values, predictions))
    print(f'{text} Root Mean Square Error (RMSE): %.3f' % np.sqrt(mean_squared_error(y_values, predictions)))
    print(f'{text} Mean Squared Error (MSE): %.3f ' % mean_squared_error(y_values, predictions))
    print(f'{text} Mean Absolute  Error (MAE): %.3f ' % mean_absolute_error(y_values, predictions))
    print(f'{text} Mean Absolute Percentage  Error (MAPE): %.3f ' % MAPE(y_values, predictions))
    print(f'{text} Symmetric Mean Absolute Percentage  Error (SMAPE): %.3f ' % SMAPE(y_values, predictions))
    print('---------------------------------------------------------\n')


def models_sum(row, predictions, x_values, y_values, text='', plot_metric=[True, True]):
    if plot_metric[0]:
        algo_scatter(row, predictions, x_values, y_values, text)
    if plot_metric[1]:
        metrics_score(text, predictions, y_values)


def compare_models(models, models_name, compare_metric):
    pass
