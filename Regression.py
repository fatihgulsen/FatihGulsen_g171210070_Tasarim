# %%
import re
from eda import *
from data_prep import *
from Model_Sum import *
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from Timer import Timer

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##################################################
# Veri Seti
##################################################
all_timer = Timer(text="{:.4f} saniyedir makine öğrenmesi yapılıyor.",name='All')
all_timer.start()
all_data = pd.read_csv(r"VeriSetleri/All_Data/import-exported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)
all_data = all_data.rename(columns=lambda lm: re.sub('[^A-Za-z0-9_]+', '_', lm))
all_data.sort_values(by=['Exporter', 'Importers'], axis=0, inplace=True)

# all_data = all_data.loc[(all_data['Exporter'] == 'Argentina') & (all_data['Importers'] == 'Costa Rica')]
# Rastgele 2 ülke seçilmiştir görselleştirme ve sonuçlar için

imp_exp_list = ['Importers', 'Exporter']
imp_exp_outer_list = [col for col in all_data.columns if col not in ['Importers', 'Exporter']]

plot_metric = [True, True]
# Model oluşturma

output_df = pd.DataFrame()
pred_array = np.array([2020, 2021], dtype='int64').reshape(-1, 1)
lineer_timer = Timer(text="{:.4f} saniyedir lineer reg yapılıyor.",name='Lineer')
poly_timer = Timer(text="{:.4f} saniyedir poly reg yapılıyor.",name='Poly')
random_timer = Timer(text="{:.4f} saniyedir random forest yapılıyor.",name='Random')
tree_timer = Timer(text="{:.4f} saniyedir Decision Tree yapılıyor.",name='Tree')
xg_timer = Timer(text="{:.4f} saniyedir XGBoost yapılıyor.",name='xg')

# %%
for row in all_data.iterrows():  # 0 pandas index olmak üzere diğerleri sıra ile gitmektedir.
    index = row[0]
    row = row[1]  # pandas serieslere dönüştürmek için kullanıldı indexi ayrı tutuyoruz

    imp_exp = row[imp_exp_list]
    xy = row[imp_exp_outer_list].reset_index().rename(columns={'index': 'Year', index: 'Value'})
    xy = label_encoder(xy, 'Year')
    xy['Year'] = xy['Year'].apply(lambda lm: lm + 2009)

    # 0 yılı 2009 olarak kabul ediyoruz ve bunları label encoder ile integer değerlere dönüştürüoyurz

    x = xy.iloc[:-1, :-1]
    y = xy.iloc[:-1, 1]  # 2020 verileri hiç eğitime katmıyoruz makine öğrenmesinde hiç görmemesi açısında

    x_2020 = xy.iloc[:, :-1].values
    y_2020 = xy.iloc[:, 1].values

    X_values = x.values
    Y_values = y.values

    # Linear Regression
    lineer_timer.start()
    lin_reg = LinearRegression(n_jobs=-1)
    lin_reg.fit(X_values, Y_values)

    predictedLinear = lin_reg.predict(X_values)
    models_sum(row, predictedLinear, X_values, Y_values, 'Linear Regression ', plot_metric)

    predictedLinear2020 = lin_reg.predict(x_2020)
    models_sum(row, predictedLinear2020, x_2020, y_2020, 'Linear Regression 2020-2021 ', plot_metric)
    lineer_timer.stop()

    sonuc_2020_2021 = lin_reg.predict(pred_array)  # SADECE 2020-2021
    sonuc_2020_2021 = np.round(sonuc_2020_2021, 0)  # Tek bir formatta görüntülemek adına ,(virgül)den sonrası yuvarlandı

    row['Lineer Reg Pred 2020'] = sonuc_2020_2021[0]
    row['Lineer Reg Pred 2021'] = sonuc_2020_2021[1]

    # Polynomial Regression
    poly_timer.start()
    poly_fea = PolynomialFeatures(degree=3)
    x_poly = poly_fea.fit_transform(X_values)
    poly_reg = LinearRegression(n_jobs=-1)
    poly_reg.fit(x_poly, Y_values)

    predictedPoly = poly_reg.predict(poly_fea.fit_transform(X_values))
    models_sum(row, predictedPoly, X_values, Y_values, 'Polynomial Regression Degree=3 ', plot_metric)

    predictedPoly2020 = poly_reg.predict(poly_fea.fit_transform(x_2020))
    models_sum(row, predictedPoly2020, x_2020, y_2020, 'Polynomial Regression Degree=3  2020-2021 ', plot_metric)
    poly_timer.stop()
    sonuc_2020_2021 = poly_reg.predict(poly_fea.fit_transform(pred_array))
    sonuc_2020_2021 = np.round(sonuc_2020_2021,
                               0)
    row['Poly Reg d3 Pred 2020'] = sonuc_2020_2021[0]
    row['Poly Reg d3 Pred 2021'] = sonuc_2020_2021[1]

    # Random Forest
    random_timer.start()
    rf_reg = RandomForestRegressor(n_estimators=15, random_state=1, n_jobs=-1)
    rf_reg.fit(X_values, Y_values.ravel())

    rf_reg_predict = rf_reg.predict(X_values)
    models_sum(row, rf_reg_predict, X_values, Y_values, 'Random Forest ', plot_metric)

    rf_reg_predict2020 = rf_reg.predict(x_2020)
    models_sum(row, rf_reg_predict2020, x_2020, y_2020, 'Random Forest 2020-2021 ', plot_metric)
    random_timer.stop()
    sonuc_2020_2021 = rf_reg.predict(pred_array)
    sonuc_2020_2021 = np.round(sonuc_2020_2021,
                               0)
    row['Random Forest Pred 2020'] = sonuc_2020_2021[0]
    row['Random Forest Pred 2021'] = sonuc_2020_2021[1]

    # Decision Tree
    tree_timer.start()
    r_dt = DecisionTreeRegressor(random_state=0)
    r_dt.fit(X_values, Y_values)

    predictedDTR = r_dt.predict(X_values)
    models_sum(row, predictedDTR, X_values, Y_values, 'Decision Tree ', plot_metric)

    predictedDTR2020 = r_dt.predict(x_2020)
    models_sum(row, predictedDTR2020, x_2020, y_2020, 'Decision Tree 2020-2021 ', plot_metric)
    if plot_metric[0]:
        tree_plot(r_dt, row)
    tree_timer.stop()

    sonuc_2020_2021 = r_dt.predict(pred_array)
    sonuc_2020_2021 = np.round(sonuc_2020_2021,
                               0)
    row['Decision Tree Pred 2020'] = sonuc_2020_2021[0]
    row['Decision Tree Pred 2021'] = sonuc_2020_2021[1]

    # XG Boost
    xg_timer.start()
    XGB = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100,n_jobs=-1)
    XGB.fit(X_values, Y_values)

    predictedXGB = XGB.predict(X_values)
    models_sum(row, predictedXGB, X_values, Y_values, 'XGBOOST ', plot_metric)

    predictedXGB2020 = XGB.predict(x_2020)
    models_sum(row, predictedXGB2020, x_2020, y_2020, 'XGBOOST 2020-2021 ', plot_metric)
    xg_timer.stop()
    sonuc_2020_2021 = XGB.predict(pred_array)
    sonuc_2020_2021 = np.round(sonuc_2020_2021,
                               0)
    row['XGBoost Pred 2020'] = sonuc_2020_2021[0]
    row['XGBoost Pred 2021'] = sonuc_2020_2021[1]

    output_df = output_df.append(row)

print('\n**********')
print('\n**********')
all_timer.stop()

print('\n**********')
lineer_timer.timer_detail()
print('\n**********')
poly_timer.timer_detail()
print('\n**********')
random_timer.timer_detail()
print('\n**********')
tree_timer.timer_detail()
print('\n**********')
xg_timer.timer_detail()
print('\n**********')
# %%
output_file = True
output_dir = 'argentina-costarica'
if output_file:
    try:
        output_df.to_csv(output_dir + '.csv', sep="\t", index=False)
        output_df.to_excel(output_dir + '.xlsx', index=False)
    except Exception as e:
        print(e)
