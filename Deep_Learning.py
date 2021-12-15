import numpy as np
import pandas as pd
import warnings
from eda import *
from data_prep import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##################################################
# Veri Seti
##################################################
all_data = pd.read_csv(r"VeriSetleri/All_Data/import-exported-value.csv", sep='\t')
all_data.replace({'0': np.nan, 0: np.nan}, inplace=True)
all_data = all_data.dropna(axis=0)

df = all_data.melt(id_vars=["Exporter", 'Importers'],
                   var_name="Year",
                   value_name="Value")

x_df, y_df = df.iloc[:, :-1], df.iloc[:, -1:]

x_df = label_encoder(x_df, 'Year')
x_df = one_hot_encoder(x_df, ['Exporter', 'Importers'], True)

x_df_values = x_df.to_numpy()
y_df_values = y_df.to_numpy()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(GRU(128, return_sequences=True, input_shape=(1, x_df_values.shape[1])))
model.add(GRU(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


epochs = 40
early_stop = EarlyStopping(monitor='val_loss',patience=10)
ckpt = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_loss', verbose=1)
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=1,
    validation_data=(x_validation, y_validation),
    callbacks=[early_stop, ckpt])