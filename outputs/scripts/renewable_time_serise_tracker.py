#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Huber
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# In[133]:


df = pd.read_csv("outputs/preprocessed_data/data_cleaned.csv")


# In[134]:


df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
df.set_index('DATE', inplace=True)


# In[135]:


panel_area = 10  
efficiency = 0.20  
sun_hours = 5

df["E_produced"] = df["ALLSKY_SFC_SW_DWN"] * panel_area * efficiency * sun_hours 


# In[136]:


selected_features = ["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_KT"]  
target = ["E_produced"]

scaler = MinMaxScaler()
df[selected_features + target] = scaler.fit_transform(df[selected_features + target])


# In[137]:


def create_sequences(data, target, seq_length=15):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df[selected_features].values, df[target].values, seq_length=15)


# In[138]:


train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[-test_size:], y[-test_size:]


# In[139]:


model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.3),
    BatchNormalization(),
    
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    BatchNormalization(),

    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1)
])


# In[140]:


callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]


# In[ ]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="mse",
    metrics=["mae"],
)


# In[142]:


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


# In[ ]:


# test_loss = model.evaluate(X_test, y_test)
# test_loss
yy, xx = model.evaluate(X_test, y_test, verbose=2)
yy, xx


# In[144]:


import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[145]:


import matplotlib.pyplot as plt
# accuracy
plt.plot(history.history['learning_rate'], label="Learning Rate")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


model.save("outputs/models/model.h5")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "renewable_time_serise_tracker.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "renewable_time_serise_tracker.ipynb" --output-dir="outputs/html"')

