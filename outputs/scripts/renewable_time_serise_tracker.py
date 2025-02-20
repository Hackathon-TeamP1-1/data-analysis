#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


df = pd.read_csv("outputs/preprocessed_data/nasa_data_cleaned.csv")
df.shape


# In[24]:


# Convert to datetime
df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))

# Sort by DATE
df = df.sort_values(by=["LAT", "LON", "DATE"])


# In[25]:


df['DATE_DIFF'] = df['DATE'].diff().dt.days

disconnects = df[df['DATE_DIFF'] > 1][['DATE', 'DATE_DIFF']]

# Get value counts of the different gap lengths
disconnect_counts = disconnects['DATE_DIFF'].value_counts().reset_index()
disconnect_counts.columns = ['Gap Length (Days)', 'Frequency']
disconnect_counts = disconnect_counts.sort_values(by='Gap Length (Days)')
disconnect_counts


# In[26]:


# Create a complete date range for each unique (LAT, LON)
full_dates = pd.date_range(start=df["DATE"].min(), end=df["DATE"].max())

# Create a DataFrame with all combinations of (LAT, LON) and full dates
unique_locations = df[["LAT", "LON"]].drop_duplicates()
expanded_dates = unique_locations.assign(key=1).merge(pd.DataFrame({"DATE": full_dates, "key": 1}), on="key").drop("key", axis=1)

# Merge the original dataset with the full date range to introduce missing dates per location
df_filled = expanded_dates.merge(df, on=["LAT", "LON", "DATE"], how="left")

# Interpolate missing values within each (LAT, LON) group
df_filled = df_filled.groupby(["LAT", "LON"]).apply(lambda group: group.interpolate(method="linear")).reset_index(drop=True)

# Forward-fill and backward-fill remaining missing values (if any)
df_filled.fillna(method="ffill", inplace=True)
df_filled.fillna(method="bfill", inplace=True)

df = df_filled


# In[27]:


panel_area = 1.8
efficiency = 0.20
sun_hours = 5

df["E_produced"] = df["ALLSKY_SFC_SW_DWN"] * panel_area * efficiency * sun_hours 


# In[28]:


selected_features = ["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", 'T2M_MAX','T2M_MIN', "ALLSKY_KT", 'ALLSKY_SFC_PAR_TOT', ]  
target = ["E_produced"]

scaler = MinMaxScaler()
df[selected_features + target] = scaler.fit_transform(df[selected_features + target])


# In[29]:


def create_sequences(data, target, seq_length=15):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df[selected_features].values, df[target].values, seq_length=15)


# In[30]:


train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[-test_size:], y[-test_size:]


# In[31]:


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


# In[32]:


callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),  # More patience
    ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=6, min_lr=1e-7, verbose=1)  # Higher min_lr
]


# In[33]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Keep learning rate reasonable
    loss="mse",
    metrics=["mae"],
)


# In[34]:


history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


# In[35]:


test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")


# In[36]:


target_scaler = MinMaxScaler()
target_scaler.fit(df[['E_produced']])  # Fit only on the target column


# In[ ]:





# In[37]:


y_pred = model.predict(X_test)
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  
y_test_rescaled = target_scaler.inverse_transform(y_test)


# In[38]:


import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[39]:


plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual", marker='o', linestyle="dashed")
plt.plot(y_pred_rescaled, label="Predicted", marker='x')
plt.xlabel("Time Steps")
plt.ylabel("Energy Produced")
plt.legend()
plt.title("Actual vs. Predicted Energy Production")
plt.show()


# In[40]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")


# In[41]:


df.shape


# In[42]:


model.save("outputs/models/model.h5")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "renewable_time_serise_tracker.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "renewable_time_serise_tracker.ipynb" --output-dir="outputs/html"')

