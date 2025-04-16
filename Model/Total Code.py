import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = r"D:\Python Project\Python Learn\5010 project\New Code Data\CleanedData\cleaned_data.csv"

nvidia = pd.read_csv(file_path)

print("Total entries:", len(nvidia.index))
nvidia = nvidia.dropna(how='any', axis=0)
print("Entries after removing nulls:", len(nvidia.index))

nvidia = nvidia.sort_values('Date')

print(nvidia.head())

print("Columns in the dataset:", nvidia.columns.tolist())

plt.figure(figsize=(20, 9))
plt.plot(nvidia['Date'], nvidia['NVDA_adj_close'], label='Closing Price')
plt.xticks(range(0, nvidia.shape[0], 500), nvidia['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price', fontsize=12)
plt.title('Nvidia Stock Price Changes Over Time', fontsize=20)
plt.legend()
plt.show()

plt.figure(figsize=(20, 9))
plt.plot(nvidia['Date'], nvidia['NVDA_Volume'])
plt.xticks(range(0, nvidia.shape[0], 100), nvidia['Date'].loc[::100], rotation=45)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.title('Volume of Nvidia stocks traded over time', fontsize=20)
plt.show()

nvidia['Daily_returns'] = nvidia['NVDA_adj_close'].pct_change()
nvidia.head()

plt.figure(figsize=(20, 9))
plt.plot(nvidia['Date'], nvidia['Daily_returns'])
plt.xticks(range(0, nvidia.shape[0], 100), nvidia['Date'].loc[::100], rotation=45)
plt.title('Daily Returns of Nvidia Stock', fontsize=18)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Return', fontsize=12)
plt.show()

short_MA = nvidia['NVDA_adj_close'].rolling(window=20).mean()

long_MA = nvidia['NVDA_adj_close'].rolling(window=80).mean()
plt.figure(figsize=(20, 9))

plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('Moving Average in Comparison to Price', fontsize=18)

plt.plot(nvidia['Date'], nvidia['NVDA_adj_close'], label='Price')
plt.plot(nvidia['Date'], short_MA, label='Short MA')
plt.plot(nvidia['Date'], long_MA, label='Long MA')

plt.xticks(range(0, nvidia.shape[0], 100), nvidia['Date'].loc[::100], rotation=45)

plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

path_processed = r'D:\Python Project\Python Learn\5010 project\New Code Data\CleanedData\cleaned_data.csv'
data = pd.read_csv(path_processed, parse_dates=['Date'])
data.dropna(inplace=True)

interest_columns = ['NVDA_adj_close', 'SP500', 'GoldPrice', 'DXI', 'Google_Adj_Close',
                    'AAPL_Adj_Close', 'MSFT_Adj_Close', 'INTC_Adj_Close', 'Inflation',
                    'US_Interest_Rate']

for col in interest_columns:
    data[f'{col}_return'] = data[col].pct_change()

data.dropna(inplace=True)

fig, axes = plt.subplots(nrows=len(interest_columns), ncols=1, figsize=(10, len(interest_columns)*5))
for i, col in enumerate(interest_columns):
    sns.lineplot(data=data, x='Date', y=f'{col}_return', ax=axes[i], label=f'{col} Return')
    axes[i].set_title(f'{col} Return Over Time')
plt.tight_layout()
plt.show()

print('Data Statistics:\n', data.describe())

corr_matrix = data[[f'{col}_return' for col in interest_columns]].corr()
print('Correlation Matrix:\n', corr_matrix)

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Returns')
plt.show()

high_corr_threshold = 0.7
for col in corr_matrix.columns:
    highly_correlated = corr_matrix.index[(corr_matrix[col] > high_corr_threshold) & (corr_matrix.index != col)].tolist()
    if highly_correlated:
        print(f"{col} has high correlation with: {', '.join(highly_correlated)}")


from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data[[f'{col}_return' for col in interest_columns if col != 'NVDA_adj_close']]
y = data['NVDA_adj_close_return']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f'PCA selected {pca.n_components_} components')

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_pca, y_train)
lasso_coef = lasso.coef_
print(f'Lasso selected features: {dict(zip(X.columns, lasso_coef))}')

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
rf_feature_importance = rf.feature_importances_
print(f'Random Forest feature importance: {dict(zip(X.columns, rf_feature_importance))}')

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

file_path = r"D:\Python Project\Python Learn\5010 project\New Code Data\CleanedData\merged_with_impact_score.csv"
nvidia = pd.read_csv(file_path)

nvidia.dropna(subset=['NVDA_adj_close', 'Date'], inplace=True)

nvidia['Date'] = pd.to_datetime(nvidia['Date'])
nvidia.sort_values('Date', inplace=True)
plt.figure()
plot_acf(nvidia['NVDA_adj_close'])
plt.title('NVIDIA Adjusted Close - Autocorrelation plot')
plt.show()

result = adfuller(nvidia['NVDA_adj_close'])
print('The ADF Statistic: %f' % result[0])
print('The p-value: %f' % result[1])

plt.figure()
lag_plot(nvidia['NVDA_adj_close'], lag=3)
plt.title('Nvidia Stock - Autocorrelation plot with lag = 3')
plt.show()

training, testing = train_test_split(nvidia, test_size=0.3, random_state=21)

training = training.reset_index(drop=True)
testing = testing.reset_index(drop=True)

train_data = training['NVDA_adj_close'].values
test_data = testing['NVDA_adj_close'].values

history = [x for x in train_data]
model_preds = []
N_test_obs = len(test_data)

for time_point in range(N_test_obs):
    model = sm.tsa.arima.ARIMA(history, order=(4,1,0))
    model_fit = model.fit()
    out = model_fit.forecast()
    y_pred = out[0]
    model_preds.append(y_pred)
    history.append(test_data[time_point])

MSE_error = mean_squared_error(test_data, model_preds)
print('Testing MSE is {}'.format(MSE_error))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from arch import arch_model

file_path = r"D:\Python Project\Python Learn\5010 project\New Code Data\CleanedData\merged_with_impact_score.csv"
nvidia = pd.read_csv(file_path)

nvidia.dropna(inplace=True)

nvidia['NVDA_return'] = nvidia['NVDA_adj_close'].pct_change()

nvidia['SP500_return'] = nvidia['SP500'].pct_change()

nvidia['NVDA_return_log'] = np.log1p(nvidia['NVDA_return'])
nvidia['SP500_return_log'] = np.log1p(nvidia['SP500_return'])
nvidia['INTC_Adj_Close_return'] = nvidia['INTC_Adj_Close'].pct_change()
nvidia.dropna(inplace=True)
endog = nvidia['NVDA_return_log']
exog_vars = ['SP500_return_log', 'ImpactScore', 'INTC_Adj_Close_return']
exog = nvidia[exog_vars]

train_size = int(len(nvidia) * 0.7)
train_endog = endog[:train_size]
test_endog = endog[train_size:]
train_exog = exog[:train_size]
test_exog = exog[train_size:]

sarimax_model = sm.tsa.SARIMAX(train_endog,
                               exog=train_exog,
                               order=(1, 0, 1),
                               seasonal_order=(0, 1, 1, 12))
sarimax_result = sarimax_model.fit()

print(sarimax_result.summary())

garch = arch_model(train_endog, vol='Garch', p=1, q=1)
garch_result = garch.fit(disp='off')
print(garch_result.summary())

predictions = sarimax_result.get_forecast(steps=len(test_endog), exog=test_exog)
predicted_mean = predictions.predicted_mean
predicted_ci = predictions.conf_int()

plt.figure(figsize=(10, 5))
plt.plot(test_endog.index, test_endog, label='Observed')
plt.plot(test_endog.index, predicted_mean, label='Forecast', color='r')
plt.fill_between(test_endog.index, predicted_ci.iloc[:, 0], predicted_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()

sarimax_result.plot_diagnostics(figsize=(15, 12))
plt.show()

print(sarimax_result.summary())

coefficients = sarimax_result.params
p_values = sarimax_result.pvalues
print("\nCoefficients:\n", coefficients)
print("\nP-values:\n", p_values)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import matplotlib.dates as mdates
from tensorflow.keras.layers import Attention
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Attention, Bidirectional, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd

data = pd.read_csv(r'D:\Python Project\Python Learn\5010 project\New Code Data\CleanedData\merged_with_impact_score.csv')
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data.sort_values('Date', inplace=True)

data['NVDA_adj_close_Short_MA'] = data['NVDA_adj_close'].rolling(window=20).mean()
data['NVDA_adj_close_Long_MA'] = data['NVDA_adj_close'].rolling(window=80).mean()

selected_features = [
    'SP500', 'Google_Adj_Close', 'MSFT_Adj_Close', 'ImpactScore',
    'INTC_Adj_Close',
    'NVDA_adj_close_Long_MA', 'NVDA_adj_close_Short_MA',
    ]
features = data[selected_features]
target = data['NVDA_adj_close']

data = data.dropna(subset=selected_features + ['NVDA_adj_close'])

features = data[selected_features]
target = data['NVDA_adj_close']

scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)
scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(features_scaled, target_scaled.ravel())

rf_predictions = rf_model.predict(features_scaled)
rf_predictions_reshaped = rf_predictions.reshape(-1, 1)


extended_features = np.hstack((features_scaled, rf_predictions_reshaped))
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def plot_predictions(real_vals, lstm_predicted_vals, title="Predictions vs. Actual", num_dates=10):
    plt.figure(figsize=(14, 7))


    dates = data['Date'].iloc[train_size:train_size + len(real_vals)].map(mdates.date2num)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


    baseline_predicted_vals = np.roll(real_vals, 2)
    baseline_predicted_vals[:2] = real_vals[:2]

    plt.plot(dates, real_vals, label='Actual Biotechnology Price', color='blue')
    plt.plot(dates, lstm_predicted_vals, label='LSTM Predicted Biotechnology Price ', color='red')
    plt.plot(dates, baseline_predicted_vals, label='Baseline Predicted Biotechnology Price', color='green')

    random_indices = random.sample(range(len(real_vals)), num_dates)
    for idx in random_indices:
        date = data.iloc[train_size + idx]['Date']
        plt.scatter(mdates.date2num(date), real_vals[idx], color='blue')
        plt.scatter(mdates.date2num(date), lstm_predicted_vals[idx], color='red')
        plt.scatter(mdates.date2num(date), baseline_predicted_vals[idx], color='green')
    print(
        f"Date: {date.strftime('%Y-%m-%d')}, Actual: {real_vals[idx]}, LSTM Predicted: {lstm_predicted_vals[idx]}, Baseline Predicted: {baseline_predicted_vals[idx]}")

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Biotechnology Price')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


    lstm_mae = np.mean(np.abs(real_vals - lstm_predicted_vals))
    baseline_mae = np.mean(np.abs(real_vals - baseline_predicted_vals))

    print(f"Average MAE between LSTM predictions and actual values: {lstm_mae}")
    print(f"Average MAE between baseline predictions and actual values: {baseline_mae}")

    return lstm_mae, baseline_mae


time_steps_list = [2]
performance_metrics = {}

for time_steps in time_steps_list:
    X, y = create_dataset(extended_features, target_scaled.ravel(), time_steps=time_steps)

    train_size = int(len(X) * 0.95)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)))(input_layer)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    query = x
    value = x
    attention = Attention(use_scale=True)([query, value])
    attention = Dropout(0.05)(attention)
    attention = BatchNormalization()(attention)

    x = Bidirectional(LSTM(50, kernel_regularizer=l2(0.01)))(attention)
    x = Dropout(0.3)(x)

    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=128, batch_size=128,
                        validation_data=(X_val, y_val), verbose=1,
                        callbacks=[early_stop]
                        )

    predicted_val = model.predict(X_val)
    predicted_val_rescaled = scaler_target.inverse_transform(predicted_val)
    real_val_rescaled = scaler_target.inverse_transform(y_val.reshape(-1, 1))

    mse_val = mean_squared_error(real_val_rescaled, predicted_val_rescaled)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(real_val_rescaled, predicted_val_rescaled)
    r2_val = r2_score(real_val_rescaled, predicted_val_rescaled)

    performance_metrics[time_steps] = {
        'MSE': mse_val,
        'RMSE': rmse_val,
        'MAE': mae_val,
        'R^2': r2_val
    }



    plot_predictions(real_val_rescaled, predicted_val_rescaled,
                     title=f"LSTM Model Predictions for time_steps = {time_steps}")

for time_steps, metrics in performance_metrics.items():
    print(f"Results for time_steps = {time_steps}:")
    print(f"MSE: {metrics['MSE']}, RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, R^2: {metrics['R^2']}\n")

