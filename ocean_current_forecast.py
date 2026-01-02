import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Ocean Current Speed Data
np.random.seed(42)
n_samples = 150
dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='W')  # Weekly data
seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 26)  # ~6-month seasonal cycle
trend = 0.01 * np.arange(n_samples)  # slow increasing trend
noise = np.random.normal(0, 0.2, n_samples)
current_speed = 1.5 + seasonal + trend + noise  # in m/s

# DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Current_Speed': current_speed
})
df.set_index('Date', inplace=True)

# 2. Create Lag Features for Time Series Supervised Learning
def create_lag_features(data, lags=5):
    df_lag = pd.DataFrame()
    for i in range(1, lags+1):
        df_lag[f'lag_{i}'] = data.shift(i)
    df_lag['target'] = data.values
    df_lag.dropna(inplace=True)
    return df_lag

lags = 5
data_supervised = create_lag_features(df['Current_Speed'], lags)

# 3. Split Data into Train and Test
X = data_supervised.drop('target', axis=1)
y = data_supervised['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# 4. Train ML Models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 5. Evaluate Models
def evaluate(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} RMSE: {rmse:.4f}")

evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)

# 6. Plot Forecast Results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test, label='Actual', marker='o')
plt.plot(df.index[-len(y_test):], y_pred_lr, label='Linear Regression Forecast', linestyle='--')
plt.plot(df.index[-len(y_test):], y_pred_rf, label='Random Forest Forecast', linestyle='--')
plt.title('Ocean Current Speed Forecast (Synthetic Data)')
plt.xlabel('Date')
plt.ylabel('Current Speed (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
