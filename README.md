# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 
### Developed by: ANUSHARON.S
### Registration no.: 212222240010


### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')

target_values = np.random.randint(100, 200, size=len(date_range))

time_series_df = pd.DataFrame({'date': date_range, 'target_column': target_values})

time_series_df.to_csv('time_series_data.csv', index=False)
df = pd.read_csv('/content/time_series_data.csv', parse_dates=['date'], index_col='date')

print(df.head())
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['target_column'])  # Replace 'target_column' with the actual column name

train_size = int(len(df) * 0.8)
train, test = df['target_column'][:train_size], df['target_column'][train_size:]

model = AutoReg(train, lags=13)
ar_model_fit = model.fit()

print(ar_model_fit.summary())

plt.figure(figsize=(12,6))

plt.subplot(121)
plot_acf(train, lags=30, ax=plt.gca())

plt.subplot(122)
plot_pacf(train, lags=30, ax=plt.gca())

plt.tight_layout()
plt.show()
predictions = ar_model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

print(predictions)

comparison_df = pd.DataFrame({'Actual': test, 'Predicted': predictions})
print(comparison_df.head())
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')
plt.figure(figsize=(10,6))
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, predictions, color='red', label='Predicted Data')
plt.title('AR Model - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
```
### OUTPUT:

GIVEN DATA

PACF - ACF
![Screenshot 2024-10-16 134655](https://github.com/user-attachments/assets/64e290a3-65c7-4911-a9bc-6714a5ab4e4f)

Mean Squared Error

![Screenshot 2024-10-16 134905](https://github.com/user-attachments/assets/d2ab8076-80dc-4b17-91b9-662ab584bff4)


PREDICTION
![Screenshot 2024-10-16 134810](https://github.com/user-attachments/assets/44ba2bdb-4f51-4a17-9871-aa987534cecd)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
