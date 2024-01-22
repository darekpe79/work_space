# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:00:49 2024

@author: dariu
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Ścieżka do pliku z danymi
file_path = 'C:/Users/dariu/cdr_d.csv'

# Ładowanie danych
data = pd.read_csv(file_path)
data['Data'] = pd.to_datetime(data['Data'])
data.set_index('Data', inplace=True)

# Wybranie kolumny 'Zamknięcie' do analizy
series = data['Zamkniecie']

# Dopasowanie modelu ARIMA na całym dostępnym zbiorze danych
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

# Określenie liczby prognozowanych kroków
# Na przykład, aby prognozować do końca lutego 2024, możemy użyć około 30 dni roboczych
# Jako punkt startowy wybieramy 22 stycznia 2024
start_date = '2024-01-22'
end_date = '2024-02-28'
forecast_period = pd.bdate_range(start=start_date, end=end_date).shape[0]

# Prognozowanie
forecast = model_fit.forecast(steps=forecast_period)

# Utworzenie DataFrame z prognozami
forecast_dates = pd.date_range(start=start_date, periods=forecast_period, freq='B')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})

print(forecast_df)


#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Ścieżka do pliku z danymi
file_path = 'C:/Users/dariu/cdr_d.csv'  # Zmień na ścieżkę do pliku z Twoimi danymi

# Ładowanie danych
data = pd.read_csv(file_path)
data['Data'] = pd.to_datetime(data['Data'])
data.set_index('Data', inplace=True)

# Wybranie kolumny 'Zamknięcie' do analizy
series = data['Zamkniecie']

# Dopasowanie modelu ARIMA
model = ARIMA(series, order=(5,1,0))  # Parametry modelu mogą wymagać dostosowania
model_fit = model.fit()

# Wykresy danych i dopasowanego modelu
plt.figure(figsize=(12,6))
plt.plot(series, label='Original')
plt.plot(model_fit.fittedvalues, color='red', label='Fitted Values')
plt.title('Original Series vs ARIMA Fitted Values')
plt.legend()
plt.show()

# Sprawdzenie reszt
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(title="Residuals")
plt.show()

residuals.plot(kind='kde', title='Density of Residuals')
plt.show()

print(residuals.describe())



#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Ścieżka do pliku z danymi
file_path = 'C:/Users/dariu/cdr_d.csv'  # Zmień na ścieżkę do pliku z Twoimi danymi

# Ładowanie danych
data = pd.read_csv(file_path)
data['Data'] = pd.to_datetime(data['Data'])
data.set_index('Data', inplace=True)

# Wybranie kolumny 'Zamknięcie' do analizy
series = data['Zamkniecie']

# Dopasowanie modelu ARIMA
model = ARIMA(series, order=(5,1,0))  # Parametry modelu mogą wymagać dostosowania
model_fit = model.fit()

# Określenie liczby prognozowanych kroków
# Na przykład, aby prognozować do końca lutego, możemy użyć około 20 dni roboczych
forecast_period = 20  # Możesz dostosować tę wartość

# Prognozowanie
forecast = model_fit.forecast(steps=forecast_period)

# Wyświetlenie prognoz
forecast_dates = pd.date_range(start='2024-01-22', periods=forecast_period, freq='B')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})
print(forecast_df)
