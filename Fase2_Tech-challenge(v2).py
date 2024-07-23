# importanto as bibliotecas
import seaborn as sb
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#1-Importando os dados
url = 'https://raw.githubusercontent.com/LucasDvol/Tech_Challenge_Fase2/main/Dados%20Hist%C3%B3ricos%20-%20Ibovespa(01-1990_06-2024).csv'
df= pd.read_csv(url, delimiter=',', decimal=',', thousands='.', index_col='Data',dayfirst=True, parse_dates=True)

#1.1-Avaliando DF
# print(df.head())
# print(df.describe().round(2))
# print(df.isnull().sum())

#2-Análise exploratória dos dados

# Plotando a série temporal
plt.figure(figsize=(14, 7))
plt.plot(df['Último'], label='Fechamento')
plt.title('Fechamento Diário do IBOVESPA')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
# plt.legend()
# plt.show()

#3-Processamento dos dados

# Selecionando a coluna de fechamento
close_prices = df['Último'].values.reshape(-1, 1)

# Normalizando os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

# Dividindo os dados em treino e teste
train_size = int(len(scaled_close) * 0.8)
test_size = len(scaled_close) - train_size
train, test = scaled_close[:train_size], scaled_close[train_size:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 60
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# Verificando as dimensões antes do reshape
print(f'x_train shape before reshape: {x_train.shape}')
print(f'x_test shape before reshape: {x_test.shape}')

# Reshape para [samples, time steps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
if x_test.size > 0:
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Verificando as dimensões após o reshape
print(f'x_train shape after reshape: {x_train.shape}')
print(f'x_test shape after reshape: {x_test.shape}')

# Criando o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')




# Treinando o modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stopping])

# Fazendo previsões
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Invertendo a normalização
train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculando as métricas de erro
train_score = math.sqrt(mean_squared_error(trainY, train_predict))
test_score = math.sqrt(mean_squared_error(testY, test_predict))
print(f'Train Score: {train_score} RMSE')
print(f'Test Score: {test_score} RMSE')

# Avaliando o modelo
MAE = mean_absolute_error(testY, test_predict)
MSE = mean_squared_error(testY, test_predict)
r2 = r2_score(testY, test_predict)
print('MAE:', MAE) # Mean Absolute Error (MAE)
print('MSE:', MSE) # Erro Quadrático Médio
print('R²:', r2) # (R-quadrado)

# Plotando as previsões
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(test_predict):], testY, label='Valor Real')
plt.plot(df.index[-len(test_predict):], test_predict, label='Previsão')
plt.title('Previsões de Fechamento do IBOVESPA')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.show()


