# importanto as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score

# Passo 1: Importando os dados
url = 'https://raw.githubusercontent.com/LucasDvol/Tech_Challenge_Fase2/main/Dados%20Hist%C3%B3ricos%20-%20Ibovespa(01-1990_06-2024).csv'
ibovespa= pd.read_csv(url, delimiter=',', decimal=',', thousands='.', index_col='Data',dayfirst=True, parse_dates=True)

# Passo 2: Limpeza e Análise Exploratória de Dados
# Verificar dados faltantes
print(ibovespa.isnull().sum())
# Remover ou interpolar dados faltantes
ibovespa = ibovespa.interpolate()

# 2.1 Ordenar pela data -- 26/07/2024
ibovespa = ibovespa.sort_values(by='Data')

# 2.2 Análise Exploratória de Dados (EDA) Visualizar dados históricos:
# Converter coluna de data para datetime
ibovespa.index = pd.to_datetime(ibovespa.index)
dates = ibovespa.index

# Plotar evolução do preço de fechamento
plt.figure(figsize=(14, 7))
plt.plot(ibovespa.index, ibovespa['Último'], label='Preço de Fechamento')
plt.title('Evolução do Preço de Fechamento do IBOVESPA')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
# plt.show()

# Passo 3: Desenvolvimento do Modelo Preditivo
# 3.1 Seleção de Técnica de Previsão Justificar a técnica: Escolher um modelo que possa capturar a dependência temporal dos dados, como ARIMA, LSTM (Long Short-Term Memory) ou Prophet. Neste exemplo, usaremos o LSTM, que é eficaz para séries temporais com padrões complexos.
# 3.2 Preparação dos Dados para o Modelo Criar variáveis de entrada e saída para LSTM:
#Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
ibovespa_scaled = scaler.fit_transform(ibovespa['Último'].values.reshape(-1, 1))

# Criar dados de treino
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
train_size = int(len(ibovespa_scaled) * 0.8)
test_size = len(ibovespa_scaled) - train_size
train, test = ibovespa_scaled[0:train_size], ibovespa_scaled[train_size:len(ibovespa_scaled)]

X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape para [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#3.3 Treinamento do Modelo Construir e treinar o modelo LSTM
# Construir o modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compilar o modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Treinar o modelo
model.fit(X_train, Y_train, epochs=5, batch_size=1, verbose=2) #Aumentar epochs para aumentar acurácia (reduz desempenho)

# Passo 4: Avaliação do Modelo
# 4.1 Fazer Previsões Prever e inverter a normalização:
# Fazer previsões
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverter a normalização
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calcular RMSE
train_score = np.sqrt(np.mean((train_predict[:,0] - Y_train[0])**2))
test_score = np.sqrt(np.mean((test_predict[:,0] - Y_test[0])**2))
print('Train RMSE: %.2f' % (train_score))
print('Test RMSE: %.2f' % (test_score))


# Calcular R2 Score
r2_train = r2_score(Y_train[0], train_predict[:, 0])
r2_test = r2_score(Y_test[0], test_predict[:, 0])
print('Train R2 Score: %.2f' % r2_train)
print('Test R2 Score: %.2f' % r2_test)


# Plotando as previsões
train_predict_plot = np.empty_like(ibovespa_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

test_predict_plot = np.empty_like(ibovespa_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(ibovespa_scaled) - 1, :] = test_predict

# Ajustar o índice para os arrays de previsão
train_dates = dates[look_back:len(train_predict) + look_back]
test_dates = dates[len(train_predict) + (look_back * 2) + 1:len(ibovespa_scaled) - 1]

plt.figure(figsize=(14, 7))
plt.plot(dates, scaler.inverse_transform(ibovespa_scaled), label='Preço de Fechamento Real')
plt.plot(train_dates, train_predict, label='Previsão de Treino')
plt.plot(test_dates, test_predict, label='Previsão de Teste')
plt.title('Previsão de Preço de Fechamento do IBOVESPA')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.show()

