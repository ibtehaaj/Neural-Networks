import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizers import Adam, SGD, Adadelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess():
        
    data_source = "F:\Gautam\Tech Stuff\Python Projects\Datasets\GOOG.csv"

    data = pd.read_csv(data_source, index_col=False)

    data = data.drop(['Date', 'Volume', 'Adj Close'], axis=1)

    data['HL_PCT'] = (data['High'] - data['Close']) / data['Close'] * 100
    data['PCT_Change'] = (data['Close'] - data['Open']) / data['Open'] * 100

    data['Label'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    x = data[data.columns[0:6]].values
    y = data[data.columns[6]].values

    x = np.float32(x)
    y = np.float32(y)
    
    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.02)

##model = Sequential()
##
##model.add(Dense(10, input_dim=6, activation='relu'))
##model.add(Dense(10, activation='relu'))
##model.add(Dense(1))
##
##model.compile(loss='mean_squared_error', optimizer='adam')
##
##model.fit(train_x, train_y, verbose=2, batch_size=16, epochs=10,
##          validation_split=0.1)

model = load_model('Stock_reg_1.model')

evaluation = model.evaluate(test_x, test_y)
print('Evaluation: ', evaluation)

pred_y = model.predict(test_x)

plt.plot(test_y, color='red', label='Real Data')
plt.plot(pred_y, color='blue', label='Prediction Data')
plt.title("Prediction of Stock Market")
plt.legend()
plt.show()

