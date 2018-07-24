import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


def preprocess():
    
    data_source = "F:/Gautam/Tech Stuff/Python Projects/Datasets/titanic_dataset.csv"

    data = pd.read_csv(data_source, index_col = False)

    data = data.drop(['name', 'ticket'], axis=1)
    
    data = pd.get_dummies(data, columns=["sex"])

    x = data[data.columns[1:8]].values
    y = data[data.columns[0]].values

    x = np.float32(x)
    y = np.float32(y)

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.02)

##model = Sequential()
##
##model.add(Dense(20, activation='relu', input_dim=7))
##model.add(Dense(20, activation='relu'))
##model.add(Dense(1))
##
##model.compile(loss='mean_squared_error', optimizer='adam')
##
##model.fit(train_x, train_y, verbose=2, batch_size=1, epochs=50,
##          validation_split=0.1)

model = load_model('titanic_009.model')

evaluation = model.evaluate(test_x, test_y)
print('Evaluation: ', evaluation)
