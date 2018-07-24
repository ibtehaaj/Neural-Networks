import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam, SGD, Adadelta


def preprocess():
        
    data_source_train = "F:\Gautam\Tech Stuff\Python Projects\Datasets\poker_training.csv"
    data_source_test = "F:\Gautam\Tech Stuff\Python Projects\Datasets\poker_testing.csv"

    data_train = pd.read_csv(data_source_train, index_col = False, header=None)
    data_test = pd.read_csv(data_source_test, index_col = False, header=None)

    data_train.columns=['S1', 'R1', 'S2', 'R2', 'S3', 'R3',
                        'S4', 'R4', 'S5', 'R5', 'CLASS']

    data_train = pd.get_dummies(data_train, columns=['S1', 'R1', 'S2', 'R2', 'S3', 'R3',
                                                     'S4', 'R4', 'S5', 'R5', 'CLASS'])

    data_test.columns=['S1', 'R1', 'S2', 'R2', 'S3', 'R3',
                       'S4', 'R4', 'S5', 'R5', 'CLASS']

    data_test = pd.get_dummies(data_test, columns=['S1', 'R1', 'S2', 'R2', 'S3', 'R3',
                                                   'S4', 'R4', 'S5', 'R5', 'CLASS'])

    data = pd.concat([data_train, data_test])
    data = data.dropna(inplace=False)

    x = data[data.columns[0:85]].values
    y = data[data.columns[85:]].values

    return x, y


train_x, train_y = preprocess()

model = Sequential()

model.add(Dense(50, input_dim=85, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, verbose=2, batch_size=128, validation_split=0.1, epochs=100)
