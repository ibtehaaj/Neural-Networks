import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


def preprocess():

    data_source = 'F:\Gautam\Tech Stuff\Python Projects\Datasets\car.csv'

    data = pd.read_csv(data_source, index_col=False, header=None)

    data.columns = ['Cost', 'Maintenance', 'Doors', 'Persons',
                    'Boot', 'Safety', 'Class']

    data = pd.get_dummies(data, columns=['Cost', 'Maintenance', 'Doors',
                                    'Persons', 'Boot', 'Safety', 'Class'])

    x = data[data.columns[0:21]].values
    y = data[data.columns[21:]].values

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

model = Sequential()

model.add(Dense(12, input_dim=21, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, verbose=2, batch_size=32, epochs=100,
          validation_split=0.1)

evaluation = model.evaluate(test_x, test_y)
print('Accuracy: %0.2f%%' % (evaluation[1] * 100))
