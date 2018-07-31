# rating of an app
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocess():

    data_source = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/sensor.csv'

    data = pd.read_csv(data_source)
    data = data.drop('rn', axis = 1)
    data = pd.get_dummies(data, columns=['activity'])

    x = data[data.columns[:561]].values
    y = data[data.columns[561:]].values

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

model = Sequential()

model.add(Dense(20, input_dim=561, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
          
history_1 = model.fit(train_x, train_y, epochs=50, validation_split=0.2, verbose=2)

plt.plot(history_1.history['loss'], color='b', label='Model 1')
plt.legend()
plt.show()

accuracy = model.evaluate(test_x, test_y)
print('Accuracy: %0.2f%%' % (accuracy[1] * 100))
