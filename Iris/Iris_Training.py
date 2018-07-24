import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


def preprocess():
    
    df = pd.read_csv("F:\Gautam\Tech Stuff\Python Projects\Datasets\iris.csv",
                     index_col = False, header = None)

    df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length',
                  'Petal Width' , 'Class']

    df = pd.get_dummies(df, columns=['Class'], prefix=["Class"])

    x = df[df.columns[0:4]].values
    y = df[df.columns[4:7]].values

    return (x, y)


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)

model = Sequential()

model.add(Dense(10, activation='relu', input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=500, batch_size=1, verbose=2, validation_split=0.1)

score = model.evaluate(test_x, test_y)
print('Accuracy: %0.2f%%' % (score[1] * 100))
