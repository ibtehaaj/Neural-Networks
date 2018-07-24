import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def preprocess():
        
    df = pd.read_csv("F:/Gautam/Tech Stuff/Python Projects/Datasets/sonar.csv",
                     index_col = False, header = None)

    df = pd.get_dummies(df, columns=[60])

    x = df[df.columns[0:60]].values
    y = df[df.columns[60:62]].values

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

model = Sequential()

model.add(Dense(31, activation='relu', input_dim=60))
model.add(Dense(31, activation='relu'))
model.add(Dense(2, activation='softmax'))

adam = Adam(lr=0.003)

model.compile(loss="categorical_crossentropy", optimizer=adam,
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100, batch_size=16, verbose=2,
          validation_split=0.1)

score = model.evaluate(test_x, test_y)
print("Accuracy: %0.2f%%" % (score[1] * 100))
