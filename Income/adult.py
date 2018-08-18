import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping


def preprocess():

    dir_ = 'adult.csv'

    df = pd.read_csv(dir_, header=None)
    df = df.replace(' ?', np.nan)
    df = df.dropna()

    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital-status', 'occupation', 'relationship', 'race',
                  'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                  'native-country', 'label']

    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status',
                                'occupation', 'relationship', 'race', 'sex',
                                'native-country', 'label'])

    df['age'] /= 90
    df['fnlwgt'] /= 1.484705e+06
    df['education_num'] /= 16
    df['capital-gain'] /= 99999
    df['capital-loss'] /= 4356
    df['hours-per-week'] /= 99

    x = df[df.columns[:104]].values
    y = df[df.columns[104:]].values

    x = x.astype('float32')
    y = y.astype('float32')

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

##model = Sequential()
##
##model.add(Dense(50, input_dim=104, activation='relu'))
##model.add(Dense(2, activation='softmax'))
##
##model.compile(loss='categorical_crossentropy', optimizer='adam',
##              metrics=['accuracy'])
##
##cb_mc = ModelCheckpoint('adult_mc.model', monitor='val_loss', verbose=1,
##                        save_best_only=True, save_weights_only=False,
##                        mode='auto', period=1)
##
##cb_es = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
##                      mode='auto')
##
##history = model.fit(train_x, train_y, verbose=2, epochs=100,
##                    validation_split=0.1,
##                    callbacks=[cb_mc, cb_es])
##
##plt.plot(history.history['loss'], color='b', label='Loss')
##plt.plot(history.history['val_loss'], color='r', label='Val_Loss')
##plt.legend()
##plt.show()
##
##plt.plot(history.history['acc'], color='r', label='Accuracy')
##plt.plot(history.history['val_acc'], color='b', label='Val_Accuracy')
##plt.legend()
##plt.show()

model = load_model('adult_84.model')

acc = model.evaluate(test_x, test_y, verbose=2)
print("Accuracy: %0.2f%%" % (acc[1]*100))
print('Loss: ', (acc[0]))
