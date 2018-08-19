import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

dir_ = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/breast_cancer.csv'


def normalize(df):

    for i in df:
        df[i] /= df[i].max()


def preprocess():
    
    df = pd.read_csv(dir_, header=None)
    df = df.drop([0], axis=1)
    df = pd.get_dummies(df, columns=[1])

    normalize(df)

    x = df[df.columns[:-2]].values
    x = x.astype('float32')
    
    y = df[df.columns[-2:]].values
    y = y.astype('float32')

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

##model = Sequential()
##
##model.add(Dense(20, input_dim=30, activation='relu'))
##model.add(Dense(2, activation='softmax'))
##
##model.compile(optimizer='adam', loss='categorical_crossentropy',
##               metrics=['accuracy'])
##
##cb_mc = ModelCheckpoint('bc_mc.model', monitor='val_loss', save_best_only=True,
##                        save_weights_only=False, period=1, verbose=1, mode='auto')
##
##cb_es = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=1)
##
##history = model.fit(train_x, train_y, verbose=2, validation_split=0.1,
##                    epochs=1000, callbacks=[cb_mc, cb_es])
##
##plt.plot(history.history['acc'], color='b', label='Accuracy')
##plt.plot(history.history['val_acc'], color='r', label='Val_Acc')
##plt.legend()
##plt.show()
##
##plt.plot(history.history['loss'], color='b', label='Loss')
##plt.plot(history.history['val_loss'], color='r', label='Val_Loss')
##plt.legend()
##plt.show()

model = load_model('bc_98.model')

acc = model.evaluate(test_x, test_y, verbose=2)
print('Accuracy: %0.2f%%' % (acc[1]*100))
print('Loss: ', acc[0])
