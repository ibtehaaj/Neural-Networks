import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

train_dir = 'fashion-mnist_train.csv'
test_dir = 'fashion-mnist_test.csv' 


def preprocess(test=False):
    
    if test:
        df = pd.read_csv(test_dir)
    else:
        df = pd.read_csv(train_dir)
        
    df = pd.get_dummies(df, columns=['label'])

    x = df[df.columns[0:784]].values
    y = df[df.columns[784:]].values

    x = x.astype('float32')/255
    y = y.astype('float32')

    print('Shape of X: ', x.shape)
    print('Shape of y: ', y.shape)

    return x, y


train_x, train_y = preprocess()

model = Sequential()

model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

cb_mcp = ModelCheckpoint('fmnist_91.model', monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', period=1, verbose=1)

cb_es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)

history = model.fit(train_x, train_y, verbose=2,
                    validation_split=0.1, epochs=100,
                    callbacks=[cb_mcp, cb_es])

plt.plot(history.history['loss'], color='b', label='Loss')
plt.plot(history.history['val_loss'], color='r', label='Val_Loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], color='r', label='Accuracy')
plt.plot(history.history['val_acc'], color='b', label='Val_Accuracy')
plt.legend()
plt.show()
