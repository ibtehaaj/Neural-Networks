import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time

dir_ = "F:/Gautam/Tech Stuff/Python Projects/Datasets/chess.csv"

def preprocess():
    df = pd.read_csv(dir_, header=None)

    df.columns = ['WKF', 'WKR', 'WRF', 'WRR', 'BKF', 'BKR', 'Label']

    df = pd.get_dummies(df, columns=['WKF', 'WKR', 'WRF', 'WRR',
                                     'BKF', 'BKR', 'Label'])

    x = df[df.columns[:-17]].values
    y = df[df.columns[-17:]].values

    x = x.astype('float32')
    y = y.astype('float32')

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)

model = Sequential()

model.add(Dense(500, input_dim=41, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(17, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', verbose=1, patience=15)

mc = ModelCheckpoint('chess-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.model',
                     verbose=1, period=1, save_best_only=True,
                     save_weights_only=False, monitor='val_loss')

start = time.time()

print(model.summary())

history = model.fit(train_x, train_y, epochs=500, verbose=2,
                    validation_data=(test_x, test_y), callbacks=[es, mc])

end = time.time()
t = end-start

print('\nModel took {:0.2f}s | {:0.2f}m | {:0.2f}h to train.\n'.format(t, (t/60), (t/3600)))

plt.plot(history.history['acc'], label='Acc', color='b')
plt.plot(history.history['val_acc'], label='Val_Acc', color='r')
plt.title('Chess')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Loss', color='b')
plt.plot(history.history['val_loss'], label='Val_Loss', color='r')
plt.title('Chess')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
