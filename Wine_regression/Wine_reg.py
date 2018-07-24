import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import BatchNormalization, Activation
from tensorflow.python.keras.optimizers import Adadelta


def preprocess():

    data_source_red = 'F:\Gautam\Tech Stuff\Python Projects\Datasets\winequality-red.csv'
    data_source_white = 'F:\Gautam\Tech Stuff\Python Projects\Datasets\winequality-white.csv'

    data_red = pd.read_csv(data_source_red, index_col=False, sep=';')
    data_white = pd.read_csv(data_source_white, index_col=False, sep=';')
   
    data = pd.concat([data_red, data_white])
    data = data.dropna(inplace=False)

    x = data[data.columns[0:11]].values
    y = data[data.columns[11]].values

##    y = np.expand_dims(y, -1)

    x = np.float32(x)
    y = np.float32(y)

    return (x, y)


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

'''
model = Sequential()

model.add(Dense(100, input_dim=11))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))

opt = Adadelta()

model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit(train_x, train_y, verbose=2, validation_split=0.1, batch_size=128,
          epochs=200)

score = model.evaluate(test_x, test_y)
print('Score: \n', score)
'''

model = load_model('Wine_reg.model')

pred_y = model.predict(test_x)

plt.plot(test_y, color = 'red', label = 'Real data')
plt.plot(pred_y, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
