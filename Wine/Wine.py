import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout


def preprocess():

    data_source = 'F:\\Gautam\\Tech Stuff\Python Projects\\Datasets\\wine.csv'

    data = pd.read_csv(data_source, header = None, index_col = False)
    
    data.columns = ['Class', 'Alcohol', 'Malic Acid', 'Ash',
                    'Alcalinity of Ash', 'Magnesium', 'Total Phenols',
                    'Flavanoids', 'Nonflavanoid Phenols', 'Proanhocyanins',
                    'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines',
                    'Proline']

    data = pd.get_dummies(data, columns=['Class'], prefix=['Class'])
  
    x = data[data.columns[0:13]].values
    y = data[data.columns[13:16]].values

    x = np.float32(x)
    y = np.float32(y) 

    return (x, y)


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)


##model = Sequential()
##
##model.add(Dense(8, activation='relu', input_dim=13))
##model.add(Dropout(0.05))
##model.add(Dense(3, activation='softmax'))
##
##model.compile(loss='categorical_crossentropy', optimizer='adam',
##              metrics=['accuracy'])
##
##model.fit(train_x, train_y, epochs=20, validation_split=0.1,
##          batch_size=1, verbose=2)


model = load_model("Wine_100.model")

score = model.evaluate(test_x, test_y)
print('Accuracy: %0.2f%%' % (score[1] * 100))

result = model.predict(test_x)

for i in range(len(test_x)):

    if np.argmax(result[i]) == 0:
        print('Predicted to be of Class 1! Actual class:', test_y[i])

    elif np.argmax(result[i]) == 1:
        print('Predicted to be of Class 2! Actual class:', test_y[i])

    elif np.argmax(result[i]) == 2:
        print('Predicted to be of Class 3! Actual class:', test_y[i])

    else:
        pass
