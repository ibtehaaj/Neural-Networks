import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

por_dir = "F:\\...\\Datasets\\student\\student-por.csv"
math_dir = "F:\\...\\Datasets\\student\\student-mat.csv"

df1 = pd.read_csv(por_dir, sep=';')
df2 = pd.read_csv(math_dir, sep=';')
df = pd.concat([df1, df2], ignore_index=True)

features = ['failures', 'absences', 'G1', 'G2']
x = df[features]
y = df['G3']
# failures = n if 1<=n<3 else 4
# absences = 0-93
# G1 = 0-20
# G2 = 0-20
# G3 = 0-20

x = np.array(x)
y = np.array(y).reshape(y.shape[0], 1)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1,
                                                    random_state=42)

model = Sequential()

model.add(Dense(12, activation='relu', input_dim=4))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

history = model.fit(train_x, train_y, epochs=150, verbose=2,
                    validation_split=0.1)

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

score = model.evaluate(test_x, test_y)
print(score)
# model.save('student.model')

for i in range(len(test_x)):
    pred = model.predict(test_x[i].reshape((1, 4)))
    print(f'Input {test_x[i]}, Prediction {pred} Actual {test_y[i]}')
