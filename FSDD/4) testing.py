import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import random

print('Loading Data...')
data = np.load('digit_data.npy')
print("Data Loaded!")

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
y = np.array([i[1] for i in data])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1,
                                                    random_state=42)

print('Loading Model...')
model = load_model("digit-32-0.02.model")
print('Model Loaded Successfully!')

while True:
    print(test_x.shape)
    num = random.randint(0, test_x.shape[0])
    pred = model.predict(test_x[num].reshape(1, test_x.shape[1],
                                             test_x.shape[2], 1))

    label = str(np.argmax(test_y[num]))
    prediction = str(np.argmax(pred[0]))

    print(f"Predicted digit: {prediction}. Actual digit: {label}")
    ui = input('Press ENTER to continue, "q" to exit... ')

    if ui == 'q':
        print('Exiting...')
        break
