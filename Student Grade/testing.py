import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

print('Loading Model...')
model = load_model('student.model')
print('Model Loaded Successfully!\n')

while True:
    ui = input('Press ENTER to enter the details according to the format, "q" to exit:')

    if ui != 'q':
        fail = float(input('Number of failures (0-3): '))
        absence = float(input('Number of absences (0-93): '))
        g1 = float(input('Marks in Test 1 (0-20): '))
        g2 = float(input('Marks in Test 2 (0-20): '))
        x = np.array([[fail, absence, g1, g2]])

        scaler = MinMaxScaler()
        pred = model.predict(x)
        print(pred)
        scaler.fit(pred)
        pred = scaler.inverse_transform(pred)
        print(pred)

    else:
        break
