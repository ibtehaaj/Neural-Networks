import numpy as np
from tensorflow.python.keras.models import load_model


model = load_model('Iris_100.model')

while True:
    
    print("Input data according to the format ',' without space:\n")
    print("Sepal Length,Sepal Width,Petal Length,Petal Width")
    input('PRESS ENTER TO INPUT DATA...')

    sl, sw, pl, pw = input("Enter the data:").split(',')

    test = [sl, sw, pl, pw]
    test = np.asarray(test)

    pred = model.predict(test.reshape((1, 4)))

    print('-' * 50)

    if np.argmax(pred[0]) == 0:
        print("The flower is Iris Setosa.")

    elif np.argmax(pred[0]) == 1:
        print("The flower is Iris Versicolor.")

    elif np.argmax(pred[0]) == 2:
        print("The flower is Iris Virginica.")

    else:
        print("Error! Troubleshoot script!")

    print('\n')
    print('Chance of the flower being Iris Setosa is ', pred[0][0])
    print('Chance of the flower being Iris Versicolor is ', pred[0][1])
    print('Chance of the flower being Iris Virginica is ', pred[0][2])

    print('\n')

    user_input = input("Do you want to try again? [y / n] >>> ")

    if user_input == 'y':
        print("Reloading...")

    else:
        print('Exiting...')
        break

