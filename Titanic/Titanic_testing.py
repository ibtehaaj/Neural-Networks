import numpy as np
import pandas as pd
from keras.models import load_model


model = load_model('titanic_009.model')

print("\nThis program tells you the survival rate of a subject aboard the Titanic.")

while True:
    
    print("Enter the info: \n")

    subject = []

    p_class = int(input("Enter the passenger class [1, 2, 3]: "))
    subject.append(p_class)

    age = int(input("Enter the age: "))
    subject.append(age)

    sibsp = int(input("Enter the number of siblings/spouse aboard: "))
    subject.append(sibsp)

    parch = int(input("Enter the number of parents/children aboard: "))
    subject.append(parch)

    ticket_fare = float(input("Enter the price of the ticket: "))
    subject.append(ticket_fare)

    sex = input("Enter whether male[0] or female[1]: ")

    if sex == '0':
        gender = [0, 1]
        subject += gender

    elif sex == '1':
        gender = [1, 0]
        subject += gender

    else:
        print("Invalid Entry!")

##    print(subject)

    subject = np.asarray(subject)
    subject = subject.reshape((1, 7))

    pred = model.predict(subject)

    prediction = (pred[0] * 100)

    if prediction < 0:
        prediction = 0.00

    elif prediction > 100:
        prediction = 100
    
    print("\nThe subject has a surviving rate of ", prediction)

    ui = input("\nDo you want to try again? [y]/[n]")

    if ui == 'n':
        print("Exiting...")
        break

    else:
        print("Reloading...\n")
        pass
