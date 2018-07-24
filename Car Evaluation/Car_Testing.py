import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model


model = load_model('Car_99.model')

while True:

    print("\nWelcome to car evaluation!")
    print('Please enter the features of your car according to the format.')

    print('\n\tFeatures format: \n')
    print('\tCost:--------vhigh, high, med, low')
    print('\tMaintenance:-vhigh, high, med, low')
    print('\tDoors:-------2, 3, 4, 5more')
    print('\tPersons:-----2, 4, more')
    print('\tBoot:--------small, med, big')
    print('\tSafety:------low, med, high\n')

    print('E.g. >>>vhigh,low,4,4,med,high')


    cost, maint, door, person, boot, safety = input("Enter the features of the car >>>").split(',')


    if cost == 'vhigh':
        l_cost = [0, 0, 0, 1]

    elif cost == 'high':
        l_cost = [1, 0, 0, 0]

    elif cost == 'low':
        l_cost = [0, 1, 0, 0]

    elif cost == 'med':
        l_cost = [0, 0, 1, 0]

    else:
        print("Error! | Cost")


    if maint == 'vhigh':
        l_maint = [0, 0, 0, 1]

    elif maint == 'high':
        l_maint = [1, 0, 0, 0]

    elif maint == 'low':
        l_maint = [0, 1, 0, 0]

    elif maint == 'med':
        l_maint = [0, 0, 1, 0]

    else:
        print("Error! | Maintenance")


    if door == '2':
        l_door = [1, 0, 0, 0]

    elif door == '3':
        l_door = [0, 1, 0, 0]

    elif door == '4':
        l_door = [0, 0, 1, 0]

    elif door == '5more':
        l_door = [0, 0, 0, 1]

    else:
        print("Error! | Door")


    if person == '2':
        l_person = [1, 0, 0]

    elif person == '4':
        l_person = [0, 1, 0]

    elif person == 'more':
        l_person = [0, 0, 1]

    else:
        print("Error! | Person")


    if boot == 'small':
        l_boot = [0, 0, 1]

    elif boot == 'med':
        l_boot = [0, 1, 0]

    elif boot == 'big':
        l_boot = [1, 0, 0]

    else:
        print("Error! | Boot")


    if safety == 'low':
        l_safety = [0, 1, 0]

    elif safety == 'med':
        l_safety = [0, 0, 1]

    elif safety == 'high':
        l_safety = [1, 0, 0]

    else:
        print("Error! | Safety")


    final_list = l_cost + l_maint + l_door + l_person + l_boot + l_safety
    test = np.asarray(final_list)

    pred = model.predict(test.reshape((1, 21)))

    print('-' * 50)

    if np.argmax(pred[0]) == 0:
        print("The car is not the worst. It's acceptable.")

    elif np.argmax(pred[0]) == 1:
        print("It's a good car.")

    elif np.argmax(pred[0]) == 2:
        print("Not Worthy! DON'T buy it!")

    elif np.argmax(pred[0]) == 3:
        print("That's a fantastic choice!")

    else:
        print("Error! Troubleshoot script!")

    user_input = input("\nDo you want to try again? [y / n] >>> ")

    if user_input == 'n':
        print("Exiting...")
        break

    else:
        print('Reloading...')
        
