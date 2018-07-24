import numpy as np
from tensorflow.python.keras.models import load_model

while True:
    
    print("Welcome to Poker Card Analyzer!")
    print("Input your cards in hand:")

    print("\nHearts-h, Spades-s, Diamonds-d, Clubs-c")
    s1 = input("Enter the suit of card 1: >>>")
    print('Ace-1, Jack-11, Queen-12, King-13')
    r1 = input("Enter the rank of card 1: >>>")

    print("\nHearts-h, Spades-s, Diamonds-d, Clubs-c")
    s2 = input("Enter the suit of card 2: >>>")
    print('Ace-1, Jack-11, Queen-12, King-13')
    r2 = input("Enter the rank of card 2: >>>")

    print("\nHearts-h, Spades-s, Diamonds-d, Clubs-c")
    s3 = input("Enter the suit of card 3: >>>")
    print('Ace-1, Jack-11, Queen-12, King-13')
    r3 = input("Enter the rank of card 3: >>>")

    print("\nHearts-h, Spades-s, Diamonds-d, Clubs-c")
    s4 = input("Enter the suit of card 4: >>>")
    print('Ace-1, Jack-11, Queen-12, King-13')
    r4 = input("Enter the rank of card 4: >>>")

    print("\nHearts-h, Spades-s, Diamonds-d, Clubs-c")
    s5 = input("Enter the suit of card 5: >>>")
    print('Ace-1, Jack-11, Queen-12, King-13')
    r5 = input("Enter the rank of card 5: >>>")


    if s1 == 'h':
        s1_list = [1, 0, 0, 0]

    elif s1 == 's':
        s1_list = [0, 1, 0, 0]

    elif s1 == 'd':
        s1_list = [0, 0, 1, 0]

    elif s1 == 'c':
        s1_list = [0, 0, 0, 1]

    else:
        print('Error! | Suit of Card 1')


    if r1 == '1':
        r1_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r1 == '2':
        r1_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r1 == '3':
        r1_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r1 == '4':
        r1_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r1 == '5':
        r1_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r1 == '6':
        r1_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    elif r1 == '7':
        r1_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    elif r1 == '8':
        r1_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif r1 == '9':
        r1_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    elif r1 == '10':
        r1_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    elif r1 == '11':
        r1_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    elif r1 == '12':
        r1_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif r1 == '13':
        r1_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    else:
        print('Error! | Rank of Card 1')


    if s2 == 'h':
        s2_list = [1, 0, 0, 0]

    elif s2 == 's':
        s2_list = [0, 1, 0, 0]

    elif s2 == 'd':
        s2_list = [0, 0, 1, 0]

    elif s2 == 'c':
        s2_list = [0, 0, 0, 1]

    else:
        print('Error! | Suit of Card 2')


    if r2 == '1':
        r2_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r2 == '2':
        r2_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r2 == '3':
        r2_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r2 == '4':
        r2_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r2 == '5':
        r2_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r2 == '6':
        r2_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    elif r2 == '7':
        r2_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    elif r2 == '8':
        r2_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif r2 == '9':
        r2_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    elif r2 == '10':
        r2_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    elif r2 == '11':
        r2_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    elif r2 == '12':
        r2_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif r2 == '13':
        r2_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    else:
        print('Error! | Rank of Card 2')


    if s3 == 'h':
        s3_list = [1, 0, 0, 0]

    elif s3 == 's':
        s3_list = [0, 1, 0, 0]

    elif s3 == 'd':
        s3_list = [0, 0, 1, 0]

    elif s3 == 'c':
        s3_list = [0, 0, 0, 1]

    else:
        print('Error! | Suit of Card 3')


    if r3 == '1':
        r3_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r3 == '2':
        r3_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r3 == '3':
        r3_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r3 == '4':
        r3_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r3 == '5':
        r3_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r3 == '6':
        r3_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    elif r3 == '7':
        r3_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    elif r3 == '8':
        r3_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif r3 == '9':
        r3_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    elif r3 == '10':
        r3_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    elif r3 == '11':
        r3_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    elif r3 == '12':
        r3_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif r3 == '13':
        r3_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    else:
        print('Error! | Rank of Card 3')


    if s4 == 'h':
        s4_list = [1, 0, 0, 0]

    elif s4 == 's':
        s4_list = [0, 1, 0, 0]

    elif s4 == 'd':
        s4_list = [0, 0, 1, 0]

    elif s4 == 'c':
        s4_list = [0, 0, 0, 1]

    else:
        print('Error! | Suit of Card 4')


    if r4 == '1':
        r4_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r4 == '2':
        r4_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r4 == '3':
        r4_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r4 == '4':
        r4_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r4 == '5':
        r4_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r4 == '6':
        r4_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    elif r4 == '7':
        r4_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    elif r4 == '8':
        r4_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif r4 == '9':
        r4_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    elif r4 == '10':
        r4_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    elif r4 == '11':
        r4_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    elif r4 == '12':
        r4_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif r4 == '13':
        r4_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    else:
        print('Error! | Rank of Card 4')


    if s5 == 'h':
        s5_list = [1, 0, 0, 0]

    elif s5 == 's':
        s5_list = [0, 1, 0, 0]

    elif s5 == 'd':
        s5_list = [0, 0, 1, 0]

    elif s5 == 'c':
        s5_list = [0, 0, 0, 1]

    else:
        print('Error! | Suit of Card 5')


    if r5 == '1':
        r5_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r5 == '2':
        r5_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r5 == '3':
        r5_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r5 == '4':
        r5_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r5 == '5':
        r5_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    elif r5 == '6':
        r5_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    elif r5 == '7':
        r5_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    elif r5 == '8':
        r5_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif r5 == '9':
        r5_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    elif r5 == '10':
        r5_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    elif r5 == '11':
        r5_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    elif r5 == '12':
        r5_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif r5 == '13':
        r5_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    else:
        print('Error! | Rank of Card 5')


    final_list = s1_list + r1_list + s2_list + r2_list + s3_list + r3_list + s4_list + r4_list + s5_list + r5_list

    final_list = np.array(final_list)

    model = load_model('Poker_99.model')
    pred = model.predict(final_list.reshape((1, 85)))

    print('-' * 50)

    if np.argmax(pred[0]) == 0:
        print('Nothing in hand; not a recognized poker hand ')

    elif np.argmax(pred[0]) == 1:
        print('One pair; one pair of equal ranks within five cards ')

    elif np.argmax(pred[0]) == 2:
        print('Two pairs; two pairs of equal ranks within five cards ')

    elif np.argmax(pred[0]) == 3:
        print('Three of a kind; three equal ranks within five cards ')

    elif np.argmax(pred[0]) == 4:
        print('Straight; five cards, sequentially ranked with no gaps ')

    elif np.argmax(pred[0]) == 5:
        print('Flush; five cards with the same suit ')

    elif np.argmax(pred[0]) == 6:
        print('Full house; pair + different rank three of a kind ')

    elif np.argmax(pred[0]) == 7:
        print('Four of a kind; four equal ranks within five cards ')

    elif np.argmax(pred[0]) == 8:
        print('Straight flush; straight + flush ')

    elif np.argmax(pred[0]) == 9:
        print('Royal flush; {Ace, King, Queen, Jack, Ten} + flush ')

    else:
        print('Error in script!')


    user_input = input("Do you wish to try again? [y / n]")

    if user_input == 'y':
        print('Reloading...\n')

    else:
        print('Exiting...')
        break

