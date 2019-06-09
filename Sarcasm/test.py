import json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


def convert_text_to_index_array(text):
    words = text_to_word_sequence(text)
    wordIndices = []

    for word in words:
        if word in word_index:
            wordIndices.append(word_index[word])
        else:
            print("'%s' not in training corpus; ignoring." % (word))

    return [wordIndices]


with open('dictionary_sarcasm.json', 'r') as f:
    word_index = json.load(f)

print('Dictionary Loaded!')

model = load_model('Sarcasm.model')
print('Model Loaded!')

while True:
    ui = input("\nEnter text here, 'q' to exit: ")

    if ui == 'q':
        print('Exiting...')
        break

    else:
        try:
            ui = convert_text_to_index_array(ui)
            ui = pad_sequences(ui, maxlen=15, padding='post')

            prediction = model.predict(ui)[0][0]
            print('Prediction:', prediction)

            if prediction > 0.5:
                print('\n\tSarcasm!')
            else:
                print('\n\tReal!')
        except Exception:
            pass
