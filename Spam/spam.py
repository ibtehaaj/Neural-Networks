import json
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import plot_model
import matplotlib.pyplot as plt

directory = 'F:/.../Datasets/spam.csv'
VOCAB_SIZE = 5000
SEQ_LEN = 30
DIMENSION = 50

df = pd.read_csv(directory, encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

X = df.v2
y = df.v1
le = LabelEncoder()
y = le.fit_transform(y)  # converting ham, spam = 0, 1

tk = Tokenizer(num_words=VOCAB_SIZE)
tk.fit_on_texts(X)
dic = tk.word_index

with open('dictionary_spam.json', 'w') as f:
    json.dump(dic, f)

sequences = tk.texts_to_sequences(X)  # converting words to integers
X = pad_sequences(sequences, maxlen=SEQ_LEN, padding='post')  # padding to SEQ_LEN

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, DIMENSION, input_length=SEQ_LEN))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

model.fit(train_x, train_y, epochs=3, validation_split=0.1, verbose=1)

score = model.evaluate(test_x, test_y)
print('Accuracy:', score[1])
model.save('Spam.model')

pred = model.predict(test_x)

prediction = []
for i in pred:
    if i[0] <= 0.5:
        prediction.append(0)
    elif i[0] > 0.5:
        prediction.append(1)

confm = confusion_matrix(test_y, prediction)
sns.heatmap(confm, cmap='coolwarm', annot=True)
plt.savefig('Spam_cm.png')
plt.show()
