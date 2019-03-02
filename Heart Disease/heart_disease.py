import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense


path = "F:\\...\\Datasets\\heart.csv"
df = pd.read_csv(path)

features = ['cp', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal']

X = df[features].values
y = df['target'].values

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# ---------------------------LogisticRegression------------------------
clf = LogisticRegression()
clf.fit(train_x, train_y)
prediction = clf.predict(test_x)

acc_lr = clf.score(test_x, test_y)
cm_lr = confusion_matrix(test_y, prediction)

sns.heatmap(cm_lr, cmap='coolwarm', annot=True)
plt.show()

# --------------------------Neural Network-----------------------------
model = Sequential()

model.add(Dense(30, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100, verbose=0)

prediction = model.predict(test_x)
final_prediction = []

for i in prediction:
    if i[0] < 0.5:
        final_prediction.append(0)
    else:
        final_prediction.append(1)

acc_nn = accuracy_score(test_y, final_prediction)
cm_nn = confusion_matrix(test_y, final_prediction)

sns.heatmap(cm_nn, cmap='coolwarm', annot=True)
plt.show()

print(f'\nAccuracy of Linear Regression: {acc_lr}')
print(f'\nConfusion Matrix of Linear Regression:\n{cm_lr}')
print(f'\nAccuracy of Neural Network: {acc_nn}')
print(f'\nConfusion Matrix of Neural Network:\n{cm_nn}')
