import numpy as np
from sklearn.model_selection import train_test_split
from alexnet import alexnet

print('Loading data...')
data = np.load('data.npy')
print('Data loaded.')

x = np.array([i[0] for i in data])
x = x.astype('float32')
x = x.reshape(-1, 50, 50, 1)

y = [i[1] for i in data]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1,
                                                    random_state=42)

model = alexnet(width=50, height=50,  lr=0.001, output=2, channel=1)

for j in range(100):

    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=1,
              validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, run_id='malaria_alexnet')

    model.save(f'malaria_{j+1}.model')
    print('Saved epoch: ' + str(j + 1))
