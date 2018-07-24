import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

x, y, test_x, test_y = mnist.load_data(one_hot=True)

x = x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# Input layer
network = input_data(shape=[None, 28, 28, 1], name='Input_Layer')

# Hidden Layer 1
network = fully_connected(network, 500, activation='relu', name='Hidden_layer_1')
network = dropout(network, 0.95, name='Dropout_1')

# Hidden Layer 2
network = fully_connected(network, 500, activation='relu', name='Hidden_layer_2')
network = dropout(network, 0.95, name='Dropout_2')

# Hidden Layer 3
network = fully_connected(network, 500, activation='relu', name='Hidden_layer_3')
network = dropout(network, 0.95, name='Dropout_3')

# Output
network = fully_connected(network, 10, activation='softmax', name='Output_Layer')
network = regression(network, name='targets')

model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='log')

model.fit({'Input_Layer': x}, {'targets': y}, n_epoch=10,
          validation_set = ({'Input_Layer' : test_x}, {'targets': test_y}),
          show_metric=True, run_id='mnist_mycode')

# model.save('Mnist_tflearn_mycode.model')



