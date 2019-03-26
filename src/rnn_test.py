from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import model_from_yaml
from keras import losses
import numpy as np
import tensorflow as tf
import os
import csv

tf.reset_default_graph()
#number of epochs for training
num_epochs = 1
#number of possible labels
n_labels = 3
#number of features
n_features = 4
#number of regressive data points
n_input = 10
#size of each epoch (i.e. batch)
batch_size = 64
#number of hidden units in input layer lstm cell
input_units = 500
#hidden layer number of units
hidden1_units = 300
hidden2_units = 275
hidden3_units = 200
dense_units = 10
#percentage to drop
dropout = 0.3
#initial reading of the IMU used for calculating drift error
x_error = y_error = 0
#initial orientation reading flag
initial_reading = 1
#flag to check time jump
last_time_stamp = 0
#numpy array for fault data
data = []
data_queue = {}
data_index = 0
data_iter = 0
data_num_skip = 10
data_skip = data_num_skip

def read_data_sets(name):
    read_data_set = []
    with open(name, "r") as fault_data:
        for row in csv.reader(fault_data):
            read_data_set = np.append(read_data_set, np.array(row))

    data = np.reshape(read_data_set, [int(len(read_data_set)/n_features), n_features])
    print(len(data))
    data = np.reshape(data, [int(len(data)/n_input), n_input, n_features])
    print(len(data))
    # shuffles the set of sequences
    np.random.shuffle(data)

    return np.asarray(data)

training_file = 'test_rnn.csv'
data_set = read_data_sets(training_file)
print("Loaded training data...")

# load YAML and create model
yaml_file = open('../config/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("../config/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

count = 0

for i in range(0, len(data_set)):
    features = data_set[i, :, :]
    print(np.reshape(features, (n_input, n_features)))
    prediction = loaded_model.predict(np.reshape(features, (1, features.shape[1], features.shape[0])))
    prediction = np.reshape(prediction, (1, 3))
    if(float(prediction[0,2]) > float(prediction[0,1]) and float(prediction[0,2]) > float(prediction[0,0])):
        print(prediction)
        count += 1

accuracy = count / len(data_set)
print("accuracy : ")
print(accuracy)
