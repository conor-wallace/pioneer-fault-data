from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from keras import losses
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
#number of epochs for training
num_epochs = 50
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

def read_data_sets(name):
    csv = np.genfromtxt (name, delimiter=",")
    read_data_set = np.array(csv[:,[i for i in range(0,n_features+1)]])
    return read_data_set

training_file = 'training_data.csv'
data_set = read_data_sets(training_file)
print("Loaded training data...")

def generate_train_test_data_sets():
    # divides the entire data set into sequences of 10 data points
    data = np.reshape(data_set, [len(data_set)/n_input,n_input,n_features+1])
    # shuffles the set of sequences
    np.random.shuffle(data)

    # takes the tire pressure classification from every 10th data point in every sequence
    seg_label_data = data[:, n_input-1, n_features]
    # takes the feature data parameters from every sequence
    seg_feature_data = data[:, :, :n_features]

    return np.asarray(seg_feature_data), np.asarray(seg_label_data)

features, labels_norm = generate_train_test_data_sets()
labels = []
# convert label data to one_hot arrays
for k in range(0, len(labels_norm)):
    one_hot_label = np.zeros([3], dtype=float)
    one_hot_label[int(labels_norm[k])] = 1.0
    labels = np.append(labels, one_hot_label)
labels = np.reshape(labels, [-1, n_labels])

model_dir = 'fault_model'

# Create LSTM
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
#model.add(Embedding(batch_size, timesteps, input_length=data_dim))
model.add(LSTM(input_units, input_shape=(n_features, n_input), return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(hidden1_units, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(hidden2_units, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(hidden3_units))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(n_labels,  activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.reshape(features, (features.shape[0], n_features, features.shape[1])),labels, batch_size=batch_size, epochs=num_epochs, validation_split= .3)
