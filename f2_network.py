'''
features:   float orientation.x, float orientation.y, float velocity.linear.x
labels: tire_pressure_class

3 labels
each label represents the inflation of each tire
n_labels[0] = [1, 0, 0] = all full
n_labels[1] = [0, 1, 0] = right tires flat
n_labels[2] = [0, 0, 1] = left tires flat

4 features
time_stamp x_orient    y_orient velocity pressure_class
1          207.375     5.375    0.2      0
2          207.375     5.375    0.2      0
3          207.375     5.375    0.2      0
'''

import tensorflow as tf
import numpy as np
import os
import csv
import random
from tensorflow.contrib import rnn

tf.reset_default_graph()
training_file = 'training_data.csv'

#number of epochs for training
num_epochs = 500
#number of possible labels
n_labels = 3
#number of features
n_features = 4
#number of regressive data points
n_input = 10
#size of each epoch (i.e. batch)
batch_size = 16
#number of hidden units in input layer lstm cell
input_units = 5
#number of hidden units in multi-layer lstm cells
hidden_units = 3
dense_units = 10
#number of lstm layers
num_layers = 1
#percentage to drop
dropout = 0.5

# tf Graph input: a single line of features
x = tf.placeholder('float', [None, n_input, n_features])
y = tf.placeholder('float')

print('BATCH SIZE: {:d} HIDDEN UNITS: {:d}  NUMBER OF EPOCHS: {:d}'.format(batch_size,hidden_units,num_epochs))

def read_data_sets(name):
    csv = np.genfromtxt (name, delimiter=",")
    read_data_set = np.array(csv[:,[i for i in range(0,n_features+1)]])
    return read_data_set

data_set = read_data_sets(training_file)
print len(data_set)
print("Loaded training data...")

def generate_train_test_data_sets():
    # divides the entire data set into sequences of 10 data points
    data = np.reshape(data_set, [len(data_set)/n_input,n_input,n_features+1])
    print len(data)
    print data
    # shuffles the set of sequences
    np.random.shuffle(data)

    # takes the tire pressure classification from every 10th data point in every sequence
    seg_label_data = data[:, n_input-1, n_features]
    print len(seg_label_data)
    print seg_label_data
    # takes the feature data parameters from every sequence
    seg_feature_data = data[:, :, :n_features]
    print len(seg_feature_data)
    print seg_feature_data

    #split data set into train and test: ex = 7 -> train = 70% & test = 30%
    portion = 0.7
    training_data_features = seg_feature_data[:int(len(seg_feature_data)*portion)]
    training_data_labels = seg_label_data[:int(len(seg_label_data)*portion)]
    testing_data_features = seg_feature_data[int(len(seg_feature_data)*portion):]
    testing_data_labels = seg_label_data[int(len(seg_label_data)*portion):]

    return np.asarray(training_data_features), np.asarray(training_data_labels), np.asarray(testing_data_features), np.asarray(testing_data_labels)

train_features, train_labels, test_features, test_labels = generate_train_test_data_sets()

print len(train_features)
print len(train_labels)

def next_batch(size, feature_name, label_name, offset):
    feature_batch = []
    label_batch = []
    # grab a batch of data
    for i in range(offset, offset + size):
        feature_batch = np.append(feature_batch, feature_name[i])
        label_batch = np.append(label_batch, label_name[i])

    labels = []
    # convert label data to one_hot arrays
    for k in range(0, size):
        one_hot_label = np.zeros([3], dtype=float)
        one_hot_label[int(label_batch[k])] = 1.0
        labels = np.append(labels, one_hot_label)

    return np.asarray(feature_batch), np.asarray(labels)

def recurrent_neural_network(x):
    #weights and biases based on the number of possible labels
    layer = {'weights':tf.Variable(tf.random_normal([hidden_units,n_labels])),
             'biases':tf.Variable(tf.random_normal([n_labels]))}

    print(x)
    #reshape data to fit model
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, n_features])
    x = tf.split(x, n_input, 0)

    #input LSTM cell
    lstm_cell = rnn.BasicLSTMCell(input_units, state_is_tuple=True)
    #multi-layer LSTM cells with dropout
    lstm_cell = rnn.MultiRNNCell([make_cell(hidden_units) for _ in range(num_layers)], state_is_tuple=True)
    #lstm_cell = tf.reshape(lstm_cell, [-1,n_features])
    #dense = tf.layers.dense(inputs=lstm_cell, units=dense_units, activation=tf.nn.relu)
    #make prediction
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.layers.dense(tf.layers.batch_normalization(outputs[:, -1, :]),num_classes, activation=None, kernel_initializer=tf.orthogonal_initializer())
    #instantiate new LSTM cell
    #cell = rnn.LSTMCell(hidden_units, state_is_tuple=True)
    #cell = rnn.DropoutWrapper(cell, output_keep_prob=0.5)
    #cell = rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    #model: ([final outputs]*[weights] + biases)

    #output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    #return output

def make_cell(lstm_size):
    cell = rnn.BasicLSTMCell(hidden_units, state_is_tuple=True)
    cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    return cell

"""
Concept: batch through the entire dataset in small batches randomly spaced throughout
the dataset. The batch size will be ten features which we will then be able to
predict the eleventh datapoint as a label using one_hot encoding. This means that
if the dataset is 5000 lines long, we will send 500 batches through the network.
"""

def fit_network(x):
    with tf.variable_scope("model_fn") as scope:
        train_prediction = recurrent_neural_network(x)
        #tv = tf.trainable_variables()
        #regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        #cost = tf.reduce_sum(tf.pow(train_prediction - y, 2)) + regularization_cost
        #optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=train_prediction,labels=y) )
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(num_epochs+1):
                epoch_loss = 0
                train_offset = 0
                for _ in range(int((len(train_features))/batch_size)):
                    #retrieve a batch of data from the dataset
                    train_x, train_y_onehot = next_batch(batch_size, train_features, train_labels, train_offset)
                    #reshape it to fit inside tensor
                    train_x = np.reshape(train_x, [batch_size, n_input, n_features])
                    train_y_onehot = np.reshape(train_y_onehot, (batch_size, n_labels))
                    #optimize and compute cost
                    _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y_onehot})
                    epoch_loss += c
                    train_offset += batch_size

                print('Epoch {:d} completed out of {:d} Loss: {:f}'.format(epoch, num_epochs, epoch_loss))

            correct = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x: train_x, y: train_y_onehot}))

    print('Testing Started')
    with tf.variable_scope(scope, reuse=True):
        test_prediction = recurrent_neural_network(x)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=test_prediction,labels=y) )

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(num_epochs+1):
                epoch_loss = 0
                test_offset = 0
                for _ in range(int((len(test_features))/batch_size)):
                    #retrieve a batch of data from the dataset
                    test_x, test_y_onehot = next_batch(batch_size, test_features, test_labels, test_offset)
                    #reshape it to fit inside tensor
                    test_x = np.reshape(test_x, [batch_size, n_input, n_features])
                    test_y_onehot = np.reshape(test_y_onehot, (batch_size, n_labels))

                    #compute cost
                    c = sess.run(cost, feed_dict={x: test_x, y: test_y_onehot})
                    epoch_loss += c
                    test_offset += batch_size

                print('Epoch {:d} completed out of {:d} Loss: {:f}'.format(epoch, num_epochs, epoch_loss))

            correct = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x: test_x, y: test_y_onehot}))


print("Training Model")
fit_network(x)
