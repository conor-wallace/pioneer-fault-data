from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.optimizers import Adam
from keras import losses
import numpy as np
import tensorflow as tf
import os
import csv

class NeuralNetwork:

    def __init__(self):
        print("Setting new Neural Network")
        tf.reset_default_graph()
        #number of possible labels
        self.n_labels = 3
        #number of features
        self.n_features = 4
        #number of regressive data points
        self.n_input = 10
        self.input_units = 6
        self.hidden_units = 6
        self.model = Sequential()

    def create_model(self, weights):
        #Input Layer
        self.model.add(Dense(self.input_units, activation='sigmoid', weights=[weights[0],weights[1]]))
        #Hidden Layers
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.hidden_units, activation='sigmoid', weights=[weights[2],weights[3]]))
        self.model.add(Dropout(0.2))
        #Output Layer
        self.model.add(Dense(self.n_labels, activation='softmax', weights=[weights[4],weights[5]]))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def read_data_sets(self, name):
        read_data_set = []
        with open(name, "r") as fault_data:
            for row in csv.reader(fault_data):
                read_data_set = np.append(read_data_set, np.array(row))
        return read_data_set

    def generate_train_test_data_sets(self, data_set):
        # divides the entire data set into sequences of 10 data points
        data = np.reshape(data_set, [int(len(data_set)/5), self.n_features+1])
        print(len(data))
        #data = np.reshape(data, [int(len(data)/n_input), n_input, n_features+1])
        print(data.shape)
        # shuffles the set of sequences
        np.random.shuffle(data)

        # takes the tire pressure classification from every 10th data point in every sequence
        seg_label_data = data[:, self.n_features]
        # takes the feature data parameters from every sequence
        seg_feature_data = data[:, :self.n_features]

        return np.asarray(seg_feature_data), np.asarray(seg_label_data)

    def predict(self, features):
        prediction = self.model.predict(np.reshape(features, (1, self.n_features)))
        return prediction
        '''
        weight_origin_0=self.model.layers[0].get_weights()[0]
        weight_origin_1=self.model.layers[2].get_weights()[0]
        weight_origin_2=self.model.layers[4].get_weights()[0]

        print("model shape")
        print(np.asarray(self.model.layers).shape)
        print("Input Layer")
        print(np.asarray(weight_origin_0))
        print("Hidden Layer")
        print(np.asarray(weight_origin_1))
        print("Output Layer")
        print(np.asarray(weight_origin_2))
        '''
