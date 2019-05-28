from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
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

    def read_data_sets(self, name):
        read_data_set = []
        with open(name, "r") as fault_data:
            for row in csv.reader(fault_data):
                read_data_set = np.append(read_data_set, np.array(row))
        return read_data_set

    def generate_train_test_data_sets(self, data_set):
        # divides the entire data set into sequences of 10 data points
        #data = np.reshape(data_set, [int(len(data_set)/5), n_features+1])
        #print(data)
        #data = data[int((len(data)/20)+1):]
        #data = np.reshape(data, [int(len(data)/n_input), n_input, n_features+1])
        #print(len(data))
        #print(np.as_array(data_set.shape()))
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

    def create_model(self):
        #Input Layer
        input_weights=np.random.rand(4, 6) #weight
        input_biases=np.random.rand(6) #biases
        self.model.add(Dense(self.input_units, activation='relu', weights=[input_weights,input_biases]))
        #Hidden Layers
        hidden_weights=np.random.rand(6, 6) #weight
        hidden_biases=np.random.rand(6) #biases
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.hidden_units, activation='relu', weights=[hidden_weights,hidden_biases]))
        self.model.add(Dropout(0.2))
        #Output Layer
        output_weights=np.random.rand(6, 3) #weight
        output_biases=np.random.rand(3) #biases
        self.model.add(Dense(self.n_labels, activation='relu', weights=[output_weights,output_biases]))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def predict(self, features):
        prediction = self.model.predict(np.reshape(features[0], (1, self.n_features)))

        #history = model.fit(np.reshape(features, (features.shape[0], n_features)),labels, batch_size=batch_size, epochs=num_epochs, validation_split= .3)

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

        print(prediction)
