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
import rospy
import std_msgs
from std_msgs.msg import Float64

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
    return read_data_set

training_file = '../config/testCenter.csv'
data_set = read_data_sets(training_file)
print("Loaded training data...")

def generate_train_test_data_sets():
    # divides the entire data set into sequences of 10 data points
    #data = np.reshape(data_set, [int(len(data_set)/5), n_features+1])
    #print(data)
    #data = data[int((len(data)/20)+1):]
    #data = np.reshape(data, [int(len(data)/n_input), n_input, n_features+1])
    #print(len(data))
    #print(np.as_array(data_set.shape()))
    data = np.reshape(data_set, [int(len(data_set)/5), n_features+1])
    print(len(data))
    data = np.reshape(data, [int(len(data)/n_input), n_input, n_features+1])
    print(len(data))
    # shuffles the set of sequences
    np.random.shuffle(data)

    # takes the tire pressure classification from every 10th data point in every sequence
    seg_label_data = data[:, n_input-1, n_features]
    # takes the feature data parameters from every sequence
    seg_feature_data = data[:, :, :n_features]

    return np.asarray(seg_feature_data), np.asarray(seg_label_data)

features, labels_norm = generate_train_test_data_sets()
print(len(features))
print(features)
labels = []
# convert label data to one_hot arrays
for k in range(0, len(labels_norm)):
    one_hot_label = np.zeros([3], dtype=float)
    one_hot_label[int(float(labels_norm[k]))] = 1.0
    labels = np.append(labels, one_hot_label)
labels = np.reshape(labels, [-1, n_labels])

# load YAML and create model
yaml_file = open('../config/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("../config/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

rospy.init_node('fault_predictor', anonymous=True)
#publishers
predictionPub = rospy.Publisher("/prediction", Float64, queue_size=1, tcp_nodelay=True)
steeringPub = rospy.Publisher("/steering", Float64, queue_size=1, tcp_nodelay=True)
rate = rospy.Rate(5)
predictionMsg = std_msgs.msg.Float64()
steeringMsg = std_msgs.msg.Float64()

i = 0
while not rospy.is_shutdown():
    if(i < len(features)):
        feature_batch = features[i]
        prediction = loaded_model.predict(np.reshape(feature_batch, (1, feature_batch.shape[1], feature_batch.shape[0])))
        center_prob = prediction[0][0]*100
        left_prob = prediction[0][1]*100
        right_prob = prediction[0][2]*100

        if(center_prob > left_prob and center_prob > right_prob):
            print("center: %s" % center_prob)
            predictionMsg = center_prob
            steeringMsg = 0.0
        elif(right_prob > center_prob and right_prob > left_prob):
            print("right: %s" % right_prob)
            predictionMsg = right_prob
            steeringMsg = 1.0
        elif(left_prob > center_prob and left_prob > right_prob):
            print("left: %s" % left_prob)
            predictionMsg = left_prob
            steeringMsg = 2.0
        i += 1
        predictionPub.publish(predictionMsg)
        steeringPub.publish(steeringMsg)
        rate.sleep()
    else:
        predictionMsg = -1
        steeringMsg = -1
        predictionPub.publish(predictionMsg)
        steeringPub.publish(steeringMsg)
        rate.sleep()
	break
