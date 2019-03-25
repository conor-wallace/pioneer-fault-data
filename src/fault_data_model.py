
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
import sensor_msgs
import fault_diagnostics
from sensor_msgs.msg import Imu
from fault_diagnostics.msg import Array, featureData

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

# load YAML and create model
yaml_file = open('../config/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("../config/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def generate_train_test_data_sets(iter):
    global data_queue, data_iter
    feature_data = data_queue[iter]

    feature_data = np.reshape(feature_data, [n_input,n_features])
    i = 0
    #This could be a possible flaw
    for i in range(len(feature_data)):
        feature_data[i][1] = feature_data[i][1] - x_error
        feature_data[i][2] = feature_data[i][2] - y_error

    return np.asarray(feature_data)

def faultCallback(cdata):
    global initial_reading, data, data_queue, data_index, last_time_stamp, x_error, y_error, n_input, data_skip, data_num_skip
    if initial_reading == 1:
        x_error = cdata.orientation.x
        y_error = cdata.orientation.y
        initial_reading = 0

    if data_skip > 0:
        data_skip -= 1
        return
    data = np.append(data, np.asarray([cdata.header.seq, cdata.orientation.x, cdata.orientation.y, 0.2]))
    data = np.reshape(data, [-1, n_features])
    if len(data) >= n_input:
        data_queue[data_index] = data[-n_input:]
        size = len(data)
        data = np.delete(data, slice(0, size), axis=0)
        #print("Deleted Data")
        data_index += 1
        data_skip = data_num_skip
    if (cdata.header.seq - last_time_stamp > 1):
        print('WARNING! steps since last time stamp:')
        print(cdata.header.seq - last_time_stamp, cdata.header.seq, last_time_stamp)
    last_time_stamp = cdata.header.seq


rospy.init_node('faultDiagnostics', anonymous=True)
model_pub = rospy.Publisher('/prediction', Array, queue_size=1000, tcp_nodelay=True)
dataPub = rospy.Publisher("/feature_data", featureData, queue_size=1000, tcp_nodelay=True)
rospy.Subscriber("/imu", Imu, faultCallback)
fault_prediction = fault_diagnostics.msg.Array()
fault_data = fault_diagnostics.msg.featureData()
rate = rospy.Rate(20)
rate.sleep()
while not rospy.is_shutdown():
    if len(data_queue) > 0:
        features = generate_train_test_data_sets(data_queue.keys()[0])
        prediction = loaded_model.predict(np.reshape(features, (1, features.shape[1], features.shape[0])))
        fault_prediction.full = prediction[0][0]
        fault_prediction.right_low = prediction[0][1]
        fault_prediction.left_low = prediction[0][2]
        fault_data.timestamp1 = features[0][:]
        fault_data.timestamp2 = features[1][:]
        fault_data.timestamp3 = features[2][:]
        fault_data.timestamp4 = features[3][:]
        fault_data.timestamp5 = features[4][:]
        fault_data.timestamp6 = features[5][:]
        fault_data.timestamp7 = features[6][:]
        fault_data.timestamp8 = features[7][:]
        fault_data.timestamp9 = features[8][:]
        fault_data.timestamp10 = features[9][:]
        dataPub.publish(fault_data)
        model_pub.publish(fault_prediction)
        rate.sleep()
        data_queue.pop(data_queue.keys()[0], None)
    else:
        pass
    if len(data_queue) > 5:
        print("GTFO")
