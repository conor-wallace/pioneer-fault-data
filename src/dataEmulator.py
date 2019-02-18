import rospy
import sys
import fault_diagnostics
import numpy as np
from fault_diagnostics.msg import featureData

#number of features
n_features = 4
#number of regressive data points
n_input = 10

def read_data_sets(name):
    csv = np.genfromtxt (name, delimiter=",")
    read_data_set = np.array(csv[:,[i for i in range(0,n_features+1)]])
    return read_data_set

training_file = '../config/training_data.csv'
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
data_iter = len(np.reshape(features, (n_input, n_features)))
print data_iter

rospy.init_node('pioneer', anonymous=True)
#publishers
dataPub = rospy.Publisher("/feature_data", featureData, queue_size=10, tcp_nodelay=True)
rate = rospy.Rate(10)
rate.sleep()
data = fault_diagnostics.msg.featureData()

while not rospy.is_shutdown():
    data.timestamp1 = features[data_iter][0]
    data.timestamp2 = features[data_iter][1]
    data.timestamp3 = features[data_iter][2]
    data.timestamp4 = features[data_iter][3]
    data.timestamp5 = features[data_iter][4]
    data.timestamp6 = features[data_iter][5]
    data.timestamp7 = features[data_iter][6]
    data.timestamp8 = features[data_iter][7]
    data.timestamp9 = features[data_iter][8]
    data.timestamp10 = features[data_iter][9]
    dataPub.publish(data)
    rate.sleep()
    data_iter -= 1
    if data_iter == 0:
        sys.exit()
