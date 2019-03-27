import rospy
import numpy as np
import message_filters
import sensor_msgs
import fault_diagnostics
import nav_msgs
from sensor_msgs.msg import Imu
from fault_diagnostics.msg import featureData
from nav_msgs.msg import Odometry

#number of features
n_features = 6
#number of regressive data points
n_input = 10
#initial reading of the IMU used for calculating drift error
x_error = y_error = t_error = 0
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
# classification for tire pressure: 0=all full, 1=left flat, 2=right flat
test = 2

def generate_train_test_data_sets(iter):
    global data_queue, data_iter
    feature_data = data_queue[iter]

    feature_data = np.reshape(feature_data, [n_input,n_features+1])
    i = 0
    #Normalize time stamp and x, y orientation data
    for i in range(len(feature_data)):
        feature_data[i][0] = feature_data[i][0] - t_error
        feature_data[i][1] = feature_data[i][1] - x_error
        feature_data[i][2] = feature_data[i][2] - y_error

    return np.asarray(feature_data)

def callback(imu_data, odom_data):
    global initial_reading, data, data_queue, data_index, last_time_stamp, x_error, y_error, n_input, data_skip, data_num_skip, test
    if initial_reading == 1:
        t_error = imu_data.header.seq
        x_error = imu_data.orientation.x
        y_error = imu_data.orientation.y
        initial_reading = 0

    if data_skip > 0:
        data_skip -= 1
        return
    data = np.append(data, np.asarray([imu_data.header.seq, odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, imu_data.orientation.x, imu_data.orientation.y, 0.2, test]))
    data = np.reshape(data, [-1, n_features+1])
    if len(data) >= n_input:
        data_queue[data_index] = data[-n_input:]
        size = len(data)
        data = np.delete(data, slice(0, size), axis=0)
        #print("Deleted Data")
        data_index += 1
        data_skip = data_num_skip
    if (imu_data.header.seq - last_time_stamp > 1):
        print('WARNING! steps since last time stamp:')
        print(imu_data.header.seq - last_time_stamp, imu_data.header.seq, last_time_stamp)
    last_time_stamp = imu_data.header.seq

rospy.init_node('dataCollector', anonymous=True)
dataPub = rospy.Publisher("/feature_data", featureData, queue_size=1000, tcp_nodelay=True)
imu_sub = message_filters.Subscriber("/imu", Imu)
odom_sub = message_filters.Subscriber(ns+"/pose", Odometry)

ts = message_filters.TimeSynchronize([imu_sub, odom_sub], 10)
ts.registerCallback(callback)

fault_data = fault_diagnostics.msg.featureData()
rate = rospy.Rate(20)
rate.sleep()
while not rospy.is_shutdown():
    if len(data_queue) > 0:
        features = generate_train_test_data_sets(data_queue.keys()[0])
        print('features:')
        print np.reshape(features, (n_input, n_features+1))
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
        rate.sleep()
        data_queue.pop(data_queue.keys()[0], None)
    else:
        #print("No Data")
        pass
