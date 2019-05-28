import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import rospy
import std_msgs
from std_msgs.msg import Float64
import csv

def read_data_sets(name):
    read_data_set = []
    with open(name, "r") as fault_data:
        for row in csv.reader(fault_data):
            read_data_set = np.append(read_data_set, np.array(row))
    return read_data_set

training_file = 'fuzzyResults.csv'
data_set = read_data_sets(training_file)


time = list(xrange(40))
print(time)
print(data_set)
plt.plot(np.array(time), np.array(data_set[:40]),'.-')
plt.title('FLC Accuracy, Left Flat')
plt.xlabel('timesteps')
plt.ylabel('compensation (degrees)')

plt.show()
