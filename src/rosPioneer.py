#!/usr/bin/env python

#calibrate the imu sensor
#Use rosserial to sync the cmd_vel time stamps to the imu stime stamps
#Edit the imu_pose.txt file to include test velocity and tire pressure

#take the data from the imu and publish it in ros using rosserial
#imu data are degree orientations. If the pioneer is going perfectly straight,
#all values except y should be zero.
#sync the calibration of the imu to start the system

#imu data file reads as:
#x: y: z: vel: press1: press2: press3: press4: time:

import rospy
import geometry_msgs
import nav_msgs
import p2os_msgs
import std_msgs
import tf
import csv
import sys
from p2os_msgs.msg import MotorState
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float64

test = sys.argv[1]
seq = 0

def controlCallback(data):
    global controller
    x = data.position.x
    y = data.position.y
    z = data.position.z
    quatrenion = (data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w)
    euler = tf.transformations.euler_from_quatrenion(quatrenion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    with open('lighthouse_data.csv', mode='a') as lighthouse_csvfile:
        lighthouse_csv_writer = csv.writer(lighthouse_csvfile, delimiter=',')
        train_csv_writer.writerow([x, y, z, roll, pitch, yaw, seq, test])
    seq += 1

rospy.init_node('pioneer', anonymous=True)
#publishers
velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10, tcp_nodelay=True)
motPub = rospy.Publisher("/cmd_motor_state", MotorState, queue_size=10, tcp_nodelay=True)
rospy.Subscriber("/pose", Pose, controlCallback)
#rospy.Subscriber('/fuzzy', Float64, controlCallback)
rate = rospy.Rate(10)
rate.sleep()
vel = geometry_msgs.msg.Twist()

target = 200
count = 0
#derivative of vel.linear.z control
#dt = target/rate
#dz = controller/dt

while not rospy.is_shutdown():
    vel.linear.x = 0.2
    #vel.linear.z = 0.0
    velPub.publish(vel)
    rate.sleep()
    count = count + 1
    if count == target:
        vel.linear.x = 0.0
        #vel.linear.z = 0.0
        velPub.publish(vel)
        rate.sleep()
        rospy.loginfo("target reached")
        break
