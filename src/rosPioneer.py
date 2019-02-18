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
from p2os_msgs.msg import MotorState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

controller = 0

def controlCallback(data):
    global controller
    controller = float(data.data)

rospy.init_node('pioneer', anonymous=True)
#publishers
velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10, tcp_nodelay=True)
motPub = rospy.Publisher("/cmd_motor_state", MotorState, queue_size=10, tcp_nodelay=True)
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
