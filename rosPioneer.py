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
from p2os_msgs.msg import MotorState
from geometry_msgs.msg import Twist

def talker():
    #publishers
    velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    motPub = rospy.Publisher("/cmd_motor_state", MotorState, queue_size=10)

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)

    vel = geometry_msgs.msg.Twist()

    vel.linear.x = 0.2
    vel.angular.z = 0.0
    count = 0

    while not rospy.is_shutdown():
        velPub.publish(vel)
        rate.sleep()
        count = count + 1
        if count == 200:
            vel.linear.x = 0.0
            vel.angular.z = 0.0
            velPub.publish(vel)
            rospy.loginfo("target reached")
            break

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
