#!/usr/bin/env python
import rospy
sub = rospy.Subscriber('imu', Imu, processIMU_message)
def processIMU_message(imuMsg):
    print(imuMsg)
rospy.spin()
