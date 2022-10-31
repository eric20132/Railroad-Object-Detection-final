#!/usr/bin/env python3
import rospy
import math
import wx
import zmq
from builtins import bytes


from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")


#Create shutdown hook to kill visual displays
def shutdown_hook():
    #print "Killing displays"
    wx.Exit()

#register shutdown hook
rospy.on_shutdown(shutdown_hook)
rospy.init_node("display_3D_visualization_node")

def processIMU_message(imuMsg):
    topic = 1
    print(imuMsg.linear_acceleration)
    time = imuMsg.header.stamp.nsecs
    print(imuMsg.linear_acceleration_covariance)
    x = imuMsg.linear_acceleration.x
    y = imuMsg.linear_acceleration.y
    z = imuMsg.linear_acceleration.z
    #to_send = time + ':' + x + ':' + y + ':' + z
    socket.send_pyobj({1:[time,x,y,z]})
    
    
    
sub = rospy.Subscriber('imu', Imu, processIMU_message)
rospy.spin()
