# calculates forward kinematics and uses inverse kinematics to move robot towards desired trajectory

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray, Float64
from cv_bridge       import CvBridge

class control:
    def __init__(self):
        # subscribe to relevant topics
        self.joint_1_sub = rospy.Subscriber('joint_angle_1', Float64, queue_size=20)
        self.joint_3_sub = rospy.Subscriber('joint_angle_3', Float64, queue_size=20)
        self.joint_4_sub = rospy.Subscriber('joint_angle_4', Float64, queue_size=20)
        self.target_sub  = rospy.Subscriber('/target_control/target_pos', Float64MultiArray, queue_size=20)

        # prepare publishers for joint angles
        self.joint_1_pub = rospy.Publisher('/robot/joint1_position_controller/command', Float64, queue_size=20)
        self.joint_3_pub = rospy.Publisher('/robot/joint3_position_controller/command', Float64, queue_size=20)
        self.joint_4_pub = rospy.Publisher('/robot/joint4_position_controller/command', Float64, queue_size=20)

# TODO
# forward kinematics