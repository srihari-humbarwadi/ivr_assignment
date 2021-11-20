# calculates forward kinematics and uses inverse kinematics to move robot towards desired trajectory

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray, Float64
from cv_bridge       import CvBridge

LINK_1_LENGTH = 4
LINK_3_LENGTH = 3.2
LINK_4_LENGTH = 2.8

# short-hand
L1 = LINK_1_LENGTH
L3 = LINK_3_LENGTH
L4 = LINK_4_LENGTH

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
# short-hand for sin and cos
def s(a):
    return np.sin(a)

def c(a):
    return np.cos(a)

# calculate the position of the end-effector from the angles of joints 1, 3, 4
def K(j1, j3, j4):
    x =  c(j1)*s(j4)*L4 + s(j1)*s(j3)*c(j4)*L4 + s(j1)*s(j3)*L3
    y =  s(j1)*s(j4)*L4 - c(j1)*s(j3)*c(j4)*L4 - c(j1)*s(j3)*L3
    z =  c(j3)*c(j4)*L4 + c(j3)*L3             + L1
    return [x, y, z]

# calculate the Jacobian from joint angles
#   the rows are "written as columns" for readability
def J(j1, j3, j4):
    first_row = np.array([
        -s(j1)*s(j4)*L4 + c(j1)*s(j3)*c(j4)*L4 + c(j1)*s(j3)*L3,
         s(j1)*c(j3)*c(j4)*L4 + s(j1)*c(j3)*L3,
         c(j1)*c(j4)*L4 - s(j1)*s(j3)*s(j4)*L4
         ])

    second_row = np.array([
         c(j1)*s(j4)*L4 + s(j1)*s(j3)*c(j4)*L4 + s(j1)*s(j3)*L3,
        -c(j1)*c(j3)*c(j4)*L4 - c(j1)*c(j3)*L3,
         s(j1)*c(j4)*L4 + c(j1)*s(j3)*s(j4)*L4
        ])

    third_row = np.array([
        0,
        -s(j3)*c(j4)*L4 - s(j3)*L3,
        -c(j3)*s(j4)*L4
        ])
    return np.stack([first_row, second_row, third_row], axis=0)