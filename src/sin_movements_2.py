#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64

# publishes to topics to tell joints to move in sinusoidal movements
#   for Joint State Estimation II, as written in the IVR assignment pdf

def joint1_movement(t):
    return np.pi   * np.sin(t * np.pi/28)

def joint3_movement(t):
    return np.pi/2 * np.sin(t * np.pi/20)

def joint4_movement(t):
    return np.pi/2 * np.sin(t * np.pi/18)

def sin_movements_2():
    pub_str = '/robot/joint%i_position_controller/command'

    # initialize node and publishers to joint angles
    rospy.init_node('sin_movements_1', anonymous=True)
    joint1_pub = rospy.Publisher(pub_str % 1 , Float64, queue_size=10)
    joint3_pub = rospy.Publisher(pub_str % 3 , Float64, queue_size=10)
    joint4_pub = rospy.Publisher(pub_str % 4 , Float64, queue_size=10)

    rate = rospy.Rate(50)
    start_time = rospy.get_time()
    while not rospy.is_shutdown():
        time_from_start = rospy.get_time() - start_time
        j1 = Float64()
        j3 = Float64()
        j4 = Float64() 
        
        j1.data = joint1_movement(time_from_start)
        j3.data = joint3_movement(time_from_start)
        j4.data = joint4_movement(time_from_start)

        joint1_pub.publish(j1)
        joint3_pub.publish(j3)
        joint4_pub.publish(j4)
        
        rate.sleep()




if __name__ == '__main__':
    try:
        sin_movements_2()
    except:
        pass

