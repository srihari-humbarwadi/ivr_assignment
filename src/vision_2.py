#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - II' in the IVR assignment pdf

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray
from cv_bridge       import CvBridge

class vision_2:
    def __init__(self):
        rospy.init_node("vision_2", anonymous=True)
        self.image_2_sub      = rospy.Subscriber('image_topic2', Image, self.callback)
        self.joints_est_2_pub = rospy.Publisher("joints_est_2", Float64MultiArray, queue_size=10)

    def callback(self, data):
        # read image
        cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
        
        
        #kernel = np.array([ # for more control later
        #    [0, 0, 1, 0, 0],
        #    [0, 1, 1, 1, 0],
        #    [1, 1, 1, 1, 1],
        #    [0, 1, 1, 1, 0],
        #    [0, 0, 1, 0, 0]
        #], np.uint8)
        kernel    = np.full((5, 5), 1, np.uint8)
        no_iter   = 3
        no_joints = 5 #"including end-effector", which is normally excluded, so really this is off by 1
        no_links  = no_joints - 1 # this is not off by one but does not include "0m link from ground"
        
        # highlight blobs
        joints_colours = [
            [(0, 100, 0),   (10, 255, 10)],
            [(0, 100, 100), (0, 255, 255)],
            [(0, 100, 100), (0, 255, 255)],
            [(100, 0, 0),   (255, 0, 0) ],
            [(0, 0, 100),   (10, 10, 255)]
        ]
        joints_colours_names = ['green', 'yellow1', 'yellow2', 'blue', 'red'] # you're welcome
        
        joints_images = [
            cv2.dilate(
                cv2.inRange(cv_image, joints_colours[i][0], joints_colours[i][1]),
                kernel, iterations=no_iter
            )
        
            for i in range(no_joints)
        ]
        #links_images   = [cv2.imread("link%i.png" % i, 0) for i in [1, 2, 3]]
        
        # pull moments
        moments = [cv2.moments(img) for img in joints_images]
        
        # replace 0 by this small amount to prevent division by zero
        zero_guard = 0.0000001

        # get centres
        joints_centres = np.array([
            (
                int(moments[i]['m10'] / (moments[i]['m00'] if moments[i]['m00'] != 0 else zero_guard)), 
                int(moments[i]['m01'] / (moments[i]['m00'] if moments[i]['m00'] != 0 else zero_guard))
            )
            for i in range(no_joints)
        ])
        
        # calculate angles via trigonometry
        joints_angles = [0] * (no_joints - 1)
        for i in range(no_joints - 1):
            joints_angles[i] = np.arctan2(
                joints_centres[i][0] - joints_centres[i+1][0],
                joints_centres[i][1] - joints_centres[i+1][1]
            ) - sum(joints_angles[:i])

        joints_est = Float64MultiArray()
        joints_est.data = joints_angles
        self.joints_est_2_pub.publish(joints_est)

def main(args):
    _ = vision_2()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)