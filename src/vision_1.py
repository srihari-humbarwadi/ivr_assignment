#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - I' in the IVR assignment pdf

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray
from cv_bridge       import CvBridge

# replace 0 with arbitrarily small value
def zero_guard(val):
    # replace 0 by this small amount to prevent division by zero
    zero_guard_val = 0.001
    return zero_guard_val if val == 0 else val

class Joint:
    # colour range for opencv binary thresholding
    def __init__(self, colour_name, colour_range):
        self.x = 0
        self.y = 0
        self.z = 0
        self.angle = 0
        self.colour_name = colour_name
        self.colour_range = colour_range

class vision_1:
    def __init__(self):
        # set up ros nodes/pubs/subs etc.
        rospy.init_node("vision_1", anonymous=True)
        self.image_1_sub      = rospy.Subscriber('/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub      = rospy.Subscriber('/camera2/robot/image_raw', Image, self.callback_2)
        self.joints_est_1_pub = rospy.Publisher("joints_est_1", Float64MultiArray, queue_size=10)
        self.bridge = CvBridge()

        # set up kernel and etc. for blob detection
        #self.kernel = np.array([ # for more control later
        #    [0, 0, 1, 0, 0],
        #    [0, 1, 1, 1, 0],
        #    [1, 1, 1, 1, 1],
        #    [0, 1, 1, 1, 0],
        #    [0, 0, 1, 0, 0]
        #], np.uint8)
        self.kernel    = np.full((5, 5), 1, np.uint8)
        self.no_iter   = 3
        self.no_joints = 5 #"including end-effector", so off by 1
        self.no_links  = self.no_joints - 1 # not off by one, does not include "0m link from ground"

        # consider blob blocked (e.g. by link) if its area m00 is below this threshold
        self.obstruct_thres = 50

        # declare joints and their (range of) colours (for opencv thresholding)
        self.green = Joint('green',  [(0, 100, 0),   (10, 255, 10)])
        self.yel1  = Joint('yellow', [(0, 100, 100), (0, 255, 255)])
        self.yel2  = Joint('yellow', [(0, 100, 100), (0, 255, 255)])
        self.blue  = Joint('blue',   [(100, 0, 0),   (255, 0, 0) ])
        self.red   = Joint('red',    [(0, 0, 100),   (10, 10, 255)])
        self.joints = [self.green, self.yel1, self.yel2, self.blue, self.red] #index 1-off from pdf



    # handle images seen from the camera facing the yz-plane
    def callback_1(self, data):
        self.detect_centres(data, 1)
        # calculate angles via trigonometry and linear algebra
        # accuracy can be improved, but problems with oscillations due to obstruction
        self.yel2.angle = np.arctan2(
            self.yel2.y - self.blue.y,
            (self.yel2.z - self.blue.z) #/ zero_guard(np.cos(self.yel1.angle))
        )
        self.publish_angles()

    # handle images seen from the camera facing the xz-plane
    def callback_2(self, data):
        self.detect_centres(data, 2)
        
        # calculate angles via trigonometry and linear algebra
        # line below makes signs accurate but problems with oscillations when joints_angles[2] fluctuates fast
        #cosj2_sign = np.sign(zero_guard(np.cos(self.joints_angles[2])))
        self.yel1.angle = np.arctan2( #negated because y axis points away from screen
            -(self.yel1.x - self.blue.x), #* cosj2_sign,
            (self.yel1.z - self.blue.z)   #* cosj2_sign
        )

        # calculate cos and sin of blue joint from dot and cross proucts of links (as vectors)
        link_3 = np.array([self.blue.x - self.yel1.x, self.blue.y - self.yel1.y, self.blue.z - self.yel1.z])
        link_4 = np.array([self.red.x - self.blue.x, self.red.y - self.blue.y, self.red.z - self.blue.z])
        link_norm_prod = zero_guard(np.linalg.norm(link_3) * np.linalg.norm(link_4))

        cos_blue = np.dot(link_3, link_4) / link_norm_prod
        sin_blue = np.linalg.norm(np.cross(link_3, link_4)) / link_norm_prod
        # taking the norm eliminates sign, so need to re-derive it
        self.blue.angle = np.arctan2(np.sign(self.yel2.angle)*sin_blue, cos_blue)

        self.publish_angles()

    def detect_centres(self, data, camera):
        # read image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        # highlight blobs
        joints_images = [
            cv2.dilate(
                cv2.inRange(cv_image, j.colour_range[0], j.colour_range[1]),
                self.kernel, iterations=self.no_iter
            )
            for j in self.joints
        ]
        # pull moments
        moments = [cv2.moments(img) for img in joints_images]

        # update centres, using previous pos. if obscured
        for i in range(self.no_joints):
            area       = moments[i]['m00']
            if area < self.obstruct_thres:
                continue
            vertical   = int(moments[i]['m01'] / zero_guard(area))
            horizontal = int(moments[i]['m10'] / zero_guard(area))
            
            self.joints[i].z     = vertical
            if camera == 1:
                self.joints[i].y = horizontal
            else:
                self.joints[i].x = horizontal

    def publish_angles(self):
        joints_est = Float64MultiArray()
        joints_est.data = [self.green.angle, self.yel1.angle, self.yel2.angle, self.blue.angle]
        self.joints_est_1_pub.publish(joints_est)

def main(args):
    _ = vision_1()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)