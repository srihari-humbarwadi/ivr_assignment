#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - II' in the IVR assignment pdf

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray
from cv_bridge       import CvBridge

def near_zero(val):
    return abs(val) < 0.05

# return the argument (x or y) closer to val
def closer_to(val, x, y):
    if abs(x - val) <= abs(y - val):
        return x
    else:
        return y

# replace 0 with arbitrarily small value
def zero_guard(val):
    # replace 0 by this small amount to prevent division by zero
    zero_guard_val = 0.001
    return zero_guard_val if val == 0 else val

class Link:
    # model the link as a vector from joint1 to joint2
    def __init__(self, joint_1, joint_2):
        self.joint_1 = joint_1
        self.joint_2 = joint_2

    def get_x(self):
        return self.joint_2.x - self.joint_1.x

    def get_y(self):
        return self.joint_2.y - self.joint_1.y

    def get_z(self):
        # negated from perspective of cameras
        return self.joint_1.z - self.joint_2.z

    def as_normalized(self):
        vect = np.array([self.get_x(), self.get_y(), self.get_z()])
        norm = np.linalg.norm(vect)
        return vect / norm

class Joint:
    # colour range for opencv binary thresholding
    def __init__(self, colour_name, colour_range):
        self.x            = 0
        self.y            = 0
        self.z            = 0
        self.angle        = 0
        self.colour_name  = colour_name
        self.colour_range = colour_range

    def copy(self):
        copy = Joint(self.colour_name, self.colour_range)
        copy.x     = self.x
        copy.y     = self.y
        copy.z     = self.z
        copy.angle = self.angle

        return copy

class vision_2:
    def __init__(self):
        # set up ros nodes/pubs/subs etc.
        rospy.init_node("vision_2", anonymous=True)
        self.image_1_sub      = rospy.Subscriber('/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub      = rospy.Subscriber('/camera2/robot/image_raw', Image, self.callback_2)
        self.joints_est_2_pub = rospy.Publisher("joints_est_2", Float64MultiArray, queue_size=10)
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
        self.obstruct_thres = 1000

        # declare joints and their (range of) colours (for opencv thresholding)
        self.green  = Joint('green',  [(0, 100, 0),   (0, 255, 0)])
        self.yel1   = Joint('yellow', [(0, 100, 100), (0, 255, 255)])
        self.yel2   = Joint('yellow', [(0, 100, 100), (0, 255, 255)])
        self.blue   = Joint('blue',   [(100, 0, 0),   (255, 0, 0) ])
        self.red    = Joint('red',    [(0, 0, 100),   (0, 0, 255)])
        self.joints = [self.green, self.yel1, self.yel2, self.blue, self.red]

        # averages of the states of joints over the copies of joints
        self.avg_green  = self.green.copy()
        self.avg_yel1   = self.yel1.copy()
        self.avg_yel2   = self.yel2.copy()
        self.avg_blue   = self.blue.copy()
        self.avg_red    = self.red.copy()
        self.avg_joints = [self.avg_green, self.avg_yel1, self.avg_yel2, self.avg_blue, self.avg_red]
        # number of saved measurement states to use in average
        self.avg_window_size = 5

        # declare links between joints
        #   no reason to mess with link 2 (0m link) or link 1 (always vertical)
        self.link_3 = Link(self.yel2,  self.blue)
        self.link_4 = Link(self.blue,  self.red)

        # keep old copies of joint positions to estimate when blobs are obstructed
        joints_copy = [j.copy() for j in self.joints]
        self.saved_state_window_size = 50 # includes latest measurement not in prev_joints
        self.prev_joints = [joints_copy] * (self.saved_state_window_size - 1)

    # handle images seen from the camera facing the yz-plane
    def callback_1(self, data):
        self.detect_centres(data, 1)
        self.update_angles()
        self.publish_angles()

        self.debug() # TODO: remove

    # handle images seen from the camera facing the xz-plane
    def callback_2(self, data):
        self.detect_centres(data, 2)
        self.update_angles()
        self.publish_angles()

        self.debug() # TODO: remove

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
        copies = [j.copy() for j in self.joints]
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
        self.prev_joints.pop()
        self.prev_joints.insert(0, copies)

        # update averages of joints
        for i in range(self.no_joints):
            total_x     = self.joints[i].x
            total_y     = self.joints[i].y
            total_z     = self.joints[i].z
            total_angle = self.joints[i].angle
            for joints in self.prev_joints[0:self.avg_window_size]:
                total_x     += joints[i].x
                total_y     += joints[i].y
                total_z     += joints[i].z
                total_angle += joints[i].angle
            self.avg_joints[i].x     = total_x     / self.avg_window_size
            self.avg_joints[i].y     = total_y     / self.avg_window_size
            self.avg_joints[i].z     = total_z     / self.avg_window_size
            self.avg_joints[i].angle = total_angle / self.avg_window_size

    def publish_angles(self):
        joints_est = Float64MultiArray()
        joints_est.data = [self.green.angle, self.yel1.angle, self.yel2.angle, self.blue.angle]
        self.joints_est_2_pub.publish(joints_est)

    def update_angles(self):
        # calculate angles via trigonometry and linear algebra,
        #   using link that is visible
        # TODO: refactor magic number
        visibility_threshold = 20**2

        # try out possible angle combinations
        green_ang  = 0
        yel2_ang   = 0
        blue_ang   = 0

        # only consider orientation of links
        link_3 = self.link_3.as_normalized()
        link_4 = self.link_4.as_normalized()
        [link_3_x, link_3_y, link_3_z] = link_3
        [link_4_x, link_4_y, link_4_z] = link_4

        yel2_ang_1 = min(np.pi/2, np.arccos(min(1, link_3_z)))
        yel2_ang_2 = -yel2_ang_1
        yel2_ang   = closer_to(self.yel2.angle, yel2_ang_1, yel2_ang_2)
        if near_zero(yel2_ang):
            # cos(yel2) is near 1 (far from 0), so:
            blue_ang_1 = min(np.pi/2, np.arccos(min(1, link_4_z / link_3_z)))
            blue_ang_2 = -blue_ang_1
            blue_ang   = closer_to(self.blue.angle, blue_ang_1, blue_ang_2)

            # sin(yel2) is about 0, so approximate:
            sign = -1 if blue_ang < 0 else 1
            green_ang = np.arctan2(sign*link_4_y, sign*link_4_x)

            blue_cos = np.cos(blue_ang)
            if not near_zero(blue_cos):
                yel2_sin = (np.sin(green_ang)*link_4_x - np.cos(green_ang)*link_4_y) / blue_cos
                yel2_ang = np.arcsin(yel2_sin)
            elif near_zero(green_ang):
                yel2_ang = np.arcsin(-link_3_y / np.cos(green_ang))
            else:
                yel2_ang = np.arcsin(link_3_x / np.sin(green_ang))

        else:
            sign = -1 if yel2_ang < 0 else 1
            green_ang = np.arctan2(sign*self.link_3.get_x(), -sign*self.link_3.get_y())

            blue_sin = link_4_x*np.cos(green_ang) + link_4_y*np.sin(green_ang)
            blue_ang = np.arcsin(blue_sin)

            # TODO: correct when about to flip to wrong angles

        self.green.angle = green_ang
        self.yel2.angle  = yel2_ang
        self.blue.angle  = blue_ang

    def debug(self):
        #print(self.link_3.as_normalized())
        return


def main(args):
    _ = vision_2()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)