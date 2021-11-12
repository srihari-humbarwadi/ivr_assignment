#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - I' in the IVR assignment pdf

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray
from cv_bridge       import CvBridge


# clip angles so they're between -pi/2, pi/2
def clip(angle):
    return max(-np.pi/2, min(np.pi/2, angle))

# replace 0 with arbitrarily small value
def zero_guard(val):
    # replace 0 by this small amount to prevent division by zero
    zero_guard_val = 0.001
    return zero_guard_val if val == 0 else val

# approximate using Langrange interpolation
def approximate_with_data(ys):
    def approximator(x):
        total_points = len(ys)
        total        = 0
        for i in range(total_points):
            total += ys[i] * \
                np.product([x - j for j in range(total_points) if j != i]) /  \
                np.product([i - j for j in range(total_points) if j != i])
        return total
    return approximator

class Joint:
    # colour range for opencv binary thresholding
    def __init__(self, colour_name, colour_range):
        self.x            = 0
        self.y            = 0
        self.z            = 0
        self.angle        = 0
        self.obstructed   = False
        self.colour_name  = colour_name
        self.colour_range = colour_range

        # when not None, used to approximate angles when blobs are obstructed or too close together
        self.approximator = None
        # argument to be passed to the approximator
        self.approx_arg  = 0

    def copy(self):
        copy = Joint(self.colour_name, self.colour_range)
        copy.x     = self.x
        copy.y     = self.y
        copy.z     = self.z
        copy.angle = self.angle

        return copy

class vision_1:
    def __init__(self):
        # set up ros nodes/pubs/subs etc.
        rospy.init_node("vision_1", anonymous=True)
        self.image_1_sub      = rospy.Subscriber('/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub      = rospy.Subscriber('/camera2/robot/image_raw', Image, self.callback_2)
        self.joints_est_1_pub = rospy.Publisher("joints_est_1", Float64MultiArray, queue_size=10)
        self.bridge = CvBridge()

        # TODO: optimize all (hyper-)parameters
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
        self.green = Joint('green',  [(0, 100, 0),   (10, 255, 10) ])
        self.yel1  = Joint('yellow', [(0, 100, 100), (10, 255, 255)])
        self.yel2  = Joint('yellow', [(0, 100, 100), (10, 255, 255)])
        self.blue  = Joint('blue',   [(100, 0, 0),   (255, 10, 10) ])
        self.red   = Joint('red',    [(0, 0, 100),   (10, 10, 255) ])
        self.joints = [self.green, self.yel1, self.yel2, self.blue, self.red] #index 1-off from pdf

        # keep old copies of joint positions to estimate when blobs are obstructed
        joints_copy = [j.copy() for j in self.joints]
        self.saved_state_window_size = 50 # includes latest measurement not in prev_joints
        self.prev_joints = [joints_copy] * (self.saved_state_window_size - 1)

        # averages of the states of joints over the copies of joints
        self.avg_green  = self.green.copy()
        self.avg_yel1   = self.yel1.copy()
        self.avg_yel2   = self.yel2.copy()
        self.avg_blue   = self.blue.copy()
        self.avg_red    = self.red.copy()
        self.avg_joints = [self.avg_green, self.avg_yel1, self.avg_yel2, self.avg_blue, self.avg_red]
        # number of saved measurement states to use in average
        self.avg_window_size = 5

        # threshold for when to switch from arctan to approximating angles using previous values
        self.approx_angle_threshold   = 20**2
        # number of angles to use in approximation (should not be greater than saved_state_window_size)
        self.approx_angle_window_size = 50

    # handle images seen from the camera facing the yz-plane
    def callback_1(self, data):
        self.detect_centres(data, 1)
        # calculate angles via trigonometry and linear algebra
        # accuracy can be improved, but problems with oscillations due to obstruction
        self.yel2.angle = np.arctan2(
            self.avg_yel2.y - self.avg_blue.y,
            (self.avg_yel2.z - self.avg_blue.z) #/ zero_guard(np.cos(self.yel1.angle))
        )
        self.publish_angles()

    # handle images seen from the camera facing the xz-plane
    def callback_2(self, data):
        self.detect_centres(data, 2)
        
        # calculate angles via trigonometry and linear algebra
        # arctan(ky/kx) = arctan(y/x) so division by cos of other angle is unneeded
        x_diff = self.avg_yel1.x - self.avg_blue.x
        z_diff = self.avg_yel1.z - self.avg_blue.z

        # use approximation by interpolation when blobs are close (arctan(~0/~0) oscillate wildly)
        # not used for now, because it works badly
        if False and x_diff**2 + z_diff**2 < self.approx_angle_threshold:
            old_angles = [joints[1].angle \
                for joints in self.prev_joints[0:self.approx_angle_window_size]]
            if self.yel1.approximator == None:
                self.yel1.approximator = approximate_with_data(old_angles)
                self.yel1.approx_arg  = self.approx_angle_window_size
            self.yel1.angle = clip(self.yel1.approximator(self.yel1.approx_arg))
            self.yel1.approx_arg += 1
            
        else:
            self.yel1.approximator = None
            self.yel1.approx_time  = 0
            #negated because y axis points away from screen
            self.yel1.angle = clip(np.arctan2(-x_diff, z_diff))

        # TODO: sign is still wrong at times
        # calculate cos and sin of blue joint from dot and cross proucts of links (as vectors)
        link_3 = np.array([self.avg_blue.x - self.avg_yel1.x, self.avg_blue.y - self.avg_yel1.y, self.avg_blue.z - self.avg_yel1.z])
        link_4 = np.array([self.avg_red.x - self.avg_blue.x, self.avg_red.y - self.avg_blue.y, self.avg_red.z - self.avg_blue.z])
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
        copies = [j.copy() for j in self.joints]
        for i in range(self.no_joints):
            area       = moments[i]['m00']
            if area < self.obstruct_thres:
                self.joints[i].obstructed = True
                continue
            self.joints[i].obstructed = False
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