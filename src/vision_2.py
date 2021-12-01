#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - II' in the IVR assignment pdf

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge


PIXEL_TO_METER = 0.037

NEARNESS_CONSTANT = 0.3


def near(val1, val2):
    return abs(val1 - val2) < NEARNESS_CONSTANT


def near_zero(val):
    return near(val, 0)


def closer_to(val, x, y):
    """ Return the argument (x or y) closer to val"""
    if abs(x - val) <= abs(y - val):
        return x
    else:
        return y


def normalize_image(image):
    image = image / 255
    b, g, r = np.split(image, 3, axis=-1)
    scale = (b + g + r) + 1e-6
    return np.uint8(255 * image / scale)


def mask_background(image):
    foreground_mask = np.expand_dims(~np.logical_and(
        image[:, :, 0] == image[:, :, 1],
        image[:, :, 1] == image[:, :, 2]),
        axis=-1)
    return image * foreground_mask

class Link:
    """ Model the link as a vector from joint1 to joint2 """

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
    """ Abstraction for joints """
    _COLOR_RANGES = {
        'red':    [(0, 0, 100),   (50, 50, 255)],
        'green':  [(0, 100, 0),   (50, 255, 50)],
        'blue':   [(100, 0, 0),   (255, 50, 50)],
        'yellow': [(0, 100, 100), (50, 255, 255)],
    }

    def __init__(self, colour_name):
        self.x = 0
        self.y = 0
        self.z = 0
        self.angle = 0
        self.colour_name = colour_name
        self.colour_range = Joint._COLOR_RANGES[colour_name]

    def copy(self):
        copy = Joint(self.colour_name)
        copy.x = self.x
        copy.y = self.y
        copy.z = self.z
        copy.angle = self.angle

        return copy


class vision_2:
    def __init__(self):
        # set up ros nodes/pubs/subs etc.
        rospy.init_node("vision_2", anonymous=True)
        self.image_1_sub = rospy.Subscriber(
            '/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub = rospy.Subscriber(
            '/camera2/robot/image_raw', Image, self.callback_2)
        self.joints_est_2_pub = rospy.Publisher(
            "joints_est_2", Float64MultiArray, queue_size=10)

        self.joint_1_pub = rospy.Publisher(
            'joint_angle_1', Float64, queue_size=20)
        self.joint_3_pub = rospy.Publisher(
            'joint_angle_3', Float64, queue_size=20)
        self.joint_4_pub = rospy.Publisher(
            'joint_angle_4', Float64, queue_size=20)

        # TODO: maybe remove or comment out these topics before submission?
        self.end_effector_pos = rospy.Publisher(
            'end_effector_pos', Float64MultiArray, queue_size=20)

        self.bridge = CvBridge()

        # set up kernel and etc. for blob detection
        # self.kernel = np.array([ # for more control later
        #    [0, 0, 1, 0, 0],
        #    [0, 1, 1, 1, 0],
        #    [1, 1, 1, 1, 1],
        #    [0, 1, 1, 1, 0],
        #    [0, 0, 1, 0, 0]
        # ], np.uint8)
        self.kernel = np.full((5, 5), 1, np.uint8)
        self.no_iter = 3
        self.no_joints = 5  # "including end-effector", so off by 1
        # not off by one, does not include "0m link from ground"
        self.no_links = self.no_joints - 1

        # consider blob blocked (e.g. by link) if its area m00 is below this threshold
        self.obstruct_thres = 1000

        # declare joints and their (range of) colours (for opencv thresholding)
        self.green = Joint('green')
        self.yel1 = Joint('yellow')
        self.yel2 = Joint('yellow')
        self.blue = Joint('blue')
        self.red = Joint('red')
        self.joints = [self.green, self.yel1, self.yel2, self.blue, self.red]

        # averages of the states of joints over the copies of joints
        self.avg_green = self.green.copy()
        self.avg_yel1 = self.yel1.copy()
        self.avg_yel2 = self.yel2.copy()
        self.avg_blue = self.blue.copy()
        self.avg_red = self.red.copy()
        self.avg_joints = [self.avg_green, self.avg_yel1,
                           self.avg_yel2, self.avg_blue, self.avg_red]
        # number of saved measurement states to use in average
        self.avg_window_size = 5

        # declare links between joints
        #   no reason to mess with link 2 (0m link) or link 1 (always vertical)
        self.link_3 = Link(self.yel2,  self.blue)
        self.link_4 = Link(self.blue,  self.red)

        # keep old copies of joint positions to estimate when blobs are obstructed
        joints_copy = [j.copy() for j in self.joints]
        # includes latest measurement not in prev_joints
        self.saved_state_window_size = 50
        self.prev_joints = [joints_copy] * (self.saved_state_window_size - 1)

    # handle images seen from the camera facing the yz-plane
    def callback_1(self, data):
        self.detect_centres(data, 1)
        self.update_angles()
        self.publish_angles()

        self.debug()  # TODO: remove

    # handle images seen from the camera facing the xz-plane
    def callback_2(self, data):
        self.detect_centres(data, 2)
        self.update_angles()
        self.publish_angles()

        self.debug()  # TODO: remove

    def detect_centres(self, data, camera):
        # read image
        raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        normalized_image = normalize_image(raw_image)
        cv_image = mask_background(normalized_image)

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
            area = moments[i]['m00']
            if area < self.obstruct_thres:
                continue
            vertical = moments[i]['m01'] / area
            horizontal = moments[i]['m10'] / area

            self.joints[i].z = vertical
            if camera == 1:
                self.joints[i].y = horizontal
            else:
                self.joints[i].x = horizontal
        self.prev_joints.pop()
        self.prev_joints.insert(0, copies)

        # update averages of joints
        for i in range(self.no_joints):
            total_x = self.joints[i].x
            total_y = self.joints[i].y
            total_z = self.joints[i].z
            total_angle = self.joints[i].angle
            for joints in self.prev_joints[0:self.avg_window_size]:
                total_x += joints[i].x
                total_y += joints[i].y
                total_z += joints[i].z
                total_angle += joints[i].angle
            self.avg_joints[i].x = total_x / self.avg_window_size
            self.avg_joints[i].y = total_y / self.avg_window_size
            self.avg_joints[i].z = total_z / self.avg_window_size
            self.avg_joints[i].angle = total_angle / self.avg_window_size

    def publish_angles(self):
        joints_est = Float64MultiArray()
        joints_est.data = [self.green.angle,
                           self.yel1.angle, self.yel2.angle, self.blue.angle]

        joint_1 = Float64()
        joint_3 = Float64()
        joint_4 = Float64()

        joint_1.data = self.green.angle
        joint_3.data = self.yel2.angle
        joint_4.data = self.blue.angle

        self.joints_est_2_pub.publish(joints_est)
        self.joint_1_pub.publish(joint_1)
        self.joint_3_pub.publish(joint_3)
        self.joint_4_pub.publish(joint_4)

    def update_angles(self):
        # calculate angles via trigonometry and linear algebra,
        #   using link that is visible

        # try out possible angle combinations
        green_ang = self.green.angle
        yel2_ang = self.yel2.angle
        blue_ang = self.blue.angle

        # only consider orientation of links
        link_3 = self.link_3.as_normalized()
        link_4 = self.link_4.as_normalized()
        [link_3_x, link_3_y, link_3_z] = link_3
        [link_4_x, link_4_y, link_4_z] = link_4

        # update yellow angle
        if not near_zero(self.yel2.angle):
            # joints moving continuously as it is a not-small physical system
            #   pick one of two possible angle closer to previous one
            yel2_ang_1 = min(np.pi/2, np.arccos(min(1, link_3_z)))
            yel2_ang_2 = -yel2_ang_1
            yel2_ang = closer_to(yel2_ang, yel2_ang_1, yel2_ang_2)
        else:
            yel2_sin = np.sin(green_ang)*link_3_x - np.cos(green_ang)*link_3_y
            yel2_ang = np.arcsin(np.clip(yel2_sin, -1, 1))

        # update blue angle
        # if-branch introduces oscillations # TODO: delete?
        # if not near_zero(blue_ang) and not near(np.pi/2, abs(yel2_ang)) \
        #        and not near_zero(link_3_z):
        #    blue_cos   = np.clip(link_4_z/link_3_z, 0, 1)
        #    blue_ang_1 = np.arccos(blue_cos)
        #    blue_ang_2 = -blue_ang_1
        #    blue_ang   = closer_to(blue_ang, blue_ang_1, blue_ang_2)
        # else:
        blue_sin = link_4_x*np.cos(green_ang) + link_4_y*np.sin(green_ang)
        blue_ang = np.arcsin(blue_sin)

        # update green angle
        if not near_zero(yel2_ang):
            sign = -1 if yel2_ang < 0 else 1
            green_ang = np.arctan2(sign*link_3_x, -sign*link_3_y)
        elif not near_zero(blue_ang):
            blue_sin = np.sin(blue_ang)
            sin_cos_prod = np.sin(yel2_ang)*np.cos(blue_ang)

            green_sin = sin_cos_prod*link_4_x + blue_sin*link_4_y
            green_cos = blue_sin*link_4_x - sin_cos_prod*link_4_y
            green_ang = np.arctan2(green_sin, green_cos)

        # guard against green angle flipping +pi <-> -pi
        if abs(green_ang - self.green.angle) > np.pi*1.8:
            green_ang = self.green.angle

        # if green angle flips discontinuously (+pi <-> -pi)
        #   then we're in the wrong set of angles, so flip back
        if abs(green_ang - self.green.angle) > np.pi*1.7:
            if green_ang < 0:
                green_ang += np.pi
            else:
                green_ang -= np.pi

            yel2_ang = -yel2_ang
            blue_ang = -blue_ang

        self.green.angle = green_ang
        self.yel2.angle = yel2_ang
        self.blue.angle = blue_ang

    def debug(self):
        end_eff_pos = Float64MultiArray()
        end_eff_pos.data = [
            PIXEL_TO_METER * (self.red.x - self.green.x),
            PIXEL_TO_METER * (self.red.y - self.green.y),
            PIXEL_TO_METER * (self.green.z - self.red.z)
        ]
        self.end_effector_pos.publish(end_eff_pos)
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
