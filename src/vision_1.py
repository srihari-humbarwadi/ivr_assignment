#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - I' in the IVR assignment pdf
import os
import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64


NEARNESS_CONSTANT = 0.4


def clip(minimum, maximum, val):
    return max(minimum, min(maximum, val))


def closer_to(val, x, y):
    """ Return the argument (x or y) closer to val"""
    if abs(x - val) <= abs(y - val):
        return x
    else:
        return y


def near(val1, val2):
    return abs(val1 - val2) < NEARNESS_CONSTANT
# TODO: make near_zero use near once no worry of conflicts


def near_zero(val):
    return near(val, 0)

def approximate_with_data(ys):
    """ Approximate using Langrange interpolation """
    def approximator(x):
        total_points = len(ys)
        total = 0
        for i in range(total_points):
            total += ys[i] * \
                np.product([x - j for j in range(total_points) if j != i]) /  \
                np.product([i - j for j in range(total_points) if j != i])
        return total
    return approximator


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


class vision_1:
    def __init__(self):
        # set up ros nodes/pubs/subs etc.
        rospy.init_node("vision_1", anonymous=True)
        self.image_1_sub = rospy.Subscriber(
            '/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub = rospy.Subscriber(
            '/camera2/robot/image_raw', Image, self.callback_2)

        self.joint_2_pub = rospy.Publisher(
            'joint_angle_2', Float64, queue_size=20)
        self.joint_3_pub = rospy.Publisher(
            'joint_angle_3', Float64, queue_size=20)
        self.joint_4_pub = rospy.Publisher(
            'joint_angle_4', Float64, queue_size=20)

        self.joints_est_1_pub = rospy.Publisher(
            "joints_est_1", Float64MultiArray, queue_size=10)
        self.bridge = CvBridge()

        # TODO: optimize all (hyper-)parameters
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
        self.joints = [self.green, self.yel1, self.yel2,
                       self.blue, self.red]  # index 1-off from pdf

        # keep old copies of joint positions to estimate when blobs are obstructed
        joints_copy = [j.copy() for j in self.joints]
        # includes latest measurement not in prev_joints
        self.saved_state_window_size = 50
        self.prev_joints = [joints_copy] * (self.saved_state_window_size - 1)

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

        self.avg_link_3 = Link(self.avg_yel2, self.avg_blue)
        self.avg_link_4 = Link(self.avg_blue, self.avg_red)

        # keep a count of updates
        self._updates = {'camera_1': 0, 'camera_2': 0}

    # handle images seen from the camera facing the yz-plane
    def callback_1(self, data):
        self.detect_centres(data, 1)
        self.update_angles()
        self.publish_angles()

    # handle images seen from the camera facing the xz-plane
    def callback_2(self, data):
        self.detect_centres(data, 2)
        self.update_angles()
        self.publish_angles()

    def detect_centres(self, data, camera, dump_frames=True, output_dir='frames'):
        # read image
        raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        normalized_image = normalize_image(raw_image)
        cv_image = mask_background(normalized_image)

        if dump_frames:
            frame_count = self._updates['camera_{}'.format(camera)]

            if not os.path.exists(os.path.join(output_dir, str(camera))):
                os.makedirs(os.path.join(output_dir, str(camera)))

            frame_path = os.path.join(output_dir, str(
                camera), 'frame_{}.png'.format(frame_count))
            cv2.imwrite(frame_path, cv_image)

        self._updates['camera_{}'.format(camera)] += 1

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
                self.joints[i].obstructed = True
                continue
            self.joints[i].obstructed = False
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
        joint_2 = Float64()
        joint_2.data = self.yel1.angle

        joint_3 = Float64()
        joint_3.data = self.yel2.angle

        joint_4 = Float64()
        joint_4.data = self.blue.angle

        self.joint_2_pub.publish(joint_2)
        self.joint_3_pub.publish(joint_3)
        self.joint_4_pub.publish(joint_4)

    def update_angles(self):
        # calculate angles via trigonometry and linear algebra
        link_3_normal = self.link_3.as_normalized()
        link_4_normal = self.link_4.as_normalized()
        [link_3_x, link_3_y, link_3_z] = link_3_normal
        [link_4_x, link_4_y, link_4_z] = link_4_normal

        # update joint 3 angle
        self.yel2.angle = np.arcsin(-link_3_y)

        # update joint 2's angle
        if not near(np.pi/2, abs(self.yel2.angle)):
            self.yel1.angle = np.arctan2(link_3_x, link_3_z)
        elif not near_zero(self.blue.angle):
            prod_cos = np.cos(self.yel2.angle)*np.cos(self.blue.angle)
            blue_sin = np.sin(self.blue.angle)
            yel1_sin = (prod_cos*link_4_x - blue_sin*link_4_z)
            yel1_cos = (blue_sin*link_4_x + prod_cos*link_4_z)
            yel1_angle = np.arctan2(yel1_sin, yel1_cos)
            self.yel1.angle = np.clip(yel1_angle, -np.pi/2, np.pi/2)

        # update joint 4's angle
        blue_sin = np.cos(self.yel1.angle)*link_4_x - \
            np.sin(self.yel1.angle)*link_4_z
        self.blue.angle = np.arcsin(np.clip(blue_sin, -1, 1))


def main(args):
    _ = vision_1()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
