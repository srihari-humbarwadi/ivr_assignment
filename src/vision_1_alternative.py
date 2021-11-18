#!/usr/bin/env python3

# estimate the positions of joints, as described in 'Joint state estimation - I' in the IVR assignment pdf
import os
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg    import Float64MultiArray
from cv_bridge       import CvBridge

from copy import deepcopy

_Z_BIAS = 0.37
_PIXEL_TO_METER = 0.039
_GREEN_YELLOW_LINK = 4.0
_YELLOW_BLUE_LINK = 3.2
_BLUE_RED_LINK = 2.8


def get_missing_coordinate(coordinates, magnitude):
    a, b = coordinates
    x = np.sqrt(magnitude**2 - (a**2 + b**2))
    return x

def normalize_image(image):
    image = image / 255
    b, g, r = np.split(image, 3, axis=-1)
    scale = (b + g + r) + 1e-6
    return np.uint8(255 * image / scale)


def get_rotation_matrix(theta, axis):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    if axis == 'x':
        matrix = np.array([
            [1,          0,           0,    0],
            [0,    cos_theta, -sin_theta,   0],
            [0,    sin_theta, cos_theta,    0],
            [0,          0,           0,    1]])

    elif axis == 'y':
        matrix = np.array([
            [cos_theta,  0, sin_theta,      0],
            [0,          1,           0,    0],
            [-sin_theta, 0, cos_theta,      0],
            [0,          0,           0,    1]])

    elif axis == 'z':
        matrix = np.array([
            [cos_theta,  -sin_theta,  0,    0],
            [sin_theta, cos_theta,    0,    0],
            [0,          0,           1,    0],
            [0,          0,           0,    1]])

    else:
        raise ValueError('Invalid rotational axis')
    return matrix


class SimpleMovingAverageContainer:
    def __init__(self, initial_value, window_size):
        if initial_value is None:
            raise ValueError('`initial_value` cannot be `None`')

        self._data = [initial_value] * window_size
        self._window_size = window_size
        self._num_seen = 0

    def set(self, x):
        self._data[self._num_seen % self._window_size] = x
        self._num_seen += 1

    def over_write_last_value(self, x):
        self._data[(self._num_seen - 1) % self._window_size] = x

    @property
    def sma_value(self):
        if self._num_seen < self._window_size:
            return self.latest_value
        return sum(self._data) / self._window_size

    @property
    def latest_value(self):
        return self._data[(self._num_seen - 1) % self._window_size]


class Joint:
    _COLOR_RANGES = {
        'red':    [(0, 0, 100),   (50, 50, 255)],
        'green':  [(0, 100, 0),   (50, 255, 50)],
        'blue':   [(100, 0, 0),   (255, 50, 50)],
        'yellow': [(0, 100, 100), (50, 255, 255)],
    }

    def __init__(self, color, rotation_axis, rotation_limit):
        self._color = color
        self._x = SimpleMovingAverageContainer(0, 5)
        self._y = SimpleMovingAverageContainer(0, 5)
        self._z = SimpleMovingAverageContainer(0, 5)
        self._angle = SimpleMovingAverageContainer(0, 5)
        self._rotation_axis = rotation_axis
        self._rotation_limit = rotation_limit
        self._check_flips = False
        self._obstructed = False
        self._is_fixed = False

    def update_coordinates(self, image, camera, kernel=None, iterations=3):
        blob_mask = cv2.inRange(image, *Joint._COLOR_RANGES[self._color])

        if kernel is None:
            kernel = np.ones((5, 5), np.uint8)
        blob_mask = cv2.dilate(blob_mask, kernel, iterations=iterations)

        M = cv2.moments(blob_mask)
        area = M['m00']

        if area < 10000:
            self._obstructed = True
            return

        horizontal = M['m10'] / area
        vertical = M['m01'] / area

        self.update_coordinate('_z', vertical)
        if camera == 1:
            self.update_coordinate('_y', horizontal)
        else:
            self.update_coordinate('_x', horizontal)

    def update_angle(self, angle):
        if self._check_flips:
            print('checking if flipped for:', self._color)
            if np.abs(self._angle.latest_value - angle) > 1.22 and np.abs(angle) > 0.2:
                print('correcting flip', np.abs(self._angle.latest_value - angle))
                angle = np.sign(self._angle.latest_value) * np.abs(angle)

        self._angle.set(np.clip(
            angle, -1.0 * self._rotation_limit,
            self._rotation_limit))

    def update_coordinate(self, coordinate, value):
        if self._is_fixed:
            return
        value *= _PIXEL_TO_METER
        attr = getattr(self, coordinate)
        if np.allclose(value, 0) or self._obstructed:
            value = attr.sma_value
        attr.set(value)

    def __sub__(self, other_joint):
        return self.center - other_joint.center

    @property
    def color(self):
        return self._color

    @property
    def center(self):
        return np.array([self.x, self.y, self.z])

    @property
    def angle(self):
        return self._angle.latest_value

    @property
    def x(self):
        return self._x.latest_value

    @property
    def y(self):
        return self._y.latest_value

    @property
    def z(self):
        return self._z.latest_value + _Z_BIAS

    @property
    def rotation_axis(self):
        return self._rotation_axis

    def __repr__(self):
        msg = """color: {}\nx: {}\ny: {}\nz: {}\nangle: {}\nrotational_axis: {}""".format(
            self._color, self._x, self._y, self._z, self._angle, self._rotation_axis)
        return msg


def reverse_joint_rotation(previous_joint, current_joint, invert_sign=False):
    previous_angle = previous_joint.angle
    if invert_sign:
        previous_angle *= -1.0

    rotation_matrix = get_rotation_matrix(
        previous_angle, previous_joint.rotation_axis)
    tx, ty, tz = current_joint.center - previous_joint.center
    t_p = [tx, ty, tz, 1]

    r_tx, r_ty, r_tz, _ = np.matmul(rotation_matrix, t_p)
    r_t_p = np.array([r_tx, r_ty, r_tz])

    p = r_t_p + previous_joint.center
    return p

class RobotArm:
    def __init__(self, joints):
        self._joints = joints
        self._processed = {
            'camera_1': False,
            'camera_2': False,
        }

    def update_joint_coordinates(self, image, camera):
        for joint in self._joints:
            joint.update_coordinates(image, camera)
        self._processed['camera_{}'.format(camera)] = True

    def update_joint_angles(self, camera):
        if not (self._processed['camera_1'] and self._processed['camera_2']):
            return

        #joint names  :   1      2      3    4    5
        #axis         :   z      y      x    y    (end effector)
        #joint colors : green yellow yellow blue red
        #indices      :   0      1      2    3    4

        # update joints which have y axis as their axis of rotation
        # joint2 (yellow)
        horizontal = self._joints[1].x - self._joints[3].x
        vertical = np.clip(self._joints[1].z - self._joints[3].z, 0, _YELLOW_BLUE_LINK)
        j2_sma_angle = self._joints[1]._angle.sma_value
        j2_angle = np.arctan2(-horizontal, vertical)
        self._joints[1].update_angle(j2_angle)

        x, y, z = reverse_joint_rotation(self._joints[1], self._joints[3])
        horizontal = self._joints[2].y - y
        vertical = np.clip(self._joints[2].z - z, 0, _YELLOW_BLUE_LINK)
        j3_angle = np.arctan2(horizontal, vertical)
        self._joints[2].update_angle(j3_angle)

        if np.abs(np.abs(j3_angle) - np.pi / 2) < 0.05 or np.abs(vertical) < 0.05:
            self._joints[1]._angle.over_write_last_value(j2_sma_angle)

        blue_x, blue_y, blue_z = reverse_joint_rotation(self._joints[1], self._joints[3])
        blue_join_copy = deepcopy(self._joints[3])
        blue_join_copy._x.set(blue_x)
        blue_join_copy._y.set(blue_y)
        blue_join_copy._z.set(blue_z)
        blue_xx, blue_yy, blue_zz = reverse_joint_rotation(self._joints[2], blue_join_copy, invert_sign=False)


        red_x, red_y, red_z = reverse_joint_rotation(self._joints[1], self._joints[4])
        red_join_copy = deepcopy(self._joints[3])
        red_join_copy._x.set(red_x)
        red_join_copy._y.set(red_y)
        red_join_copy._z.set(red_z)
        red_xx, red_yy, red_zz = reverse_joint_rotation(self._joints[2], red_join_copy, invert_sign=False)

        horizontal = blue_xx - red_xx
        vertical = blue_zz - red_zz
        j4_sma_angle = self._joints[3]._angle.sma_value
        j4_angle = np.arctan2(-horizontal, vertical)
        self._joints[3].update_angle(j4_angle)

        if np.abs(np.abs(j3_angle) - np.pi / 2) < 0.05 or np.abs(vertical) < 0.05:
            self._joints[3]._angle.over_write_last_value(j4_sma_angle)


        self._processed['camera_1'] = False
        self._processed['camera_2'] = False
        for joint in self._joints:
            joint._obstructed = False

    @property
    def joints(self):
        return self._joints

    def __repr__(self):
        return str(self._joints)

class vision_1:
    def __init__(self):
        # set up ros nodes/pubs/subs etc.
        rospy.init_node("vision_1", anonymous=True)
        self.image_1_sub      = rospy.Subscriber('/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub      = rospy.Subscriber('/camera2/robot/image_raw', Image, self.callback_2)
        self.joints_est_1_pub = rospy.Publisher("joints_est_1", Float64MultiArray, queue_size=30)
        self.x_pub = rospy.Publisher("x_pub", Float64MultiArray, queue_size=30)
        self.y_pub = rospy.Publisher("y_pub", Float64MultiArray, queue_size=30)
        self.z_pub = rospy.Publisher("z_pub", Float64MultiArray, queue_size=30)

        self.bridge = CvBridge()

        # keep a count of updates
        self._updates = {'camera_1': 0, 'camera_2': 0}


        colors = ['green', 'yellow', 'yellow', 'blue', 'red']
        rotational_axis = ['z', 'y', 'x', 'y', None]
        abs_max_rotation = [np.pi, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        joints = [
            Joint(color, axis, roration_limit)
            for color, axis, roration_limit in zip(colors, rotational_axis, abs_max_rotation)]
        joints[3]._check_flips = True
        print(joints[3])
        self.arm = RobotArm(joints)
        self._published = False

    # handle images seen from the camera facing the yz-plane
    def callback_1(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imwrite('camera_1.png', cv_image)
        self.arm.update_joint_coordinates(cv_image, 1)
        self.arm.update_joint_angles(1)
        self.publish()

    # handle images seen from the camera facing the xz-plane
    def callback_2(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imwrite('camera_2.png', cv_image)
        self.arm.update_joint_coordinates(cv_image, 2)
        self.arm.update_joint_angles(2)
        self.publish()

    def publish(self):
        joints_est = Float64MultiArray()
        joints_est.data = [j.angle for j in self.arm.joints]
        self.joints_est_1_pub.publish(joints_est)

        x_est = Float64MultiArray()
        y_est = Float64MultiArray()
        z_est = Float64MultiArray()

        x_est.data = [j.x for j in self.arm.joints]
        y_est.data = [j.y for j in self.arm.joints]
        z_est.data = [j.z for j in self.arm.joints]
        self.x_pub.publish(x_est)
        self.y_pub.publish(y_est)
        self.z_pub.publish(z_est)


def main(args):
    _ = vision_1()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
