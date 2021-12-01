#!/usr/bin/env python3

import sys

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


_PIXEL_TO_METER = 0.039


class Joint:
    _COLOR_RANGES = {
        'red':    [(0, 0, 100),   (50, 50, 255)],
        'green':  [(0, 100, 0),   (50, 255, 50)],
    }

    def __init__(self, color):
        self._color = color
        self._x = 0
        self._y = 0
        self._z = [0, 0]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        z = sum(self._z) / 2
        self._z = [z, z]
        return z


    def update_coordinates(self, image, camera, kernel=None, iterations=3):
        blob_mask = cv2.inRange(image, *Joint._COLOR_RANGES[self._color])

        if kernel is None:
            kernel = np.ones((5, 5), np.uint8)
        blob_mask = cv2.dilate(blob_mask, kernel, iterations=iterations)

        M = cv2.moments(blob_mask)
        area = M['m00']

        if area < 500:
            return

        horizontal = _PIXEL_TO_METER *  M['m10'] / area
        vertical = _PIXEL_TO_METER *  M['m01'] / area

        self._z[camera - 1] = vertical
        if camera == 1:
            self._y = horizontal
        else:
            self._x = horizontal

class EndEffector:
    def __init__(self):
        rospy.init_node('end_effector', anonymous=True)

        self.bridge = CvBridge()

        self.publisher = rospy.Publisher(
            'end_effector_pos', Float64MultiArray, queue_size=20)
        self.image_1_sub = rospy.Subscriber(
            '/camera1/robot/image_raw', Image, self.callback_1)
        self.image_2_sub = rospy.Subscriber(
            '/camera2/robot/image_raw', Image, self.callback_2)

        self._green_joint = Joint('green')
        self._red_joint = Joint('red')

        
    def callback_1(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self._green_joint.update_coordinates(cv_image, 1)
        self._red_joint.update_coordinates(cv_image, 1)

        self.publish()
        
    def callback_2(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self._green_joint.update_coordinates(cv_image, 2)
        self._red_joint.update_coordinates(cv_image, 2)

        self.publish()

    def publish(self):
        position = Float64MultiArray()
        position.data = [
            self._red_joint.x - self._green_joint.x,
            self._red_joint.y - self._green_joint.y,
            self._green_joint.z - self._red_joint.z,
        ]
        self.publisher.publish(position)

def main(args):
    _ = EndEffector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
