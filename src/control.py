# calculates forward kinematics and uses inverse kinematics to move robot towards desired trajectory

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.data import Image
from std_msgs.data    import Float64MultiArray, Float64
from cv_bridge       import CvBridge

LINK_1_LENGTH = 4
LINK_3_LENGTH = 3.2
LINK_4_LENGTH = 2.8

# short-hand
L1 = LINK_1_LENGTH
L3 = LINK_3_LENGTH
L4 = LINK_4_LENGTH



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
def get_jacobian(j1, j3, j4):
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


class Control:
    def __init__(self):
        # subscribe to relevant topics
        self.joint_1_sub = rospy.Subscriber('joint_angle_1', Float64, self._read_joint1_angle)
        self.joint_3_sub = rospy.Subscriber('joint_angle_3', Float64, self._read_joint3_angle)
        self.joint_4_sub = rospy.Subscriber('joint_angle_4', Float64, self._read_joint4_angle)
        self.target_sub = rospy.Subscriber('/target_control/target_pos', Float64MultiArray, self.open_control)

        # prepare publishers for joint angles
        self.joint_1_pub = rospy.Publisher(
            '/robot/joint1_position_controller/command', Float64, queue_size=20)
        self.joint_3_pub = rospy.Publisher(
            '/robot/joint3_position_controller/command', Float64, queue_size=20)
        self.joint_4_pub = rospy.Publisher(
            '/robot/joint4_position_controller/command', Float64, queue_size=20)

        self._detected_q = {
            'joint1': 0,
            'joint2': 0,
            'joint4': 0,
        }
        self._read_status = {
            'joint1': False,
            'joint2': False,
            'joint4': False,
            'detected_end_effector': False,
        }
        self._detected_pos = 0
        self._target_pos = 0
        self._previous_time = np.array([rospy.get_time()], dtype='float64')

    def _read_joint1_angle(self, data):
        self._detected_q['joint1'] = data
        self._read_status['joint1'] = True

    def _read_joint3_angle(self, data):
        self._detected_q['joint3'] = data.data
        self._read_status['joint3'] = True

    def _read_joint4_angle(self, data):
        self._detected_q['joint4'] = data.data
        self._read_status['joint4'] = True

    def _read_detected_end_effector_position(self, data):
        self._detected_pos = data.data
        self._read_status['end_effector_position'] = True

    def _reset_read_status(self):
        for k, _ in self._read_status.items():
            self._read_status[k] = False

    def open_control(self, data):

        if not all(list(self._read_status.values())):
            return

        joint_angels = [self._detected_q['join1'],
                        self._detected_q['join3'],
                        self._detected_q['join4']]
        
        q = np.array([*joint_angels])
        J = get_jacobian(*joint_angels)
        J_pinv = np.linalg.pinv(J)

        self._target_pos = data
        curent_time = rospy.get_time()
        dt = curent_time - self._previous_time
        self._previous_time = curent_time

        self.error = (self._detected_pos - self._target_pos) / dt

        q_new = q + (dt * np.dot(J_pinv, self.error.transpose()))
        self._reset_read_status()

        return q_new


def main(args):
  _ = Control()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
