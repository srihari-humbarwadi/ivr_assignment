#!/usr/bin/env python3

# calculates forward kinematics and uses inverse kinematics to move robot towards desired trajectory

import sys
from time import time

import cv2
import numpy as np
import rospy
import tensorflow as tf
from std_msgs.msg import Float64, Float64MultiArray

LINK_1_LENGTH = 4
LINK_3_LENGTH = 3.2
LINK_4_LENGTH = 2.8

# short-hand
L1 = LINK_1_LENGTH
L3 = LINK_3_LENGTH
L4 = LINK_4_LENGTH


# def get_rotation_matrix(theta, axis):
#     sin_theta = tf.math.sin(theta)
#     cos_theta = tf.math.cos(theta)

#     if axis == 'x':
#         matrix = tf.convert_to_tensor([
#             [1,          0,           0,    0],
#             [0,    cos_theta, -sin_theta,   0],
#             [0,    sin_theta, cos_theta,    0],
#             [0,          0,           0,    1]])

#     elif axis == 'y':
#         matrix = tf.convert_to_tensor([
#             [cos_theta,  0, sin_theta,      0],
#             [0,          1,           0,    0],
#             [-sin_theta, 0, cos_theta,      0],
#             [0,          0,           0,    1]])

#     elif axis == 'z':
#         matrix = tf.convert_to_tensor([
#             [cos_theta,  -sin_theta,  0,    0],
#             [sin_theta, cos_theta,    0,    0],
#             [0,          0,           1,    0],
#             [0,          0,           0,    1]])

#     else:
#         raise ValueError('Invalid rotational axis')
#     return tf.cast(matrix, dtype=tf.float32)


# matrix multiplications are costly in tf avoid this

# def FK(j1, j3, j4):
#     j1 = tf.clip_by_value(j1, -np.pi, np.pi)
#     j3 = tf.clip_by_value(j3, -np.pi / 2, np.pi / 2)
#     j4 = tf.clip_by_value(j4, -np.pi / 2, np.pi / 2)

#     a1_r = get_rotation_matrix(j1, 'z')
#     a1_t = np.eye(4)
#     a1_t[2, 3] = LINK_1_LENGTH
#     a1_t = tf.constant(a1_t, dtype=tf.float32)
#     a1 = tf.matmul(a1_r, a1_t)

#     a3_r = get_rotation_matrix(j3, 'x')
#     a3_t = np.eye(4)
#     a3_t[2, 3] = LINK_3_LENGTH
#     a3_t = tf.constant(a3_t, dtype=tf.float32)
#     a3 = tf.matmul(a3_r, a3_t)

#     a4_r = get_rotation_matrix(j4, 'y')
#     a4_t = np.eye(4)
#     a4_t[2, 3] = LINK_4_LENGTH
#     a4_t = tf.constant(a4_t, dtype=tf.float32)
#     a4 = tf.matmul(a4_r, a4_t)

#     a = tf.matmul(a1, tf.matmul(a3, a4))
#     xyz = a[:3, 3]
#     return xyz

def s(a):
    return tf.math.sin(a)


def c(a):
    return tf.math.cos(a)

# calculate the position of the end-effector from the angles of joints 1, 3, 4
def FK(j1, j3, j4):
    x = c(j1)*s(j4)*L4 + s(j1)*s(j3)*c(j4)*L4 + s(j1)*s(j3)*L3
    y = s(j1)*s(j4)*L4 - c(j1)*s(j3)*c(j4)*L4 - c(j1)*s(j3)*L3
    z = c(j3)*c(j4)*L4 + c(j3)*L3 + L1
    return tf.convert_to_tensor([x, y, z])

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


optimizer = tf.optimizers.SGD(learning_rate=0.01)

@tf.function(jit_compile=True)
def solve(target_pos, j1, j3, j4):
    q = [j1, j3, j4]

    # compiled tf graph requires us to initialize all outputs
    error = tf.constant(0.0)

    # run gradient descent for 500 steps 
    # (in most of the cases we find a solution within 100 steps)
    for step in tf.range(1, 501):
        with tf.GradientTape() as tape:
            # compute current position of the 
            # end effector using forward kinematics
            current_pos = FK(j1, j3, j4)

            # compute mean squared error of the current
            # and target positions of the end effector
            error = tf.reduce_mean((target_pos - current_pos) ** 2)

            # Use automatic differentiation to compute 
            # gradients d(error)/dq. Since we are using
            # forward kinematics to compute the current
            # position, we can show that the gradients 
            # can be expressed in terms of the Jacobian matrix.
            gradients = tape.gradient(error, q)

        # exit optimization when error drop to a very low value
        mean_abs_diff = tf.reduce_mean(tf.abs(current_pos - target_pos))
        if mean_abs_diff < 0.005:
            break

        # apply the gradient descent update rule
        # theta_t+1 = theta_t - alpha * d(error)/d(thetha)
        # where alphais the learning rate
        optimizer.apply_gradients(zip(gradients, q))
    return {'steps': step, 'error': mean_abs_diff, 'q': q}

def get_tf_variable(name, clip):
    value = np.random.normal(0, 0.005 * clip)
    return tf.Variable(value, name=name, dtype=tf.float32)

class Control:
    def __init__(self):
        rospy.init_node('control', anonymous=True)

        # subscribe to relevant topics
        self.target_sub = rospy.Subscriber('/target_pos', Float64MultiArray, self.open_control)

        # prepare publishers for joint angles
        self.joint_1_pub = rospy.Publisher(
            '/robot/joint1_position_controller/command', Float64, queue_size=20)
        self.joint_3_pub = rospy.Publisher(
            '/robot/joint3_position_controller/command', Float64, queue_size=20)
        self.joint_4_pub = rospy.Publisher(
            '/robot/joint4_position_controller/command', Float64, queue_size=20)

        self._initialize_variables()

    def _initialize_variables(self):
        self._q = {
            'joint1': get_tf_variable('joint1', np.pi / 4),
            'joint3': get_tf_variable('joint3', np.pi / 2),
            'joint4': get_tf_variable('joint4', np.pi / 2),
        }

    def open_control(self, data):
        # we would either use the current estimated joint angles
        # or reinitialize the join angles to the same initial 
        # seed for each new target. We empiracally found that, 
        # neither of those work the best. Randomly initializing
        # the joint angles around zero, worked the best.
        self._initialize_variables()

        j1 = self._q['joint1']
        j3 = self._q['joint3']
        j4 = self._q['joint4']
        target_pos = tf.constant(data.data, dtype=tf.float32)

        s = time()
        results = solve(target_pos, j1, j3, j4)
        e = time()
        print('Reached error: {} in {} steps for target: {} in {:.3f} secs'.format(
            np.round(results['error'].numpy(), 3), results['steps'].numpy(), target_pos.numpy(), e - s))

        # publish new angles to reach the target
        joint_1_command = Float64()
        joint_3_command = Float64()
        joint_4_command = Float64()

        joint_1_command.data = results['q'][0]
        joint_3_command.data = results['q'][1]
        joint_4_command.data = results['q'][2]

        self.joint_1_pub.publish(joint_1_command)
        self.joint_3_pub.publish(joint_3_command)
        self.joint_4_pub.publish(joint_4_command)


def main(args):
    _ = Control()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
