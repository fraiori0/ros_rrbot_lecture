#!/usr/bin/env python
import rospy

import numpy as np
from math import cos, sin

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point


class RRBot_Controller:
    def __init__(self, l1, l2, l3) -> None:
        """ 
        l1,l2,l3 are the lengths of the links of RRBot, starting from the base
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.q = np.zeros(2)

        self.x_des = np.zeros(2)

        # Publishers
        ## Joint actuation commands
        self.joint_cmd_pub = rospy.Publisher(
            '/rrbot/joints_position_controller/command', Float64MultiArray, queue_size=5)
        # init prototype message, not needed but it helps to avoid doing it everytime we publish
        self.joint_cmd_msg = Float64MultiArray()
        self.joint_cmd_msg.layout.dim.append(MultiArrayDimension(
            label='positions', size=self.q.shape[0], stride=1))
        ## End-Effector Pose
        self.ee_state_pub = rospy.Publisher(
            '~ee_state', Point, queue_size=5)
        self.ee_state_msg = Point()

        # Subscribers
        # Joint state
        self.joint_state_sub = rospy.Subscriber(
            '/rrbot/joint_states', JointState, self.update_joint_state, queue_size=5)
        # Desired End-Effector position
        self.x_des_sub = rospy.Subscriber(
            '~x_des', Point, self.update_x_des, queue_size=5)

    def f(self, q):
        """ Forward kinematics. Should return a 2D array with the x-z position of the end-effector"""
        return np.array((
            self.l2 * sin(q[0]) + self.l3*sin(q[0]+q[1]),
            self.l1 + self.l2 * cos(q[0]) + self.l3*cos(q[0]+q[1])
        ))

    def J(self, q):
        """Compute the Jacobian for the given joint state"""
        return np.array(
            [[self.l2*cos(q[0]) + self.l3*cos(q[0] + q[1]),  self.l3*cos(q[0] + q[1])],
             [-self.l2*sin(q[0]) - self.l3*sin(q[0] + q[1]), -self.l3*sin(q[0] + q[1])]])

    def update_joint_state(self, joint_state):
        # Required because the order of the joints as published in the joint_state
        # does not necessarily match the order in the URDF; so, we have to match them by name
        state_dict = dict(zip(joint_state.name, zip(
                joint_state.position, joint_state.velocity, joint_state.effort)))
        self.q = np.array((
            state_dict['joint1'][0],
            state_dict['joint2'][0]
        ))


    def update_x_des(self, point):
        self.x_des = np.array((point.x, point.z))

    def publish_joint_cmd(self, q_des):
        self.joint_cmd_msg.data = list(q_des)
        self.joint_cmd_pub.publish(self.joint_cmd_msg)
    
    def publish_ee_state(self):
        # compute forward kinematic
        x = self.f(self.q)
        # Fill message fields
        self.ee_state_msg.x = x[0]
        self.ee_state_msg.y = 0.0
        self.ee_state_msg.z = x[1]
        # Publish Message
        self.ee_state_pub.publish(self.ee_state_msg)


# To publish a new x_des
# $ rostopic pub /rrbot/x_des geometry_msgs/Point '{x: 0.9, y: 0.0, z: 2.5}'


if __name__ == '__main__':
    try:

        NODE_NAME = 'rrbot_controller'
        # Start node
        rospy.init_node(NODE_NAME)

        # Retrieve rate from ROS server parameters, set to default if no specific value has been set
        HZ = rospy.get_param('~hz', default=100.0)
        dt0 = 1.0/HZ
        rate = rospy.Rate(HZ)

        ctrl = RRBot_Controller(2, 1, 1)

        # Gain for closed-loop Inverse Kinematics
        K = 2.0

        while not rospy.is_shutdown():

            # Forward kinematics: where is the end-effector?
            x = ctrl.f(ctrl.q)
            # Jacobian
            J = ctrl.J(ctrl.q)
            # Error
            ex = ctrl.x_des - x

            # Closed-loop Inverse Kinematics
            # -- inverse
            # Jinv = np.linalg.inv(J)
            # -- pseudo-inverse
            # Jpinv = np.linalg.pinv(J)
            # -- damped pseudo-inverse
            rho = 1e0
            Jdpinv = J.T.dot(np.linalg.inv(J.dot(J.T) + (rho**2)*np.eye(2)))

            # qd_des = Jpinv.dot(K*ex)
            qd_des = Jdpinv.dot(K*ex)

            # Forward Euler: what's the position desired
            # (i.e. that we should command) for the joints?
            q_des = ctrl.q + dt0 * qd_des

            # Publish command
            ctrl.publish_joint_cmd(q_des)

            # Publish End-Effector's (X,Z) Position
            ctrl.publish_ee_state()

            # sleep until next iteration
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    rospy.spin()
