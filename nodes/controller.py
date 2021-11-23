import rospy

import numpy as np
from math import cos, sin

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

class RRBot_Controller:
    def __init__(self, l1, l2 , l3) -> None:
        """ 
        l1,l2,l3 are the lengths of the links of RRBot, starting from the base
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.q = np.zeros(2)

        self.x_des = np.zeros(2)

        # Publishers
        # Joint actuation
        self.joint_cmd_pub = rospy.Publisher('/rrbot/joints_position_controller/command', Float64MultiArray, queue_size=5)
        # init prototype message, not needed but helps to avoid doing it everytime we publish
        self.joint_cmd_msg = Float64MultiArray()
        self.joint_cmd_msg.layout.dim.append(MultiArrayDimension(label='positions', size=self.q.shape[0], stride=1))

        # Subscribers
        # Joint state
        self.joint_state_sub = rospy.Subscriber('/rrbot/joint_states', JointState, self.update_joint_state, queue_size=5)
        # Desired End-Effector position
        self.x_des_sub = rospy.Subscriber('/rrbot/x_des', Point, self.update_x_des, queue_size=5)
    
    def f(self, q):
        """ Forward kinematics. Should return a 2D array with the x-z position of the end-effector"""
        return np.array((
            self.l2 * sin(q[0]) + self.l3*sin(q[0]+q[1]),
            self.l1 + self.l2 * cos(q[0]) + self.l3*cos(q[0]+q[1])
        ))

    def J(self, q):
        """Compute the Jacobian for the given joint state"""
        return np.array(
            [[ self.l2*cos(q[0]) + self.l3*cos(q[0] + q[1]),  self.l3*cos(q[0] + q[1])],
            [-self.l2*sin(q[0]) - self.l3*sin(q[0] + q[1]), -self.l3*sin(q[0] + q[1])]])
    
    def update_joint_state(self, joint_state):
        self.q = np.array(joint_state.position)
    
    def update_x_des(self, point):
        self.x_des = np.array((point.x, point.z))
    
    def publish_joint_cmd(self, q_des):
        self.joint_cmd_msg.data = list(q_des)
        self.joint_cmd_pub.publish(self.joint_cmd_msg)


# To publish a new x_des
# $ rostopic pub /rrbot/x_des geometry_msgs/Point '{x: -0.4, y: 0.0, z: 0.7}'


if __name__ == '__main__':
    try:

        NODE_NAME = 'rrbot_controller'
        # Start node 
        rospy.init_node(NODE_NAME)

        # Retrieve rate from ROS server parameters, set to default if no specific value has been set
        HZ = rospy.get_param('~hz', default=100.0)
        dt0 = 1.0/HZ
        rate = rospy.Rate(HZ)

        ctrl = RRBot_Controller(2,1,1)

        # Gain for closed-loop Inverse Kinematics
        K = 2.0

        while not rospy.is_shutdown():
            # Forward kinematics: where is the end-effector?
            x = ctrl.f(ctrl.q)
            # Jacobian
            J = ctrl.J(ctrl.q)
            # Error
            ex = ctrl.x_des - x
            # Inverse Kinematics: what's the speed desired for the joints
            # Jinv = np.linalg.inv(J)
            Jpinv = np.linalg.pinv(J)
            rho = 1e0
            Jdpinv = J.T.dot(np.linalg.inv(J.dot(J.T) + (rho**2)*np.eye(2)))
            qd_des = Jdpinv.dot(K*ex)
            # Forward Euler: what's the position desired for the joints?
            q_des = ctrl.q + dt0 * qd_des

            # Publish command
            ctrl.publish_joint_cmd(q_des)
            # sleep until next iteration
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()