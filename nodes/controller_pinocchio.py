#!/usr/bin/env python
import rospy

import numpy as np
from math import cos, sin

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

import pinocchio as pin
from urdf_parser_py.urdf import URDF

from dynamic_reconfigure.server import Server as DynServer
from ros_rrbot_lecture.cfg import ControllerConfig


class RRBot_Controller:
    def __init__(self, urdf_param_name, ee_frame) -> None:
        """
        urdf_param_name is the name on the ROS parameter server of the URDF to be used for the controller
        (e.g. will be loaded by some launch file)
        """
        # Read the urdf from the parameter server
        urdf_xml = rospy.get_param(urdf_param_name)

        # Init Pinocchio model, used to compute kynematic and dynamic quantities
        self.model = pin.buildModelFromXML(urdf_xml)
        self.data = self.model.createData()

        # Print informations about the model
        rospy.loginfo(
            "Controller : Robot model initialized with name: %s" % self.model.name
        )
        rospy.loginfo(
            "Number of joints: %d (NOTE: the first joint is just the one connecting the robot to the world"
            % self.model.njoints
        )
        rospy.loginfo("Number of links: %d" % self.model.nbodies)
        rospy.loginfo("Dimensions of robot state: %d" % self.model.nq)

        # Select the ID of the frame, from those listed in the URDF file,
        # that is going to act as the End-Effector of the robot
        if self.model.existFrame(ee_frame):
            self.ee_frame_id = self.model.getFrameId(ee_frame)
        else:
            rospy.loginfo("Controller: selected ee_frame does not exist.")
            raise ValueError
        # order of the joints, used to map from the topic /joint_state to self.q properly
        # (i.e. using directly the name of the joints, instead of their indexes, which is not always consistent)
        # NOTE: exclude first joint, as is the one connecting the model to the world
        self.joint_order = list(self.model.names[1:])
        rospy.loginfo(f"Controller joint order: {self.joint_order}")

        # Initialize joint state and target
        # NOTE: two joints are fixed
        self.q = np.zeros(self.model.nq)
        self.x_des = np.zeros(2)

        # Publishers
        ## Joint actuation commands
        self.joint_cmd_pub = rospy.Publisher(
            "/rrbot/joints_position_controller/command", Float64MultiArray, queue_size=5
        )
        # init prototype message, not needed but it helps to avoid doing it everytime we publish
        self.joint_cmd_msg = Float64MultiArray()
        self.joint_cmd_msg.layout.dim.append(
            MultiArrayDimension(label="positions", size=self.q.shape[0], stride=1)
        )
        ## End-Effector Pose
        self.ee_state_pub = rospy.Publisher("~ee_state", Point, queue_size=5)
        self.ee_state_msg = Point()

        # Subscribers
        # Joint state
        self.joint_state_sub = rospy.Subscriber(
            "/rrbot/joint_states", JointState, self.update_joint_state, queue_size=5
        )
        # Desired End-Effector position
        self.x_des_sub = rospy.Subscriber(
            "~x_des", Point, self.update_x_des, queue_size=5
        )

        # Dynamic Reconfigure Server to update parameters online
        self.dynamic_reconfigure_server = DynServer(ControllerConfig, self.dyn_callback)

    def f(self, q):
        """Forward kinematics. Should return a 2D array with the x-z position of the end-effector"""
        """Update end-effector state based on the stored value of the joint state and the Jacobian."""
        # Compute forward kinematics on the model
        # this function update the self.data structure, relative to our pinocchio model
        pin.forwardKinematics(self.model, self.data, q)
        # Update the pose of each frame according to the new forward kinematics
        pin.updateFramePlacements(self.model, self.data)

        # Extract end-effector position
        x = self.data.oMf[self.ee_frame_id].translation
        # # Extract end-effector rotation matrix
        # R = self.data.oMf[self.ee_frame_id].rotation

        # return only the components for (X,Z)
        
        return x[[0,2]]

    def J(self, q):
        """Compute the Jacobian for the given joint state"""
        # Create matrix
        Jee = np.zeros((6, self.model.nq))
        # Use pinocchio to compute the Jacobian
        Jee = pin.computeFrameJacobian(
            self.model,
            self.data,
            # joint state
            q,
            # id of the reference frame for which we are computing the Jacobian
            self.ee_frame_id,
            # reference frame in which we want the Jacobian to be expressed
            # in this case, the base of the robot
            # NOTE: should be the same in which we express x_des
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # return only the rows relative to (x,z)
        return Jee[[0, 2]]

    def update_joint_state(self, joint_state):
        # Required because the order of the joints as published in the topic /joint_state
        # does not necessarily match the order in the URDF; so, we have to match them by name.
        # Create a dictionary with key = joint name , value = [position, velocity, effort]
        state_dict = dict(
            zip(
                joint_state.name,
                zip(joint_state.position, joint_state.velocity, joint_state.effort),
            )
        )
        # take the position of the first two free joints in the URDF
        self.q = np.array((
            state_dict[self.joint_order[0]][0],
            state_dict[self.joint_order[1]][0],
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
    
    def dyn_callback(self, config, level):
        self.K = config['IK_gain']
        self.rho = config['pseudoinv_damping']
        return config


# To publish a new x_des
# $ rostopic pub /rrbot/rrbot_controller/x_des geometry_msgs/Point '{x: 0.9, y: 0.0, z: 2.5}'


if __name__ == "__main__":
    try:
        NODE_NAME = "rrbot_controller"
        # Start node
        rospy.init_node(NODE_NAME)

        # Retrieve rate from ROS server parameters, set to default if no specific value has been set
        HZ = rospy.get_param("~hz", default=100.0)
        dt0 = 1.0 / HZ
        rate = rospy.Rate(HZ)

        ctrl = RRBot_Controller('~pinocchio_robot_description', 'ee_frame')

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
            Jdpinv = J.T.dot(np.linalg.inv(J.dot(J.T) + (ctrl.rho**2) * np.eye(2)))

            # qd_des = Jpinv.dot(K*ex)
            qd_des = Jdpinv.dot(ctrl.K * ex)

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
