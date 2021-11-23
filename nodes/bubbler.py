import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

if __name__ == '__main__':
    try:

        NODE_NAME = 'motor_bubbler'
        # Start node
        rospy.init_node(NODE_NAME)

        # Retrieve rate from ROS server parameters, set to default if no specific value has been set
        HZ = rospy.get_param('~hz', default=100.0)
        dt0 = 1.0/HZ
        rate = rospy.Rate(HZ)

        # Publisher
        joint_cmd_pub = rospy.Publisher(
            '/rrbot/joints_position_controller/command', Float64MultiArray, queue_size=5)
        # init prototype message; it's not needed, but it helps to avoid doing it everytime we publish
        joint_cmd_msg = Float64MultiArray()
        # size=2 since there a 2 joints
        joint_cmd_msg.layout.dim.append(
            MultiArrayDimension(label='positions', size=2, stride=1))

        # generate an array of shape (2,) with random number between 0 and 1
        # np.random.rand(2)

        while not rospy.is_shutdown():
            joint_cmd_msg.data = list(np.random.rand(2))
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    rospy.spin()
