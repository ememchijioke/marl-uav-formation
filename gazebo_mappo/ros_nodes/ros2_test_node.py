#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class Ros2TestNode(Node):
    def __init__(self):
        super().__init__("ros2_test_node")
        self.get_logger().info("ROS 2 test node is running from the thesis repo.")


def main():
    rclpy.init()
    node = Ros2TestNode()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
