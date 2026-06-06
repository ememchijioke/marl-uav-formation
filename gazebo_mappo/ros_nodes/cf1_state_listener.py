#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class CF1StateListener(Node):
    def __init__(self):
        super().__init__("cf1_state_listener")

        self.subscription = self.create_subscription(
            Odometry,
            "/model/cf1/odometry",
            self.odom_callback,
            10,
        )

        self.message_count = 0
        self.get_logger().info("Listening to /model/cf1/odometry")

    def odom_callback(self, msg: Odometry):
        self.message_count += 1

        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear

        if self.message_count % 20 == 0:
            self.get_logger().info(
                f"cf1 position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} | "
                f"velocity: vx={vel.x:.3f}, vy={vel.y:.3f}, vz={vel.z:.3f}"
            )


def main():
    rclpy.init()
    node = CF1StateListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("State listener stopped by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()