
# ROS 2 and Gazebo Setup Notes

##  System

- Ubuntu 22.04.5 LTS
- ROS 2 Humble
- Gazebo Sim 7.9.0
- `rclpy` works outside Conda
- ROS 2 test node runs successfully from this thesis repository

## Important Environment Rule

For ROS 2 and Gazebo work, do not use the Conda `uav-marl` environment.

Use:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
cd ~/Projects/marl-uav-formation
git checkout gazebo-migration