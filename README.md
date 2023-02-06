# Car Goal Reinforcement Learning Environment
A reinforcement learning environment, in which the agent learns to navigate and drive a car (ackermann steering) to a goal position.

<img src="https://drive.google.com/uc?export=view&id=1160h1EbougIVE9nxo7znePh-PEOIfchJ" />

## Installation Instructions
Clone the repository
```
git clone --recurse-submodules https://github.com/retinfai/car-goal.git
```

Install [Ros2 Humble](https://docs.ros.org/en/humble/Installation.html)

Install [Gazebo Garden](https://gazebosim.org/docs/garden/install)

Install [Cares Reinforcement Learning Package](https://github.com/UoA-CARES/cares_reinforcement_learning)

The following instructions are taken from the [ros_gz README](https://github.com/gazebosim/ros_gz)

1. Set Gazebo Version:
```
export GZ_VERSION=garden
```

2. Install Dependencies
```
# cd into workspace
cd ~/car-goal
rosdep install -r --from-paths src -i -y --rosdistro humble
```

3. Build Workspace
```
# Source ROS distro's setup.bash
source /opt/ros/humble/setup.bash

# Build and install into workspace
cd ~/car-goal
colcon build
```

Done! You should be able to run the environment :)

## Running this Environment

To run the environment, run the following

```
# cd into the workspace
cd ~/car-goal

# Build
colcon build

# Source
. install/setup.bash

# Run
ros2 launch simulation simulation.launch.py
```

## Testing your own Reinforcement Learning Algorithms/Networks
If you would like to try out different RL algorithms/networks, edit `src/simulation/simulation/training.py`

**Note** while editing this file, that the environment is a ros node, and thus can only be created after ros initialisation `rclpy.init()`
