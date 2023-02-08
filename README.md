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

The following instructions are taken from the [ros_gz](https://github.com/gazebosim/ros_gz) docs

1. Set Gazebo Version:
```
export GZ_VERSION=garden
```

2. Install Dependencies
```
# cd into workspace
cd ~/car-goal
rosdep install -r --from-paths src -i -y --rosdistro humble
# Note: the keys for the latest gazebo packages – gz-transport12, gz_sim7 etc. – are not indexed by rosdep.
```

3. Build Workspace
```
# Source ROS distro's setup.bash. Hopefully this line is already in your .bashrc or equivalent.
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

# Environment Details
The following contains details about the environment, such as the observation and action spaces as well as constraints

## Interface
The Environment generally follows the Open AI gym interface; with methods such as `reset` and `step`. For more details on this, read the [gym](https://www.gymlibrary.dev/) documentaion.

## Observation
Observations in this environmnet is a **1D array** of *floats*; the *size* of the array is **8**. Look below:
|Index in Array| Observation      | Observation Type |
|----| ----------- | ----------- |
|0| car x position      | float       |
|1| car y position   | float    |
|2| car z orientation   | float        |
|3| car w orientation   | float        |
|4| goal x position | float |
|5| goal y position | float |
|6| car linear velocity | float |
|7| car angular velocity | float|

**Note:** Quaternion is used for orientation

## Action
Actions in this environment is a **1D array** of *floats*; the *size* of the array is 2.

|Index in Array | Action | Type | Min | Max |
|----|----|----|----|----|
|0 | car linear velocity | float | 0 | 3 |
|1 | car angular velocity | float | -1 | 1 |
