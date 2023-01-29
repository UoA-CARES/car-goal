from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    # Getting Robot Description   

    robot_description_pkg = get_package_share_directory('robot_description')
    
    # Launching the Gazebo Simulation 

    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f"-r {robot_description_pkg}/world.sdf --render-engine ogre" # render-engine option to run on Linux ARM64
        }.items(),
    )
    
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            f'/world/ackermann_steering/control@ros_gz_interfaces/srv/ControlWorld',
            f'/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            f'/world/ackermann_steering/model/vehicle_blue/pose@geometry_msgs/msg/Pose@gz.msgs.Pose'
        ]

    )

    training_node = Node(
        package='simulation',
        executable='training',
        output='screen')



    return LaunchDescription([
        gz_sim,
        gz_bridge,
        training_node
    ])
