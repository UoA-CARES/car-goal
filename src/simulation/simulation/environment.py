import rclpy
import os
from rclpy.node import Node
import subprocess
import time 
import numpy as np
import random
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Vector3
from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldReset, WorldControl
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Odometry
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
import math


class Environment(Node):

    def __init__(self, max_steps=10, reward_range=2):
        super().__init__('environment')
        """
        Arguments:
            max_steps: the max number of steps before truncating episode
            reward_range: how close the car must be to the goal in order to recieve large reward (100)
        """
       
        self.robot_description_pkg = get_package_share_directory('robot_description')

        # Parameters
       
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps

        self.step_counter = 0
        

        # Observation Fields

        self.goal_position = np.random.uniform(0, 10, (2,))

        self.obs = [0, 0, 0, 0, 0, 0, 0, 0]
        self.last_obs = [0, 0, 0, 0, 0, 0, 0, 0]
        
        self.obs_future = Future() 

        # Publishers & Subscribers

        self.cmd_vel_publisher = self.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
        
        self.odom_subscriber = self.create_subscription(Odometry, '/model/vehicle_blue/odometry', self.odom_callback, 10)
        self.pose_subscriber = self.create_subscription(Pose, '/world/ackermann_steering/model/vehicle_blue/pose', self.pose_callback, 10)


        # Simulation Reset Service

        self.reset_client = self.create_client(ControlWorld, 'world/ackermann_steering/control') 

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        
        self.reset_request = ControlWorld.Request()
        
        self.reset_future = Future()


        
        

    def spawn_entity(self, sdf_path, model_name, x=0, y=0, z=1):
        
        # ros_gz_bridge doesn't support this service yet, so must run gazebo service in subprocess

        x = str(x)
        y = str(y)
        z = str(z)

        subprocess.Popen(
                ['gz service -s /world/ackermann_steering/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req \'pose:{position: {x: ' + x + ', y: ' + y + ', z: ' + z + '}}, sdf_filename: \"' + sdf_path + '\", name: \"' + model_name + '\"\''],
               shell=True
               )

    def delete_entity(self, name):

        # ros_gz_bridge doesn't support this service yet, so must run gazebo service in subprocess
        
        # Modelled after command:
        #
        #   gz service -s world/ackermann_steering/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 1000 --req 'type: 2, name: "vehicle_blue"'

        subprocess.run(
                ['gz service -s world/ackermann_steering/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 1000 --req \'' + 'type: 2, name: \"' + name + '\"\''],
                shell=True
                )
    
    def generate_goal(self):
        # x_pos = self.obs[0]
       #  y_pos = self.obs[1]

        self.goal_position = [random.uniform(-10, 10), random.uniform(-10, 10)]

        self.spawn_entity(f"{self.robot_description_pkg}/goal.sdf", 'goal', self.goal_position[0], self.goal_position[1])


    def reset(self):
        # self.delete_entity('goal')
        self.set_action(0, 0)

        self.send_reset_request()

        time.sleep(0.5)

        self.generate_goal()
        
        self.step_counter = 0

        self.last_obs = [0, 0, 0, 0, 0, 0, 0, 0]

        return self.obs, self.get_info()

    
    def step(self, action):

        self.step_counter += 1
        self.obs_future = Future()
        
        lin_vel, ang_vel = action
        self.set_action(lin_vel, ang_vel)

        self.last_obs = self.obs

        time.sleep(0.5)
        
        rclpy.spin_until_future_complete(self, self.obs_future) 
        
        reward = self.compute_reward()

        return self.obs, reward, self.is_terminated(), self.step_counter > self.MAX_STEPS, self.get_info()


    def compute_reward(self):

        current_distance = math.dist(self.goal_position, self.obs[:2])

        old_distance = math.dist(self.goal_position, self.last_obs[:2])


        delta_distance = old_distance - current_distance

        reward = -0.5

        if current_distance < self.REWARD_RANGE:
            reward += 100

        reward += delta_distance


        return reward



    def is_terminated(self):
        # Returns whether the distance between the car and the goal is within the reward range threshold
        return math.dist(self.goal_position, self.obs[:2]) < self.REWARD_RANGE


 
    def set_action(self, linear_vel, angular_vel):
        linear = Vector3()
        angular = Vector3()

        linear.x = float(linear_vel) 
        linear.y = 0.0
        linear.z = 0.0

        angular.x = 0.0
        angular.y = 0.0
        angular.z = float(angular_vel)

        twist = Twist()

        twist.linear = linear
        twist.angular = angular

        self.cmd_vel_publisher.publish(twist)



    def send_reset_request(self):
        self.reset_request.world_control = WorldControl()

        self.reset_request.world_control.reset = WorldReset()

        self.reset_request.world_control.reset.all= True

        
        self.reset_future = self.reset_client.call_async(self.reset_request)
        rclpy.spin_until_future_complete(self, self.reset_future)

        return self.reset_future.result()


    def odom_callback(self, msg):
        twist = msg.twist.twist

        self.obs_future.velocity = [twist.linear.x, twist.angular.z]
        
        self.evaluate_future()
        
        
    def pose_callback(self, msg):
        pos = msg.position 
        ang = msg.orientation

        self.obs_future.position = [pos.x, pos.y, ang.z, ang.w]
        
        self.evaluate_future()

    def evaluate_future(self):
        
        if self.obs_future.position and self.obs_future.velocity: 

            self.obs = self.obs_future.position + self.goal_position + self.obs_future.velocity
            
            self.obs_future.set_result(self.obs)

    def get_info(self):
        return {'distance': math.dist(self.obs[:2], self.goal_position)}


class Future(rclpy.task.Future):
    def __init__(self):
        super().__init__()
        self.velocity = None
        self.position = None
