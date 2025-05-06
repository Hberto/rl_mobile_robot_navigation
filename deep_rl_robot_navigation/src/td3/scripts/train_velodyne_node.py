#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading

import math
import random

import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Quaternion as RosQuaternion

GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.35
TIME_DELTA = 0.2

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

last_odom = None
environment_dim = 20
velodyne_data = np.ones(environment_dim) * 10

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

class td3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        # ToDo: change this with every run
        self.writer = SummaryWriter(log_dir="/home/ubuntu/Desktop/RL_Projekt/rl_mobile_robot_navigation/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/runs")
        # os.path.dirname(os.path.realpath(__file__)) + "/runs"
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=16,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.3,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        env.get_logger().info(f"writing new results for a tensorboard")
        env.get_logger().info(f"loss, Av.Q, Max.Q, iterations : {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )

class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        self.environment_dim = 20
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.success_rate = 0.0  
        self.successful_episodes = 0 
        self.total_episodes = 0       

        self.upper = 1.5
        self.lower = -1.5


        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

    # Perform an action and read a new state
    def step(self, action):
        global velodyne_data
        target = False
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(velodyne_data)
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            #env.get_logger().info("GOAL is reached!")
            env.get_logger().info(f"GOAL REACHED! Target was: ({self.goal_x:.2f}, {self.goal_y:.2f})")
            target = True
            done = True

        # my new approach for better training in less episodes
        if target:
            self.successful_episodes += 1

        self.success_rate = self.successful_episodes / max(self.total_episodes, 1)
        env.get_logger().info(f"Success_Rate: {self.success_rate}")


        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        distance_to_g = self.odom_x
        reward= self.get_reward(target, collision, action, min_laser, distance)
        return state, reward, done, target

    def reset(self):
        # small maze
        #MAZE_MIN_X = 0.0
        #MAZE_MAX_X = 3.9
        #MAZE_MIN_Y = -2.8
        #MAZE_MAX_Y = 1.6
        # large maze
        MAZE_MIN_X = 0
        MAZE_MAX_X = 4.4
        MAZE_MIN_Y = -3.0
        MAZE_MAX_Y = 1.9
        self.total_episodes += 1

        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")


        x = 0
        y = 0
        position_ok = False
        max_attempts = 50
        attempt = 0
        while not position_ok and attempt < max_attempts:
            #x = np.random.uniform(-4.5, 4.5)
            #y = np.random.uniform(-4.5, 4.5)
            x = np.random.uniform(MAZE_MIN_X + 0.5, MAZE_MAX_X - 0.5)  # 0.5m margin from walls
            y = np.random.uniform(MAZE_MIN_Y + 0.5, MAZE_MAX_Y - 0.5)
            position_ok = check_pos(x, y)
            attempt += 1
        if not position_ok:
            x, y = (MAZE_MAX_X - MAZE_MIN_X)/2, (MAZE_MAX_Y - MAZE_MIN_Y)/2  # Center
            self.get_logger().warn(f"⚠️ Using center position ({x:.1f}, {y:.1f})")
        
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.get_logger().info(f"New robot position: ({x}, {y})")
        #self.set_state.publish(object_state)

        self.set_state.publish(object_state)
        time.sleep(1)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state


    def change_goal(self):
        # Small maze
        #MAZE_MIN_X = 0.0
        #MAZE_MAX_X = 3.9
        #MAZE_MIN_Y = -2.8
        #MAZE_MAX_Y = 1.6
        # large maze
        MAZE_MIN_X = 0
        MAZE_MAX_X = 4.4
        MAZE_MIN_Y = -3.0
        MAZE_MAX_Y = 1.9
        # Updated to match SDF ground plane coverage
        #MAZE_MIN_X = -4.5  # Extended from -3.8
        #MAZE_MAX_X= 4.5   # Extended from 3.8
        #MAZE_MIN_Y = -3.0   # Adjusted from -2.9
        #MAZE_MAX_Y = 3.0    # Extended from 1.5

        #[INFO] [1741604917.518130227] [env]: GOAL with success_rate X: -0.5747097301337947, GOAL Y with success_rate: 0.5294583322601398



        # Dynamische Ziele basierend auf letzter Performance
        if self.success_rate <= 0.3:
                for _ in range(50):
                    self.goal_x = np.clip(self.odom_x + np.random.uniform(-0.1, 0.1), 
                            MAZE_MIN_X + 0.5, MAZE_MAX_X - 0.5)
                    self.goal_y = np.clip(self.odom_y + np.random.uniform(-0.1, 0.1),
                            MAZE_MIN_Y + 0.5, MAZE_MAX_Y - 0.5)
                    if check_pos(self.goal_x, self.goal_y):
                        env.get_logger().info(f"GOAL with success_rate X: {self.goal_x}, GOAL Y with success_rate: {self.goal_y}")
                        return

        else:
            # Feste herausfordernde Ziele
            #fixed_goals = [            
            #(2.4, -0.1),   # Central region (from log)
            #(1.7, -2.0),  # Central-lower region
            #(3.5, 1.0)    # Upper right region
            #]
            fixed_goals = [
            (1.43, 0.99),    # up central
            (3.67, -2.37),  # Lower-left corner
            (2.99, 0.31),   # Upper central
            (1.29, -2.34)    #down central
            ]
            self.goal_x, self.goal_y = random.choice(fixed_goals)
            env.get_logger().info(f"GOAL X: {self.goal_x}, GOAL Y: {self.goal_y}")

        # Boundary-Check mit Fallback
        #self.goal_x = np.clip(self.goal_x, MAZE_MIN_X + 0.5, MAZE_MAX_X - 0.5)
        #self.goal_y = np.clip(self.goal_y, MAZE_MIN_Y + 0.5, MAZE_MAX_Y - 0.5)

        if not check_pos(self.goal_x, self.goal_y):
            env.get_logger().warn("Invalid goal detected! Using safe fallback")
            self.goal_x, self.goal_y = 0.7, -0.7  # Central verified safe position


    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-3, 3) # -6,6 vorher
                y = np.random.uniform(-3, 3)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        #marker.header.frame_id = "odom"
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(self.goal_x)
        marker.pose.position.y = float(self.goal_y)
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = float(abs(action[0]))
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5.0
        marker2.pose.position.y = 0.0
        marker2.pose.position.z = 0.0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = float(abs(action[1]))
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0.0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            env.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser, distance):
        if target:
            env.get_logger().info("reward 100")
            return 100.0
        elif collision:
            env.get_logger().info("reward -100")
            return -100.0
        else:
            #r3 = lambda x: 1 - x if x < 1 else 0.0
            #return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

            # Keep strong forward movement reward
            linear_speed_reward = action[0] / 2  

            # Keep strong rotation penalty
            rotation_penalty = -abs(action[1]) / 2  

            # Keep strong obstacle avoidance penalty
            #obstacle_repulsion = - (1 - min_laser) / 2 
            obstacle_repulsion = - (1 - min_laser) * 10 

            # Distance-based reward: Higher reward when closer to the goal
            # with small: 10
            #goal_proximity_reward = 20.0 / (distance + 1e-3)  # Avoid division by zero
            goal_proximity_reward = 50.0 / (distance + 1e-3)

            # Total reward calculation
            reward = linear_speed_reward + rotation_penalty + obstacle_repulsion + goal_proximity_reward
            return reward

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

#def check_pos(x, y):
#    # Check if the random goal position is located on an obstacle and do not accept it if it is
#    goal_ok = True
#
#    # Maze-Grenzen (6x6 Meter, zentriert um den Ursprung)
#    maze_min_x = -3.0
#    maze_max_x = 3.0
#    maze_min_y = -3.0
#    maze_max_y = 3.0
#
#    # Definition der Wände, basierend auf den SDF-Daten:
#    walls = [
#        (2.36565, -0.539019, 5.29645, 0.15),  # Wall_16 (small_maze_2)
#        (0.068925, 2.57321, 4.52249, 0.15),    # Wall_18 (small_maze_2)
#        (2.26396, 0.017589, 5.29645, 0.15),     # Wall_19 (small_maze_2)
#        (1.95518, -3.22728, 4.54012, 0.15),     # Wall_20 (small_maze_2)
#        (0.44035, 1.29102, 1.0, 0.15),          # Wall_22 (small_maze_2)
#        (1.90836, 0.051842, 3.5, 0.15),         # Wall_24 (small_maze_3)
#        (-0.311357, -0.695592, 2.5, 0.15),       # Wall_27 (small_maze_2)
#        (0.999475, -0.769438, 2.5, 0.15),        # Wall_27 (small_maze_3)
#    ]
#    
#    #if x > -0.55 and 1.7 > y > -1.7:
#    #    goal_ok = True
#
#    if not (maze_min_x <= x <= maze_max_x and maze_min_y <= y <= maze_max_y):
#        goal_ok = False 
#
#    for wall_x, wall_y, width, thickness in walls:
#        if (wall_x - width/2 <= x <= wall_x + width/2) and \
#           (wall_y - thickness/2 <= y <= wall_y + thickness/2):
#            return False
#
#    return goal_ok
                    
def check_pos(x, y):
    """
    Checks if the given (x, y) position is valid by ensuring that:
      1. It is within the maze boundaries.
      2. It does not fall inside any of the obstacles defined by wall bounding boxes.
    Maze boundaries (manually determined):
      x: [0.0, 3.9]
      y: [-2.8, 1.6]
    Walls are defined as tuples: (wall_center_x, wall_center_y, width, thickness)
    where width is the size along the x-axis and thickness along the y-axis.
    """
    # Maze boundaries based on your measurements for small maze
    #maze_min_x = 0.0
    #maze_max_x = 3.9
    #maze_min_y = -2.8
    #maze_max_y = 1.6
    # Maze boundaries for larger maze
    maze_min_x = 0
    maze_max_x = 4.4
    maze_min_y = -3.0
    maze_max_y = 1.9

    # Updated to match SDF ground plane coverage
    #maze_min_x = -4.5  # Extended from -3.8
    #maze_max_x = 4.5   # Extended from 3.8
    #maze_min_y = -3.0   # Adjusted from -2.9
    #maze_max_y = 3.0    # Extended from 1.5

    if not (maze_min_x+0.3 <= x <= maze_max_x-0.3 and 
            maze_min_y+0.3 <= y <= maze_max_y-0.3):
        return False
    # Definition of walls (dead zones) based on your SDF data:
    #walls = [
    #    (2.36565, -0.539019, 5.29645, 0.15),  # Wall_16 (small_maze_2)
    #    (0.068925, 2.57321, 4.52249, 0.15),    # Wall_18 (small_maze_2)
    #    (2.26396, 0.017589, 5.29645, 0.15),     # Wall_19 (small_maze_2)
    #    (1.95518, -3.22728, 4.54012, 0.15),     # Wall_20 (small_maze_2)
    #    (0.44035, 1.29102, 1.0, 0.15),          # Wall_22 (small_maze_2)
    #    (1.90836, 0.051842, 3.5, 0.15),         # Wall_24 (small_maze_3)
    #    (-0.311357, -0.695592, 2.5, 0.15),       # Wall_27 (small_maze_2)
    #    (0.999475, -0.769438, 2.5, 0.15),        # Wall_27 (small_maze_3)
    #]
    # Static obstacles matching SDF structure
    walls = [
        # Vertical walls (x_center, y_center, width_x, thickness_y)
        (-3.4, 0.0, 0.2, 5.8), (3.4, 0.0, 0.2, 5.8),
        # Horizontal walls
        (0.0, 1.1, 6.8, 0.2), (0.0, -2.4, 6.8, 0.2),
        # Central barriers
        (1.7, -1.0, 2.2, 0.2), (-1.7, 0.5, 0.2, 2.0)
    ]
    
    # Cylinder obstacles from SDF (x, y, radius)
    cylinders = [
        (2.7, 1.1, 0.5), (-2.5, -1.8, 0.6)
    ]

    # Check if the (x, y) falls inside any wall's bounding box.
    # The bounding box is defined by [wall_center - (dimension/2), wall_center + (dimension/2)]
    for wx, wy, w, t in walls:
        if (wx-w/2 <= x <= wx+w/2) and (wy-t/2 <= y <= wy+t/2):
            return False
    for cx, cy, cr in cylinders:
        if math.hypot(x-cx, y-cy) < cr + 0.3:
            return False
    return True

#def evaluate(network, epoch, eval_episodes=10):
#    avg_reward = 0.0
#    col = 0
#    success_rate = 0
#    collision_rate = 0
#    avg_path_length = 0
#    avg_time = 0
#    distance_history = []
#
#
#    for _ in range(eval_episodes):
#        env.get_logger().info(f"evaluating episode {_}")
#        count = 0
#        state = env.reset()
#        done = False
#        while not done and count < 501:
#            action = network.get_action(np.array(state))
#            env.get_logger().info(f"action : {action}")
#            a_in = [(action[0] + 1) / 2, action[1]]
#            state, reward, done, _ = env.step(a_in)
#            avg_reward += reward
#            count += 1
#            if reward < -90:
#                col += 1
#    avg_reward /= eval_episodes
#    avg_col = col / eval_episodes
#    env.get_logger().info("..............................................")
#    env.get_logger().info(
#        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
#        % (eval_episodes, epoch, avg_reward, avg_col)
#    )
#    env.get_logger().info("..............................................")
#    return avg_reward

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    success_rate = 0
    collision_rate = 0
    avg_path_length = 0
    avg_time = 0
    distance_history = []
    all_actions = []

    for ep in range(eval_episodes):
        env.get_logger().info(f"Evaluating episode {ep + 1}/{eval_episodes}")
        state = env.reset()
        done = False
        episode_steps = 0
        episode_distance = []
        episode_reward = 0

        while not done and episode_steps < 501:
            action = network.get_action(np.array(state))
            env.get_logger().info(f"Action taken: {np.round(action, 2)}")
            
            # Store actions for statistics
            all_actions.append(action)
            
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            
            # Track metrics
            current_distance = np.linalg.norm([env.odom_x - env.goal_x, env.odom_y - env.goal_y])
            episode_distance.append(current_distance)
            episode_reward += reward
            avg_reward += reward

            if target:
                success_rate += 1
                avg_time += episode_steps * TIME_DELTA
            if reward <= -90:
                collision_rate += 1

            episode_steps += 1
            state = next_state

        # Episode statistics
        avg_path_length += np.sum(episode_distance)
        distance_history.extend(episode_distance)
        env.get_logger().info(
            f"Episode {ep + 1} | "
            f"Reward: {episode_reward:.1f} | "
            f"Steps: {episode_steps} | "
            f"Final Distance: {current_distance:.2f}m"
        )

    # Calculate aggregates
    avg_reward /= eval_episodes
    success_rate /= eval_episodes
    collision_rate /= eval_episodes
    avg_path_length /= eval_episodes
    avg_time = avg_time / max(success_rate * eval_episodes, 1)  # Avoid division by zero

    # Convert actions to numpy array
    all_actions = np.array(all_actions)

    # Log to TensorBoard
    network.writer.add_scalar("Evaluation/Average Reward", avg_reward, epoch)
    network.writer.add_scalar("Evaluation/Success Rate", success_rate, epoch)
    network.writer.add_scalar("Evaluation/Collision Rate", collision_rate, epoch)
    network.writer.add_scalar("Evaluation/Avg Path Length", avg_path_length, epoch)
    network.writer.add_scalar("Evaluation/Avg Time to Goal", avg_time, epoch)
    network.writer.add_histogram("Evaluation/Distance Distribution", np.array(distance_history), epoch)
    network.writer.add_histogram("Actions/Linear", all_actions[:, 0], epoch)
    network.writer.add_histogram("Actions/Angular", all_actions[:, 1], epoch)

    # Console logging
    env.get_logger().info("\n" + "="*60)
    env.get_logger().info(f"Evaluation Epoch {epoch} Results:")
    env.get_logger().info(f"- Average Reward: {avg_reward:.2f}")
    env.get_logger().info(f"- Success Rate: {success_rate*100:.1f}%")
    env.get_logger().info(f"- Collision Rate: {collision_rate*100:.1f}%")
    env.get_logger().info(f"- Avg Path Length: {avg_path_length:.2f}m")
    env.get_logger().info(f"- Avg Time to Goal: {avg_time:.2f}s")
    env.get_logger().info("="*60 + "\n")

    return avg_reward
if __name__ == '__main__':

    rclpy.init(args=None)

    # run1: small maze

    seed = 0  # Random seed number
    eval_freq = 5e3  # After how many steps to perform the evaluation
    #max_ep = 300  # maximum number of steps per episode
    max_ep = 500
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over
    )
    expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 32  # Size of the mini-batch
    #discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
    #tau = 0.005  # Soft target update variable (should be close to 0)
    discount = 0.99  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.02 
    #policy_noise = 0.2  # Added noise for exploration
    policy_noise = 0.5
    noise_clip = 0.5  # Maximum clamping values of the noise
    #policy_freq = 2  # Frequency of Actor network updates
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "td3_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False  # Weather to load a stored model
    random_near_obstacle = True  # To take random actions near obstacles or not

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Create the training environment
    environment_dim = 20
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    # Create the network
    network = td3(state_dim, action_dim, max_action)
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            print("Will load existing model.")
            network.load(file_name, "./pytorch_models")
        except:
            print("Could not load the stored model parameters, initializing training with random parameters")

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []

    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Velodyne_subscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            if timestep < max_timesteps:
                # On termination of episode
                if done:
                    env.get_logger().info(f"Done. timestep : {timestep}")
                    if timestep != 0:
                        env.get_logger().info(f"train")
                        network.train(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                        )

                    if timesteps_since_eval >= eval_freq:
                        env.get_logger().info("Validating")
                        timesteps_since_eval %= eval_freq
                        evaluations.append(
                            evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
                        )
                        # ToDO: change this with every run 
                        network.save(file_name, directory="/home/ubuntu/Desktop/RL_Projekt/rl_mobile_robot_navigation/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models/run13")
                        np.save("/home/ubuntu/Desktop/RL_Projekt/rl_mobile_robot_navigation/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/results/%s" % (file_name), evaluations)
                        epoch += 1
                        env.get_logger().info(f"Epoch: {epoch}")

                    state = env.reset()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                # add some exploration noise
                if expl_noise > expl_min:
                    expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                # If the robot is facing an obstacle, randomly force it to take a consistent random action.
                # This is done to increase exploration in situations near obstacles.
                # Training can also be performed without it
                if random_near_obstacle:
                    if (
                        #np.random.uniform(0, 1) > 0.85
                        np.random.uniform(0, 1) > 0.7
                        #and min(state[4:-8]) < 0.6
                        and min(state[4:-8]) < 0.4
                        and count_rand_actions < 1
                    ):
                        #count_rand_actions = np.random.randint(8, 15)
                        #random_action = np.random.uniform(-1, 1, 2)
                        # Kürzere Aktionssequenz für präzisere Exploration
                        count_rand_actions = np.random.randint(5, 10)  # Statt 8-15
                        random_action = np.random.uniform(-0.5, 0.5, 2)  # Kleinere Aktionen

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action = random_action
                        action[0] = -1
                        #action[0] = -0.5

                # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                a_in = [(action[0] + 1) / 2, action[1]]
                next_state, reward, done, target = env.step(a_in)
                done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
                done = 1 if episode_timesteps + 1 == max_ep else int(done)
                episode_reward += reward

                # Save the tuple in replay buffer
                replay_buffer.add(state, action, reward, done_bool, next_state)

                # Update the counters
                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
