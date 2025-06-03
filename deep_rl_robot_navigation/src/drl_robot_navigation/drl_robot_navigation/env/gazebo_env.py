
import time
import numpy as np

import rclpy
from rclpy.node import Node
import threading

import math
import random

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Quaternion as RosQuaternion
from squaternion import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

#import point_cloud2 as pc2

from config.config import TIME_DELTA, GOAL_REACHED_DIST, COLLISION_DIST, ENVIRONMENT_DIM


#TODO: change goal move to another script
#TODO: get reward move to another script
#TODO: rate of freq of odom sub

class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        self.environment_dim = ENVIRONMENT_DIM
        self.odom_x = 0
        self.odom_y = 0
        
        self.done = False
        self.last_odom = None

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
        # Odometry data
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        # Velodyne data
        self.velodyne_sub = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10)
        
        self.velodyne_data = np.ones(ENVIRONMENT_DIM) * 10  
        self.gaps = self.calculate_gaps()
        
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)
        
    
    def odom_callback(self, msg):
        self.last_odom = msg
    
    def velodyne_callback(self, msg):
        data = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(ENVIRONMENT_DIM) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0 # Calculate angle relative to robot front
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                # Find appropriate gap
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break
        
        
    # Perform an action and read a new state
    def step(self, action):
        
        laser_data = self.velodyne_data.copy()
        target = False
        
        if self.last_odom is None:
            self.get_logger().warn("Waiting for initial odometry data...")
            return np.zeros(self.state_dim), 0, True, False
        
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
        self.done, collision, min_laser = self.observe_collision(laser_data)
        v_state = []
        v_state[:] = laser_data[:] # or laser_data.tolist()
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x, self.odom_y, quaternion = self.get_current_pose()
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
            self.get_logger().info(f"GOAL REACHED! Target was: ({self.goal_x:.2f}, {self.goal_y:.2f})")
            target = True
            self.done = True

        # my new approach for better training in less episodes
        if target:
            self.successful_episodes += 1

        self.success_rate = self.successful_episodes / max(self.total_episodes, 1)
        self.get_logger().info(f"Success_Rate: {self.success_rate}")


        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        distance_to_g = self.odom_x
        reward= self.get_reward(target, collision, action, min_laser, distance)
        return state, reward, self.done, target

    def reset(self):
        # small maze
        #MAZE_MIN_X = 0.0
        #MAZE_MAX_X = 3.9
        #MAZE_MIN_Y = -2.8
        #MAZE_MAX_Y = 1.6
        # Reset velodyne data to default
        self.velodyne_data = np.ones(self.environment_dim) * 10
        
        # large maze
        self.done = False
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
            self.get_logger().info("/gazebo/reset_simulation service call failed")

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
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.get_logger().info("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            self.get_logger().info("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = self.velodyne_data.copy().tolist()
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
                        self.get_logger().info(f"GOAL with success_rate X: {self.goal_x}, GOAL Y with success_rate: {self.goal_y}")
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
            self.get_logger().info(f"GOAL X: {self.goal_x}, GOAL Y: {self.goal_y}")

        # Boundary-Check mit Fallback
        #self.goal_x = np.clip(self.goal_x, MAZE_MIN_X + 0.5, MAZE_MAX_X - 0.5)
        #self.goal_y = np.clip(self.goal_y, MAZE_MIN_Y + 0.5, MAZE_MAX_Y - 0.5)

        if not check_pos(self.goal_x, self.goal_y):
            self.get_logger().warn("Invalid goal detected! Using safe fallback")
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

    #@staticmethod
    def observe_collision(self, laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            self.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    #@staticmethod
    def get_reward(self, target, collision, action, min_laser, distance):
        if target:
            self.get_logger().info("reward 100")
            return 100.0
        elif collision:
            self.get_logger().info("reward -100")
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
        
    ### helpers ###    
    def get_current_pose(self):
        if self.last_odom is None:
            return None, None, None
            
        x = self.last_odom.pose.pose.position.x
        y = self.last_odom.pose.pose.position.y
        orientation = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        return x, y, orientation
    
    def calculate_gaps(self):
        """Pre-calculate angle gaps for Velodyne processing"""
        gaps = [[-np.pi/2 - 0.03, -np.pi/2 + np.pi/self.environment_dim]]
        for m in range(self.environment_dim - 1):
            gaps.append([gaps[m][1], gaps[m][1] + np.pi/self.environment_dim])
        gaps[-1][-1] += 0.03
        return gaps

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