#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = 'td3.world'
    #world_file_name = 'small_maze.world'
    world = os.path.join(get_package_share_directory('drl_robot_navigation'), 'worlds', world_file_name)
    launch_file_dir = os.path.join(get_package_share_directory('drl_robot_navigation'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    rviz_file = os.path.join(get_package_share_directory('drl_robot_navigation'), 'rviz', 'pioneer3dx.rviz')
    
    use_gui = DeclareLaunchArgument(
        name='gui',
        default_value='true',
        description='Flag to enable or disable Gazebo GUI'
    )
    
    use_rviz = DeclareLaunchArgument(
        name='rviz',
        default_value='true',
        description='Flag to enable or disable RViz'
    )

    start_gazebo_server= IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        )

    start_gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    robot_state= IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        )

    static_transform = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom']
        )
    
    rviz = Node(package='rviz2',
            executable='rviz2',
            name='rviz2',  
            arguments=['-d', rviz_file],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
            ],
            condition=IfCondition(LaunchConfiguration('rviz'))
        )

    return LaunchDescription([
        use_gui,
        use_rviz,
        start_gazebo_server,
        start_gzclient,
        robot_state,
        static_transform,
        rviz
    ])