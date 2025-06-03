from setuptools import setup, find_packages
import os
import glob

package_name = 'drl_robot_navigation'
root = os.path.dirname(__file__)  # where setup.py lives

# collect everything under models/td_robot/meshes/p3dx
p3dx_dir = os.path.join(root, 'models', 'td_robot', 'meshes', 'p3dx')
p3dx_files = [
    os.path.relpath(abs_path, root)
    for abs_path in glob.glob(os.path.join(p3dx_dir, '*.stl'))
]

# collect everything under models/td_robot/meshes/laser
laser_dir = os.path.join(root, 'models', 'td_robot', 'meshes', 'laser')
laser_files = [
    os.path.relpath(abs_path, root)
    for abs_path in glob.glob(os.path.join(laser_dir, '*.dae'))
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(package_name),
    package_dir={'': package_name},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # rviz file
        ('share/' + package_name + '/rviz', ['drl_robot_navigation/rviz/pioneer3dx.rviz']),
        # Install launch files
        ('share/' + package_name + '/launch', ['launch/training.launch.py']),
        ('share/' + package_name + '/launch', ['launch/robot_state_publisher.launch.py']),
        # Install URDF models
        ('share/' + package_name + '/urdf', ['urdf/td_robot.urdf']),
        # Install meshes
        ('share/' + package_name + '/models/td_robot/meshes/p3dx', p3dx_files),
        ('share/' + package_name + '/models/td_robot/meshes/laser', laser_files),
        # Install Gazebo worlds
        ('share/' + package_name + '/worlds', ['worlds/td3.world']),
        # Install robot models
        ('share/' + package_name + '/models/td_robot', 
            ['models/td_robot/model.config',
             'models/td_robot/td_robot.sdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='herberto.werner@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'training_node = nodes.training:main',
        ],
    },
)
