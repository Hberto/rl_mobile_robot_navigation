# rl_mobile_robot_navigation
This repo consists of Training and Evaluation of a Deep Reinforcement Learning based mobile robot navigation. It's part of my learning journey how to apply Deep Reinforcement Learning to real-world problems. As base I'm using the example of [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation?tab=readme-ov-file). The main goal is to train a robot to navigate through a maze and reach the goal. The robot must avoid obstacle and navigate to the goal. 

## Installation with Docker
Following commands will start the Installation via Docker. 
```bash
cd ./Docker
docker-compose up --build
```
It will install:
- Ubuntu 20.04
- Python 3.8.10
- Pytorch 1.10
- ROS Noetic
- Tensorboard

## Deep Reinforcement Learning
This [medium article](https://medium.com/@reinis_86651/deep-reinforcement-learning-in-mobile-robot-navigation-tutorial-part1-installation-d62715722303) is part of my learning journey and source of this project. The downloaded code is based on the [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation?tab=readme-ov-file).

### Training
tba.
### Tests
tba.

## Credits
This project is based on the following sources:
- [DRL-robot-navigation base repo (ros1)](https://github.com/reiniscimurs/DRL-robot-navigation?tab=readme-ov-file)

- [Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning](https://arxiv.org/abs/2103.07119)

- [DRL-robot-navigation-ROS2](https://github.com/vishweshvhavle/deep-rl-navigation)

- [Transformer TD3 Base model](https://link.springer.com/article/10.1007/s11370-025-00620-2)

## TODOS
- [x] impl basic modular structure
- [x] adjust package.xml and cmake_list
- [x] read about robot state publisher
- [x] read paper about lstm navigation
- [x] seperate nodes to odom, gazebo env, velodyne node
- [x] params should be seperated or in config.yaml
- [ ] read about transformer td3
- [ ] impl transformer td3
- [ ] docker ros2
- [ ] test and evaluate transformer td3
- [ ] optimize transformer td3






