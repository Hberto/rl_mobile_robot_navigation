# Deep RL Mobile Robot Navigation

Training and evaluation of a **Deep Reinforcement Learning** agent for mobile robot navigation in
ROS 2 + Gazebo. A differential-drive robot equipped with a Velodyne 3D LiDAR learns to navigate
toward a goal while avoiding obstacles.

The project explores two policy architectures:

- **TD3** — the classic Twin Delayed DDPG actor–critic baseline.
- **Transformer-TD3** — TD3 extended with a Transformer encoder that consumes a history of the last
  `HISTORY_LENGTH` observations, so the policy can reason over temporal context instead of a single
  state.

This is a learning project: the goal is to apply Deep RL to a realistic, partially observable
robotics problem end-to-end (simulation, sensor processing, reward shaping, training, evaluation).
It builds on [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation) and its
[ROS 2 port](https://github.com/vishweshvhavle/deep-rl-navigation).

## Stack

| Component | Version / Tool |
|-----------|----------------|
| ROS 2     | Humble |
| Simulator | Gazebo (Classic) via `gazebo_ros_pkgs` |
| Sensor    | Velodyne 3D LiDAR (`velodyne_simulator`) |
| RL        | PyTorch (CUDA 12.1) |
| Logging   | TensorBoard |
| Runtime   | Docker / docker-compose with NVIDIA GPU |

## Repository structure

```
deep_rl_robot_navigation/src/drl_robot_navigation/
├── drl_robot_navigation/
│   ├── agent/
│   │   ├── td3/                 # TD3 baseline (actor, critic, replay buffer)
│   │   └── td3_transformer/     # Transformer-TD3 (history encoder, positional encoding)
│   ├── config/config.py         # All hyperparameters and paths
│   ├── env/gazebo_env.py        # ROS 2 / Gazebo environment + reward shaping
│   ├── evaluation/evaluation.py # Evaluation loop (success rate, path length, time-to-goal)
│   ├── nodes/training.py        # Training entry point (training_node)
│   └── utils/                   # Helpers
├── launch/                      # Gazebo + robot_state_publisher + RViz launch
├── models/, urdf/, worlds/      # Robot model, URDF, Gazebo worlds
Docker/                          # Dockerfile + docker-compose
```

## Installation (Docker)

The recommended way to run the project is the provided Docker setup (ROS 2 Humble, Gazebo, PyTorch
with CUDA, TensorBoard). It requires the **NVIDIA Container Toolkit** for GPU access and an X server
for the Gazebo/RViz GUI.

```bash
cd Docker
docker-compose up --build
```

The image installs:

- ROS 2 Humble desktop + `gazebo_ros_pkgs`
- PyTorch / torchvision / torchaudio (CUDA 12.1) and TensorBoard
- `squaternion`, `numpy` and ROS build tooling (`colcon`, `rosdep`)

The host workspace is mounted into the container at `/ros2_ws/src`, and TensorBoard's port `6006`
is forwarded to the host.

## Build & run

Inside the container (`docker exec -it ros2_drl bash`):

```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

**1. Start the simulation** (Gazebo world, robot state publisher, RViz):

```bash
ros2 launch drl_robot_navigation training.launch.py
# headless: append gui:=false rviz:=false
```

**2. Start training** (in a second shell, after sourcing `install/setup.bash`):

```bash
ros2 run drl_robot_navigation training_node
```

**3. Monitor with TensorBoard:**

```bash
tensorboard --logdir drl_robot_navigation/evaluation/run --port 6006 --bind_all
# open http://localhost:6006
```

Trained models are written to `evaluation/models/` and run logs to `evaluation/run/`
(both git-ignored).

## Configuration

All hyperparameters live in
[config/config.py](deep_rl_robot_navigation/src/drl_robot_navigation/drl_robot_navigation/config/config.py),
including TD3 parameters (discount, tau, policy noise/freq), exploration schedule, the
Transformer architecture (`MODEL_DIM`, `N_HEADS`, `N_ENCODER_LAYERS`, `HISTORY_LENGTH`), and the
environment constants (`GOAL_REACHED_DIST`, `COLLISION_DIST`, LiDAR/robot state dimensions).
Set `LOAD_MODEL = True` to resume from a saved policy.

## Evaluation

Evaluation runs periodically during training (every `EVAL_FREQ` steps, `EVAL_EP` episodes) and
reports success rate, average path length, time-to-goal and collisions. See
[evaluation/evaluation.py](deep_rl_robot_navigation/src/drl_robot_navigation/drl_robot_navigation/evaluation/evaluation.py).

## Roadmap

- [x] Modular package structure (env / agent / config / evaluation nodes)
- [x] TD3 baseline
- [x] Transformer-TD3 with observation-history encoder
- [x] Reward shaping (progress reward + obstacle penalty)
- [x] Dockerized ROS 2 Humble + GPU setup
- [x] Evaluation pipeline (success rate, path length, time-to-goal)
- [ ] Hyperparameter optimization of Transformer-TD3
- [ ] Generalization across multiple maze worlds

## Credits

This project is based on the following sources:

- [DRL-robot-navigation (ROS 1 base repo)](https://github.com/reiniscimurs/DRL-robot-navigation)
- [Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning](https://arxiv.org/abs/2103.07119)
- [DRL-robot-navigation (ROS 2 port)](https://github.com/vishweshvhavle/deep-rl-navigation)
- [Transformer-TD3 base model](https://link.springer.com/article/10.1007/s11370-025-00620-2)
- [Tutorial / Medium article](https://medium.com/@reinis_86651/deep-reinforcement-learning-in-mobile-robot-navigation-tutorial-part1-installation-d62715722303)

## License

Released under the [MIT License](LICENSE).
