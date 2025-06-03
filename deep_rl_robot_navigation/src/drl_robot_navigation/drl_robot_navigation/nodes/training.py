from env.gazebo_env import GazeboEnv
import rclpy
import threading
import os
import torch
import numpy as np
from agent.td3.td3 import td3
from agent.td3.replay_buffer import ReplayBuffer
from config import config
from evaluation.evaluation import evaluate

def main():
    rclpy.init(args=None)
    
    # Add Configs
      # Create the network storage folders
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    if config.SAVE_MODEL and not os.path.exists(config.PYTORCH_MODELS_DIR):
        os.makedirs(config.PYTORCH_MODELS_DIR)
    if not os.path.exists(config.SUMMARY_WRITER_RUN_LOG):
        os.makedirs(config.SUMMARY_WRITER_RUN_LOG)
    
    # Create the training environment
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    state_dim = config.ENVIRONMENT_DIM + config.ROBOT_DIM
    action_dim = 2
    max_action = 1
    
    # Get the agent
    env = GazeboEnv()
     
    # Create the network
    network = td3(state_dim, action_dim, max_action, env)
    # Create a replay buffer
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE, config.SEED)
    if config.LOAD_MODEL:
        try:
            env.get_logger().info("Will load existing model.")
            network.load(config.FILE_NAME, config.PYTORCH_MODELS_DIR)
        except:
            env.get_logger().info("Could not load the stored model parameters, initializing training with random parameters")
     
    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        while rclpy.ok():
            if timestep < config.MAX_TIMESTEPS:
                # On termination of episode
                if done:
                    env.get_logger().info(f"Done. timestep : {timestep}")
                    if timestep != 0:
                        env.get_logger().info(f"train")
                        network.train(
                        replay_buffer,
                        episode_timesteps,
                        config.BATCH_SIZE,
                        config.DISCOUNT,
                        config.TAU,
                        config.POLICY_NOISE,
                        config.NOISE_CLIP,
                        config.POLICY_FREQ,
                        )

                    if timesteps_since_eval >= config.EVAL_FREQ:
                        env.get_logger().info("Validating")
                        timesteps_since_eval %= config.EVAL_FREQ
                        evaluations.append(
                            evaluate(network=network, epoch=epoch, eval_episodes=config.EVAL_EP, env=env)
                        )
                        if config.SAVE_MODEL:
                            model_path = config.PYTORCH_MODELS_DIR
                            network.save(config.FILE_NAME, directory=model_path)
                        # Save evaluation
                        path = os.path.join(config.RESULTS_DIR, config.FILE_NAME + ".npy")
                        np.save(path, np.array(evaluations))
                        epoch += 1
                        env.get_logger().info(f"Epoch: {epoch}")

                    state = env.reset()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                # add some exploration noise
                if config.EXPL_NOISE > config.EXPL_MIN:
                    config.EXPL_NOISE = config.EXPL_NOISE - ((1 - config.EXPL_MIN) / config.EXPL_DECAY_STEPS)

                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, config.EXPL_NOISE, size=action_dim)).clip(
                     -max_action, max_action
                )

                # If the robot is facing an obstacle, randomly force it to take a consistent random action.
                # This is done to increase exploration in situations near obstacles.
                # Training can also be performed without it
                if config.RANDOM_NEAR_OBSTACLE:
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
                done_bool = 0 if episode_timesteps + 1 == config.MAX_EP else int(done)
                done = 1 if episode_timesteps + 1 == config.MAX_EP else int(done)
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
            
    

if __name__ == '__main__':
    main()