from env.gazebo_env import GazeboEnv
import rclpy
import threading
import os
import torch
import numpy as np
from collections import deque

from agent.td3_transformer.td3 import td3
from agent.td3_transformer.replay_historic_buffer import ReplayBuffer
from config import config
from evaluation.evaluation import evaluate

def main():
    rclpy.init(args=None)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU found! Training will be applied on '{config.DEVICE}'")
        print(f"Name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"No GPU found. Training is running on '{config.DEVICE}' instead.")
    
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
    network = td3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        env=env,
        history_size=config.HISTORY_LENGTH,
        model_dim=config.MODEL_DIM,
        nhead=config.N_HEADS,
        num_encoder_layers=config.N_ENCODER_LAYERS
    )
    
    # Create a replay buffer
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE, config.HISTORY_LENGTH, config.SEED)
    history_deque = deque(maxlen=config.HISTORY_LENGTH)
    
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
    
    # Reset and Init
    state = env.reset()
    done = False 
    episode_timesteps = 0
    episode_reward = 0


    history_deque = deque(maxlen=config.HISTORY_LENGTH)
    for _ in range(config.HISTORY_LENGTH):
        history_deque.append(state)
    
    try:
        while rclpy.ok():
            if timestep < config.MAX_TIMESTEPS:
                # On termination of episode
                if done:
                    if timestep != 0 and replay_buffer.size() > config.BATCH_SIZE + config.HISTORY_LENGTH:
                        env.get_logger().info(f"Done. timestep : {timestep}")
                       
                        env.get_logger().info(f"Training for {episode_timesteps} iterations...")
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
                                evaluate(network=network, epoch=epoch, env=env, history_size=config.HISTORY_LENGTH, eval_episodes=config.EVAL_EP)
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
                        
                        history_deque.clear()
                        for _ in range(config.HISTORY_LENGTH):
                            history_deque.append(state)

                # add some exploration noise
                if config.EXPL_NOISE > config.EXPL_MIN:
                    config.EXPL_NOISE = config.EXPL_NOISE - ((1 - config.EXPL_MIN) / config.EXPL_DECAY_STEPS)
                
                history = np.array(history_deque)
                
                action = network.get_action(np.array(state), history)
                
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

                # Update state and history
                state = next_state
                history_deque.append(next_state)
                
                # Update the counters                
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

    except KeyboardInterrupt:
        env.get_logger().info("Keyboard interrupt, shutting down.")
        pass

    rclpy.shutdown()
    executor_thread.join()
            
    

if __name__ == '__main__':
    main()