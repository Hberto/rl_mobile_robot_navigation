import numpy as np
from config.config import TIME_DELTA
from collections import deque

def evaluate(network, epoch, env, history_size, eval_episodes=10):
    episode_rewards = []
    successful_path_lengths = []
    successful_times_to_goal = []
    total_collisions = 0
    total_successes = 0

    distance_history = []
    all_actions = []

    for ep in range(eval_episodes):
        env.get_logger().info(f"Evaluating episode {ep + 1}/{eval_episodes}")
        state = env.reset()
        done = False
        episode_steps = 0

        episode_reward = 0
        episode_actions = []
        path_length = 0.0

        history_deque = deque(maxlen=history_size)
        for _ in range(history_size):
            history_deque.append(state)

        while not done and episode_steps < 501:
            history = np.array(history_deque)
            action = network.get_action(np.array(state), history)
            env.get_logger().info(f"Action taken: {np.round(action, 2)}")
            
            episode_actions.append(action)
            all_actions.append(action)
            
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            
            history_deque.append(next_state)

            # Track metrics
            current_distance = np.linalg.norm([env.odom_x - env.goal_x, env.odom_y - env.goal_y])
            distance_history.append(current_distance)
            episode_reward += reward

            state = next_state
            episode_steps += 1
        episode_rewards.append(episode_reward)

        if target:
            total_successes += 1
            episode_linear_velocities = [(act[0] + 1) / 2 for act in episode_actions]
            path_length = np.sum(episode_linear_velocities) * TIME_DELTA
            successful_path_lengths.append(path_length)
            successful_times_to_goal.append(episode_steps * TIME_DELTA)

        if reward <= -90:
            total_collisions += 1
        

        # Episode statistics
        env.get_logger().info(
            f"Episode {ep + 1} | "
            f"Reward: {episode_reward:.1f} | "
            f"Steps: {episode_steps} | "
            f"Path: {path_length:.2f}m | "
            f"Success: {target}"
        )

    # Calculate aggregates
    avg_reward = np.mean(episode_rewards)
    success_rate = total_successes / eval_episodes
    collision_rate = total_collisions / eval_episodes
    
    avg_path_length = np.mean(successful_path_lengths) if total_successes > 0 else 0
    avg_time = np.mean(successful_times_to_goal) if total_successes > 0 else 0

    # Convert actions to numpy array
    all_actions_np = np.array(all_actions)

    # Log to TensorBoard
    network.writer.add_scalar("Evaluation/Average Reward", avg_reward, epoch)
    network.writer.add_scalar("Evaluation/Success Rate", success_rate, epoch)
    network.writer.add_scalar("Evaluation/Collision Rate", collision_rate, epoch)
    network.writer.add_scalar("Evaluation/Avg Path Length (Success)", avg_path_length, epoch)
    network.writer.add_scalar("Evaluation/Avg Time to Goal (Success)", avg_time, epoch)
    network.writer.add_histogram("Evaluation/Distance Distribution", np.array(distance_history), epoch)
    
    if all_actions_np.size > 0:
        network.writer.add_histogram("Actions/Linear", all_actions_np[:, 0], epoch)
        network.writer.add_histogram("Actions/Angular", all_actions_np[:, 1], epoch)

    # Console logging
    env.get_logger().info("\n" + "="*60)
    env.get_logger().info(f"Evaluation Epoch {epoch} Results:")
    env.get_logger().info(f"- Average Reward: {avg_reward:.2f}")
    env.get_logger().info(f"- Success Rate: {success_rate*100:.1f}%")
    env.get_logger().info(f"- Collision Rate: {collision_rate*100:.1f}%")
    env.get_logger().info(f"- Avg Path Length (successful episodes): {avg_path_length:.2f}m")
    env.get_logger().info(f"- Avg Time to Goal (successful episodes): {avg_time:.2f}s")
    env.get_logger().info("="*60 + "\n")

    return avg_reward