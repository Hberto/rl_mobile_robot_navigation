import numpy as np
from config.config import TIME_DELTA

def evaluate(network, epoch, env, eval_episodes=10):
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