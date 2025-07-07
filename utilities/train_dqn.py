import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utilities.dqn_model import DQN
from utilities.replay_buffer import ReplayBuffer
from utilities.environment import Environment
from tqdm import tqdm

device = "cpu"

def train():
    env = Environment()
    input_dim = 12
    output_dim = 4

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer(5000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    target_update_freq = 10
    num_episodes = 1000
    max_steps = 1000
    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        env = Environment()
        state = np.array(env.step(0)['observations'], dtype=np.float32)
        total_reward = 0

        for t in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(output_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, device=device))
                    action = q_values.argmax().item()

            step_result = env.step(action)
            next_state = step_result['observations']
            reward = step_result['rewards']
            done = step_result['done']

            next_state_array = np.array(next_state, dtype=np.float32) if next_state is not None else None

            buffer.push(state, action, reward, next_state_array, done)
            state = next_state_array
            total_reward += reward

            if done:
                break

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors
                states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                # Handle non-final next states
                non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool, device=device)
                non_final_next_states = torch.tensor(
                    [s for s in next_states if s is not None],
                    dtype=torch.float32,
                    device=device
                )

                next_states_tensor = torch.zeros((batch_size, input_dim), dtype=torch.float32, device=device)
                next_states_tensor[non_final_mask] = non_final_next_states

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_net(next_states_tensor).max(1)[0].unsqueeze(1)

                expected_q = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                # Compute current Q-values
                q_values = policy_net(states_tensor).gather(1, actions_tensor)

                # Loss and update
                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return policy_net, episode_rewards
