import gymnasium as gym
import os
import time
import numpy as np
from collections import deque
import random

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def to_file(self, file: str):
        """Saves the model's state dictionary to a file."""
        torch.save(self.state_dict(), file)

    @classmethod
    def from_file(cls, file: str, input_dim=8, hidden_dim=128, output_dim=4):
        """Loads the model from a file if it exists; otherwise, returns a new model."""
        model = cls(input_dim, hidden_dim, output_dim)  # Create an instance
        if os.path.exists(file):
            model.load_state_dict(torch.load(file))  # Load weights
            model.eval()  # Set to evaluation mode
        return model  # Return new or loaded model


def landing(args):
    gamma = args.gamma
    epsilon_min = args.epsilon
    epsilon = 1.0
    max_buffer = args.max_buffer
    max_episodes = args.episodes
    model_path = args.model
    max_moves = args.max_moves
    static_punishment = args.static_punishment
    todo = args.action
    batch = args.batch
    lr = args.learning_rate
    display = args.display_every
    device = args.device

    model = DQN.from_file(model_path).to(device)
    model_target = DQN().to(device)
    model_target.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)

    env_show = gym.make("LunarLander-v3", render_mode="human")

    if todo == "inference":
        keep_inferencing = True
        episodes = 0
        while keep_inferencing:
            moves = 0
            state, _ = env.reset()
            keep_disp = True
            while keep_disp:
                action = np.argmax(model(torch.from_numpy(dispstate)).detach().numpy())
                dispstate, reward, terminated, truncated, info = env_show.step(action)
                if terminated:
                    keep_disp = False
                moves += 1
            episodes += 1
            if episodes > max_episodes:
                keep_inferencing = False
        return

    env = gym.make("LunarLander-v3")

    state, info = env.reset(seed=42)

    D = deque(maxlen=max_buffer)
    iteration = 0

    xs = []
    ys = []

    optimizer.zero_grad()
    loss_func = nn.MSELoss()

    rewards_history = deque(maxlen=500)
    episode_count = 0
    reward_sum = 0

    # Training
    keep_training = True
    training_iteration_moves = 0
    while keep_training:
        state = torch.from_numpy(state).to(device)
        action = np.argmax(model(state).cpu().detach().numpy())
        e = torch.rand(1).item()
        if e < epsilon:
            action = torch.randint(0, 4, (1,)).item()
        newstate, reward, terminated, truncated, info = env.step(action)
        reward -= static_punishment
        reward_sum += reward
        newstate = torch.from_numpy(newstate)

        training_iteration_moves += 1

        if training_iteration_moves > max_moves:
            training_iteration_moves = 0
            state, _ = env.reset()
            continue

        D.append((state, action, reward, newstate, terminated))

        state = newstate.numpy()

        epsilon = max(args.epsilon, epsilon * 0.9995)

        if terminated:
            state, _ = env.reset()
            rewards_history.append(reward_sum)
            episode_count += 1

            xs.append(episode_count)
            reward_avg = sum(rewards_history) * 1.0 / len(rewards_history)
            ys.append(reward_avg)

            reward_sum = 0

            # Update target network weights
            model_target.load_state_dict(model.state_dict())

            if episode_count > max_episodes:
                keep_training = False

            print(
                f"Episode {episode_count}: reward_avg: {reward_avg} (epsilon: {epsilon})"
            )
            if (episode_count % display) == 0:
                keep_disp = True
                dispstate, _ = env_show.reset()
                moves = 0
                while keep_disp:
                    action = np.argmax(
                        model(torch.from_numpy(dispstate).to(device)).cpu().detach().numpy()
                    )
                    dispstate, reward, terminated, truncated, info = env_show.step(
                        action
                    )
                    if terminated:
                        keep_disp = False
                    moves += 1
                    if moves > max_moves:
                        keep_disp = False

        # Learning
        experience = random.sample(D, min(batch, len(D)))

        optimizer.zero_grad()

        loss_total = 0

        states, actions, rewards, newstates, terminations = zip(*experience)

        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
        newstates = torch.stack(
            [torch.tensor(ns, dtype=torch.float32) for ns in newstates]
        ).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        terminations = torch.tensor(terminations, dtype=torch.bool).to(device)

        # Get Q-values from target model for new states
        q_target = model_target(newstates)
        q_max = q_target.max(dim=1)[0]

        y = rewards + gamma * q_max * (
            ~terminations
        )

        # Get Q-values for chosen actions from the main model
        q_values = model(states)  # Get Q-values for all actions
        q_selected = q_values.gather(1, torch.tensor(actions).unsqueeze(1).to(device)).squeeze(
            1
        )  # Get Q-values for taken actions
        # Compute loss
        loss = loss_func(q_selected, y)

        # Backpropagate and update model
        loss.backward()
        optimizer.step()

        iteration += 1

    model = model.to("cpu")
    model.to_file(model_path)

    # Create plot
    plt.plot(xs, ys, linestyle="-")

    # Labels and title
    plt.xlabel("episode")
    plt.ylabel("reward accumulated")
    plt.title("rewards over episodes")

    # Show plot
    plt.show()  # Show without blocking execution


def main():
    # argument parser
    import argparse

    parser = argparse.ArgumentParser(description="Policy Iteration")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon")
    parser.add_argument(
        "--learning-rate", type=float, default=0.0003, help="Learning rate"
    )
    parser.add_argument(
            "--static-punishment", type=float, default=0.01, help="Static negative reward, to avoid suboptimal 'fly forever' solution"
    )
    parser.add_argument(
        "--action", type=str, default="retrain", help="What to do: [train|inference]"
    )
    parser.add_argument(
        "--model", type=str, default="lunar_lander.pth", help="Path to a .pth model"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to use"
    )
    parser.add_argument("--display-every", type=int, default=10, help="Display every")
    parser.add_argument("--batch", type=int, default=32, help="Display every")
    parser.add_argument("--max-moves", type=int, default=1000, help="Max moves (updates) to display during training")
    parser.add_argument(
        "--max-buffer", type=int, default=100000, help="Max buffer size"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Episodes to train or inference"
    )
    args = parser.parse_args()
    print(args)
    landing(args)


if __name__ == "__main__":
    main()
