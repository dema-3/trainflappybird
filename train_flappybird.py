

import os
import time
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_HYPERPARAMS = {
    "env_id": "FlappyBird-v0",
    "use_lidar": True,
    "render_mode": None,
    "seed": 42,
    "episodes": 800,
    "max_steps_per_episode": 5000,
    "gamma": 0.99,
    "lr": 1e-3,
    "batch_size": 64,
    "replay_memory_size": 50000,
    "min_replay_size": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 100000,
    "target_update_every_steps": 2000,
    "train_every_steps": 4,
    "hidden_size": 128,
    "save_plots": True,
    "plots_dir": "plots",
}

def load_hyperparameters(path="hyperparameters.yml"):
    hp = DEFAULT_HYPERPARAMS.copy()
    if os.path.exists(path) and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for k, v in loaded.items():
            if k in hp:
                hp[k] = v
    return hp


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hp, device):
        self.device = device
        self.gamma = hp["gamma"]
        self.batch_size = hp["batch_size"]
        self.train_every_steps = hp["train_every_steps"]
        self.target_update_every_steps = hp["target_update_every_steps"]

        self.eps_start = hp["epsilon_start"]
        self.eps_end = hp["epsilon_end"]
        self.eps_decay_steps = hp["epsilon_decay_steps"]
        self.steps_done = 0

        self.policy_net = QNetwork(state_dim, action_dim, hp["hidden_size"]).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hp["hidden_size"]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hp["lr"])
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayMemory(hp["replay_memory_size"])
        self.min_replay_size = hp["min_replay_size"]

    def epsilon(self):
        if self.steps_done >= self.eps_decay_steps:
            return self.eps_end
        frac = self.steps_done / float(self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state_np):
        eps = self.epsilon()
        self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.policy_net.net[-1].out_features)

        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(torch.argmax(q, dim=1).item())

    def optimize_model(self):
        if len(self.memory) < self.min_replay_size:
            return None

        transitions = self.memory.sample(self.batch_size)

        state = torch.tensor(np.array(transitions.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(transitions.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(transitions.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)

        with torch.no_grad():
            next_q_max = self.target_net(next_state).max(dim=1, keepdim=True)[0]
            target = reward + self.gamma * next_q_max * (1.0 - done)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def maybe_update_target(self):
        if self.steps_done % self.target_update_every_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def make_env(hp):
    return gym.make(
        hp["env_id"],
        render_mode=hp["render_mode"],
        use_lidar=hp["use_lidar"],
    )


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_rewards(reward_history, out_dir):
    ensure_dir(out_dir)
    window = 50
    means = []
    for i in range(len(reward_history)):
        start = max(0, i - window + 1)
        means.append(np.mean(reward_history[start:i + 1]))

    plt.figure()
    plt.title("Training Rewards (Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(means)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_rewards.png"))
    plt.close()


def train():
    hp = load_hyperparameters()

    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = make_env(hp)
    obs, info = env.reset(seed=hp["seed"])

    state_dim = int(np.array(obs).shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(state_dim, action_dim, hp, device)

    reward_history = []
    loss_history = []
    total_steps = 0

    for ep in range(1, hp["episodes"] + 1):
        obs, info = env.reset()
        state = np.array(obs, dtype=np.float32)

        ep_reward = 0.0
        ep_losses = []

        for t in range(hp["max_steps_per_episode"]):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = np.array(next_obs, dtype=np.float32)
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

            if total_steps % agent.train_every_steps == 0:
                loss = agent.optimize_model()
                if loss is not None:
                    ep_losses.append(loss)

            agent.maybe_update_target()
            total_steps += 1

            if done:
                break

        reward_history.append(ep_reward)
        if ep_losses:
            loss_history.append(np.mean(ep_losses))

        if ep % 10 == 0:
            avg_last_50 = np.mean(reward_history[-50:])
            print(f"Episode {ep} | Reward: {ep_reward:.2f} | Avg(50): {avg_last_50:.2f} | Epsilon: {agent.epsilon():.3f}")

    env.close()

    if hp["save_plots"]:
        plot_rewards(reward_history, hp["plots_dir"])

    torch.save(agent.policy_net.state_dict(), "dqn_flappybird_policy.pt")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()
