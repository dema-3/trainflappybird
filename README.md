# Train Flappy Bird using Deep Q-Network (DQN)

This project trains an AI agent to play Flappy Bird using Deep Reinforcement Learning (DQN).  

The implementation includes:
- Experience Replay (Replay Memory)
- Target Network
- Epsilon-Greedy Exploration
- Reward Tracking and Training Plot
- Model Saving

---

## Environment

This project uses:

- gymnasium
- flappy-bird-gymnasium
- PyTorch
- NumPy
- Matplotlib

Environment ID:
```
FlappyBird-v0
```

---

## How It Works

The agent interacts with the environment using the following process:

1. Observe current state
2. Select action using epsilon-greedy strategy
3. Store transition in replay memory
4. Sample mini-batches from replay memory
5. Update Q-network using target network
6. Periodically update target network

The goal is to maximize cumulative reward over episodes.

---

## Files

- `train_flappybird.py` → Main training script (single-file implementation)
- `requirements.txt` → Required Python libraries
- `dqn_flappybird_policy.pt` → Saved model (after training)
- Project PDF slides → Explanation of implementation and results

---

## Installation

Create a virtual environment (recommended):

```
python -m venv venv
```

Activate it:

Windows:
```
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Run Training

```
python train_flappybird.py
```

Training rewards will be plotted and the model will be saved after training.

---

## Training Results

![Training Result](images/page_12.png)

![Training Log](images/page_13.png)

The original project code was reconstructed based on my project slides after the initial files were lost.  
The implementation follows the same structure and algorithm (DQN with replay memory and target network).

