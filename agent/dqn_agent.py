"""
agent/dqn_agent.py
==================
Deep Q-Network (DQN) agent with:
  - MLP and RNN (LSTM) architectures
  - Experience replay buffer
  - Target network (synced every `target_update` gradient steps)
  - ε-greedy exploration with exponential decay
  - Huber (SmoothL1) loss for stable training
  - Gradient clipping
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config


# ── MLP Q-Network ─────────────────────────────────────────────────────────────
class MLPQNetwork(nn.Module):
    """Standard Feedforward network: state → Q(s, a)"""

    def __init__(self, state_size: int, action_size: int, hidden: int = config.HIDDEN_SIZE):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ── Recurrent Q-Network ────────────────────────────────────────────────────────
class RecurrentQNetwork(nn.Module):
    """LSTM-based network: sequence of (state, prev_action) → Q(s_t, a)"""

    def __init__(self, state_size: int, action_size: int, hidden: int = config.RNN_HIDDEN_SIZE):
        super().__init__()
        # Input: state + one-hot of previous action
        self.input_size = state_size + action_size
        self.hidden_size = hidden
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state for Q-values
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        
        s_arr = np.array(s, dtype=np.float32)
        ns_arr = np.array(ns, dtype=np.float32)

        return (
            torch.from_numpy(s_arr),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.from_numpy(ns_arr),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = config.LR,
        gamma: float = config.GAMMA,
        epsilon_start: float = config.EPSILON_START,
        epsilon_end: float = config.EPSILON_END,
        epsilon_decay: float = config.EPSILON_DECAY,
        batch_size: int = config.BATCH_SIZE,
        target_update: int = config.TARGET_UPDATE,
        buffer_cap: int = config.BUFFER_CAPACITY,
        hidden_size: int = None,
        use_double: bool = config.USE_DDQN,
        seq_len: int = config.SEQUENCE_LENGTH,
        model_type: str = config.MODEL_TYPE
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double = use_double
        self.seq_len = seq_len
        self.model_type = model_type
        self.is_recurrent = (model_type == "RNN")
        self._update_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.is_recurrent:
            h = hidden_size or config.RNN_HIDDEN_SIZE
            self.q_net = RecurrentQNetwork(state_size, action_size, hidden=h).to(self.device)
            self.target_net = RecurrentQNetwork(state_size, action_size, hidden=h).to(self.device)
        else:
            h = hidden_size or config.HIDDEN_SIZE
            self.q_net = MLPQNetwork(state_size, action_size, hidden=h).to(self.device)
            self.target_net = MLPQNetwork(state_size, action_size, hidden=h).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self.loss_fn = nn.SmoothL1Loss()

        if self.is_recurrent:
            self.history = deque(maxlen=self.seq_len)
            self.last_action = 0

    def _get_one_hot_action(self, action: int) -> np.ndarray:
        oh = np.zeros(self.action_size, dtype=np.float32)
        oh[action] = 1.0
        return oh

    def _get_combined_input(self, state: np.ndarray, last_action: int) -> np.ndarray:
        oh_action = self._get_one_hot_action(last_action)
        return np.concatenate([state, oh_action])

    def _get_padded_history(self) -> np.ndarray:
        history_list = list(self.history)
        if len(history_list) < self.seq_len:
            padding = [np.zeros(self.state_size + self.action_size, dtype=np.float32)] * (self.seq_len - len(history_list))
            history_list = padding + history_list
        return np.array(history_list)

    def reset_history(self):
        if self.is_recurrent:
            self.history.clear()
            self.last_action = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if self.is_recurrent:
            combined = self._get_combined_input(state, self.last_action)
            self.history.append(combined)
        
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                if self.is_recurrent:
                    seq = self._get_padded_history()
                    inp = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                else:
                    inp = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.q_net(inp).argmax(dim=1).item()
        
        if self.is_recurrent:
            self.last_action = action
        return action

    def store(self, state, action, reward, next_state, done):
        if not self.is_recurrent:
            self.buffer.push(state, action, reward, next_state, float(done))
            return

        curr_seq = self._get_padded_history()
        next_combined = self._get_combined_input(next_state, action)
        next_history_list = list(self.history) + [next_combined]
        if len(next_history_list) > self.seq_len:
            next_history_list = next_history_list[1:]

        if len(next_history_list) < self.seq_len:
            padding = [np.zeros(self.state_size + self.action_size, dtype=np.float32)] * (self.seq_len - len(next_history_list))
            next_seq = np.array(padding + next_history_list)
        else:
            next_seq = np.array(next_history_list)
        self.buffer.push(curr_seq, action, reward, next_seq, float(done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        q_curr = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.use_double:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                q_next = self.target_net(next_states).max(dim=1)[0]
                
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._update_count += 1
        if self._update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "q_net":        self.q_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "update_count": self._update_count,
            "model_type":   self.model_type
        }, path)
        print(f"  [ckpt saved] {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon      = ckpt["epsilon"]
        self._update_count = ckpt["update_count"]
        print(f"  [ckpt loaded] {path}")
