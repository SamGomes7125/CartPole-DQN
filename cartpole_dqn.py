import sys
import gymnasium as gym
import keras
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Training Parameters
n_episodes = 1000
n_win_ticks = 195
max_env_steps = None

gamma = 0.95  # Discount Factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
alpha = 0.01  # Learning rate
alpha_decay = 0.01

batch_size = 64
monitor = False
quiet = False

# Environment Setup
memory = deque(maxlen=50000)
env = gym.make("CartPole-v1", render_mode="human")
if max_env_steps is not None:
    env.max_episode_steps = max_env_steps

# Neural Network Model
model = Sequential([
    Dense(24, input_dim=4, activation="relu"),
    Dense(48, activation="relu"),
    Dense(2, activation="linear")
])
model.compile(loss="mse", optimizer=Adam(learning_rate=alpha, decay=alpha_decay))

# Memory Replay
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Epsilon-Greedy Action Selection
def choose_action(state, epsilon):
    return env.action_space.sample() if np.random.rand() <= epsilon else np.argmax(model.predict(state, verbose=0)[0])

# Epsilon Decay Strategy
def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t + 1) * epsilon_decay)))

# Reshape State
def preprocess_state(state):
    return np.reshape(state, [1, 4])

# Experience Replay
def replay(batch_size, epsilon):
    if len(memory) < batch_size:
        return  # Skip training if memory is too small

    minibatch = random.sample(memory, batch_size)
    x_batch, y_batch = [], []

    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state, verbose=0)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state, verbose=0)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=0)

# Training Function
def run():
    scores = deque(maxlen=100)

    for e in range(n_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        done = False
        i = 0
        epsilon = get_epsilon(e)  # Compute epsilon once per episode

        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Handle truncation
            if e % 20 == 0:  
                env.render()

            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1

        scores.append(i)
        mean_score = np.mean(scores)

        if mean_score >= n_win_ticks and e >= 100:
            if not quiet:
                print(f"Solved after {e-100} episodes!")
            return e - 100

        if e % 20 == 0 and not quiet:
            print(f"[Episode {e}] - Mean survival time over last 100 episodes: {mean_score} ticks.")

        replay(batch_size, epsilon)

    if not quiet:
        print(f"Did not solve after {n_episodes} episodes.")
    return n_episodes

# Run Training
run()
