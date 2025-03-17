🏆 CartPole DQN - Deep Q-Network (DQN) with Gymnasium
A Deep Q-Learning (DQN) agent trained using Keras & Gymnasium to balance a pole on a moving cart.

📌 Project Overview
This project implements Deep Q-Learning (DQN) using a Neural Network to solve the CartPole-v1 environment in OpenAI Gymnasium. The goal is to train an agent to keep the pole balanced by applying reinforcement learning techniques.

✅ Key Features
✔ Implements Deep Q-Network (DQN) using Keras & TensorFlow
✔ Uses Experience Replay to stabilize training
✔ Implements Epsilon-Greedy Exploration
✔ Trains on CartPole-v1 environment

⚡ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/cartpole-dqn.git
cd cartpole-dqn
2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the DQN Agent
python cartpole_dqn.py

🎮 How It Works
1️⃣ Environment Setup: The agent interacts with the CartPole-v1 environment from OpenAI Gym.
2️⃣ Neural Network: A 2-layer MLP (Multi-Layer Perceptron) is used as the Q-function approximator.
3️⃣ Training: The agent learns using Experience Replay & Epsilon-Greedy Exploration.
4️⃣ Rewards & Learning: The agent gets a reward for balancing the pole and is penalized when it falls.

📌 Future Improvements
🔹 Implement Double DQN to reduce overestimation bias
🔹 Use Prioritized Experience Replay for better sample efficiency
🔹 Experiment with different neural network architectures

📚 References & Acknowledgments
OpenAI Gymnasium Docs: https://gymnasium.farama.org/
Deep Q-Networks (DQN) Paper: https://arxiv.org/abs/1312.5602
Keras & TensorFlow Documentation: https://www.tensorflow.org/

👨‍💻 Contributing
Feel free to fork this repo, submit pull requests, or open issues for any suggestions or improvements!

