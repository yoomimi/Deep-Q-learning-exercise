# Deep Q-Learning Implementation for Grid World Environment

This project implements Deep Q-Learning (DQN) to train an agent to navigate a grid world environment. The agent learns to maximize its cumulative reward by interacting with the environment and updating its knowledge using a neural network. The main components of the project are described below:

## Environment Setup

The environment is a grid world where the agent can move in four possible directions: left, right, up, and down. The agent starts at a random position and aims to reach the goal while avoiding obstacles. The environment provides rewards based on the agent's actions and positions.

## Replay Buffer

The `ReplayBuffer` class is used to store and manage the agent's experiences, which consist of state transitions, actions, rewards, next states, and done flags. The buffer enables efficient sampling of mini-batches for training the neural network. This helps in breaking the correlation between consecutive experiences and stabilizing training.

## Agent

The `AGENT` class encapsulates the DQN algorithm. The main components of the agent are:

1. **Initialization**:
   - The agent is initialized with the environment, a replay buffer, and the necessary parameters for training.
   - The neural network (`Q_net`) and its target network (`Q_net_target`) are created using the `MLP` class. The target network helps in stabilizing the training by providing fixed Q-values for a certain number of steps.

2. **Episode Initialization**:
   - At the start of each episode, the agent's position is randomly initialized, ensuring it is not placed on a goal or an obstacle.

3. **Deep Q-Learning**:
   - The `deep_Q_learning` method is the core training loop where the agent interacts with the environment for a specified number of episodes.
   - For each episode, the agent takes actions based on an epsilon-greedy policy, stores the transitions in the replay buffer, and updates the Q-values using mini-batches sampled from the buffer.
   - The agent’s neural network is trained using stochastic gradient descent (SGD) to minimize the mean squared error (MSE) loss between predicted Q-values and target Q-values.
   - The target Q-values are computed using the Bellman equation: `target = r + γ max(Q'(s', a'))`, where `γ` is the discount factor.

4. **Action Selection**:
   - The `get_action` method implements an epsilon-greedy policy, balancing exploration and exploitation. With probability epsilon, the agent selects a random action; otherwise, it selects the action with the highest Q-value.

5. **Updating the Target Network**:
   - Periodically, the weights of the target network are updated to match the current network, helping to stabilize training by providing consistent targets.

6. **Logging and Monitoring**:
   - Training progress is logged periodically, including metrics such as the number of episodes, loss, and average rewards. Value functions are visualized to monitor the agent’s learning process.

7. **Saving Model Weights**:
   - The trained model weights are saved for future use, enabling the agent to resume training or deployment without starting from scratch.

## Training and Visualization

The project includes visualization tools (`visualize_train.py`) to draw the value and policy images, providing insights into the agent's learned value function and optimal policy.

## Conclusion

This DQN implementation demonstrates the use of deep reinforcement learning to solve a grid world navigation problem. By leveraging experience replay and neural networks, the agent effectively learns to maximize its cumulative reward through interactions with the environment.
