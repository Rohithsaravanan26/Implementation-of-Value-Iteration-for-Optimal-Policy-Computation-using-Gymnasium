# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

To implement the Value Iteration algorithm using the Gymnasium FrozenLake-v1 environment to solve a finite Markov Decision Process (MDP). The objective is to calculate the optimal state-value function and determine the optimal policy using the Bellman Optimality Equation, which helps the agent reach the goal state while avoiding hole states.


## Software Requirements
1. Programming Language: Python 3.x
2. IDE: Jupyter Notebook / Google Colab / VS Code
3. Libraries:
   - Gymnasium – To create and manage the FrozenLake environment.
   - NumPy – To perform numerical calculations and array operations.
   - Matplotlib – To visualize the results if required.
Installation command:
pip install gymnasium numpy matplotlib

## Environment Description
FrozenLake is a grid-based environment in which an agent must move from a starting position to a goal position without falling into holes.

The environment consists of a 4 × 4 grid containing four types of states:
Symbol	Description
S	Starting state
F	Frozen / Normal state
H	Hole state
G	Goal state

The environment used in this program is:
```
F F F S
F H F H
F H F F
G F F H
```

- The agent starts at the top-right position.
- The goal is located at the bottom-left position.
- The agent receives a reward of 1 when reaching the goal.
- Falling into a hole ends the episode.
- The environment is slippery, meaning the agent may move in an unintended direction.

## MDP Representation
FrozenLake can be represented as a Markov Decision Process (MDP) using the following components:
1. States (S): There are 16 states in the 4 × 4 grid, numbered from 0 to 15.
2. Actions (A): The agent can perform four actions:
   - 0 – Left
   - 1 – Down
   - 2 – Right
   - 3 – Up
3. Transition Probability (P): Represents the probability of moving from one state to another after performing an action.
4. Reward (R): The agent receives a reward of 1 for reaching the goal and 0 for other transitions.
5. Discount Factor (γ): The discount factor is set to 0.99 to calculate future rewards.
6. Policy (π): Defines the best action to perform in each state to reach the goal.

## Theory

Value Iteration is a dynamic programming algorithm used to find the optimal policy in a Markov Decision Process.

It repeatedly updates the value of each state using the Bellman Optimality Equation until the values converge.

The Bellman Optimality Equation is:
\[
V_{k+1}(s)=\max_a\sum_{s'}P(s'|s,a)
[R(s,a,s')+\gamma V_k(s')]
\]
Where:
- \(V(s)\) – Value of the current state.
- \(P(s'|s,a)\) – Transition probability.
- \(R(s,a,s')\) – Immediate reward.
- \(\gamma\) – Discount factor.
- \(s'\) – Next state.
- \(a\) – Possible action.
The algorithm calculates the maximum expected reward for every state. This process continues until the difference between successive state values becomes smaller than a predefined threshold.

The optimal policy is then obtained by selecting the action with the highest expected value in each state.

## Algorithm
Start the program.



Import the required libraries such as Gymnasium and NumPy.



Create the FrozenLake environment with a 4 × 4 grid.



Define the discount factor γ = 0.99 and convergence threshold θ = 1e-8.



Initialize the state-value function \(V(s)\) with zeros.



For every state, calculate the expected value of all possible actions.



Select the maximum action value using the Bellman Optimality Equation.



Update the state-value function.



Calculate the maximum difference between old and new state values.



Repeat the process until the difference becomes smaller than the convergence threshold.



Compute the optimal policy by selecting the action with the highest expected value in each state.



Display the optimal state-value function, optimal policy, and number of iterations.



Stop the program.




## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "FFFS",
    "FHFH",
    "FHFF",
    "GFFH"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
# write your code here


# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    iteration = 0

    while True:

        delta = 0

        for s in range(n_states):

            action_values = np.zeros(n_actions)

            for a in range(n_actions):

                for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:

                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )

            best_value = np.max(action_values)

            delta = max(delta, abs(best_value - V[s]))

            V[s] = best_value

        iteration += 1

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):

        action_values = np.zeros(n_actions)

        for a in range(n_actions):

            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:

                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )

        policy[s] = np.argmax(action_values)

    return V, policy, iteration

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iterations = value_iteration(env)



# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: ROHITH S")
print("Register Number:  21225240121")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()


```

---

## Output

```text

Name: ROHITH S
Register Number:  21225240121
Value Iteration Completed
Number of Iterations: 191

Optimal State-Value Function:
[[0.8514 0.7835 0.7393 0.7176]
 [0.8772 0.     0.3642 0.    ]
 [0.9296 0.     0.3644 0.1795]
 [0.     0.7736 0.5605 0.    ]]

Optimal Policy:
[['L' 'U' 'U' 'U']
 ['L' 'L' 'L' 'L']
 ['L' 'L' 'R' 'D']
 ['L' 'D' 'L' 'L']]


```

---

## Result
```text
The Value Iteration algorithm was implemented successfully using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed using the Bellman Optimality Equation. The algorithm converged in 191 iterations.
```
---

## Inference
```text
The Value Iteration algorithm successfully determined the optimal state-value function and optimal policy for the FrozenLake environment. By repeatedly applying the Bellman Optimality Equation, the algorithm calculated the maximum expected future reward for each state.

The agent learns the best possible actions to reach the goal state while avoiding holes. Since the environment is slippery, the agent considers transition probabilities when selecting actions. Thus, Value Iteration provides an effective method for finding an optimal policy in a finite Markov Decision Process.

```
---

