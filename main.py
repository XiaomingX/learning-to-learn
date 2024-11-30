# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

# Constants for the grid size and special states
BOARD_ROWS = 5
BOARD_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1, 0), (3, 1), (4, 2), (1, 3)]

class State:
    """
    This class represents the state of the agent on the grid.
    It handles the reward mechanism, end conditions, and next state transitions.
    """
    def __init__(self, state=START):
        self.state = state  # Current position of the agent on the grid
        self.is_end = False  # Flag to indicate if the current state is an end state (win or loss)

    def get_reward(self):
        """
        Returns the reward for the current state:
        -5 for falling into a hole, +1 for winning, -1 for other states.
        """
        if self.state in HOLE_STATE:
            return -5
        if self.state == WIN_STATE:
            return 1
        return -1

    def check_if_end(self):
        """
        Check if the current state is an end state (win or loss).
        """
        if self.state == WIN_STATE or self.state in HOLE_STATE:
            self.is_end = True

    def next_position(self, action):
        """
        Given an action (up, down, left, right), return the next position of the agent.
        The action should be one of [0, 1, 2, 3], representing [up, down, left, right].
        """
        if action == 0:  # Up
            next_state = (self.state[0] - 1, self.state[1])
        elif action == 1:  # Down
            next_state = (self.state[0] + 1, self.state[1])
        elif action == 2:  # Left
            next_state = (self.state[0], self.state[1] - 1)
        else:  # Right
            next_state = (self.state[0], self.state[1] + 1)

        # Ensure the next state is within the grid boundaries
        if 0 <= next_state[0] < BOARD_ROWS and 0 <= next_state[1] < BOARD_COLS:
            return next_state
        return self.state  # Return current state if next state is out of bounds

class Agent:
    """
    This class represents an agent that learns to navigate the grid using Q-learning.
    The agent uses an epsilon-greedy policy to choose actions and update its Q-values.
    """
    def __init__(self):
        self.state = State()  # Initial state of the agent
        self.actions = [0, 1, 2, 3]  # Possible actions: [up, down, left, right]
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Epsilon value for epsilon-greedy policy

        # Initialize Q-table and new Q-table (for updating)
        self.Q = {}
        self.new_Q = {}

        # Initialize rewards tracking
        self.total_rewards = 0
        self.reward_history = []

        # Initialize all Q-values to 0
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for action in self.actions:
                    self.Q[(i, j, action)] = 0
                    self.new_Q[(i, j, action)] = 0

    def choose_action(self):
        """
        Choose an action based on the epsilon-greedy policy.
        - With probability (1 - epsilon), choose the action with the highest Q-value.
        - With probability epsilon, choose a random action.
        """
        rnd = random.random()

        if rnd > self.epsilon:  # Exploit: Choose the action with the highest Q-value
            best_action = None
            max_q_value = -float('inf')
            for action in self.actions:
                i, j = self.state.state
                q_value = self.Q[(i, j, action)]
                if q_value > max_q_value:
                    best_action = action
                    max_q_value = q_value
            return self.state.next_position(best_action), best_action
        else:  # Explore: Choose a random action
            action = random.choice(self.actions)
            return self.state.next_position(action), action

    def update_q_values(self, action, reward, next_state):
        """
        Update the Q-value for the current state-action pair using the Q-learning formula:
        Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max_a Q(s', a'))
        """
        i, j = self.state.state
        next_i, next_j = next_state

        # Find the maximum Q-value for the next state
        max_q_next = max(self.Q[(next_i, next_j, a)] for a in self.actions)

        # Update the Q-value for the current state-action pair
        self.new_Q[(i, j, action)] = round((1 - self.alpha) * self.Q[(i, j, action)] +
                                           self.alpha * (reward + self.gamma * max_q_next), 3)

    def q_learning(self, episodes):
        """
        Perform Q-learning for a specified number of episodes.
        In each episode, the agent explores the environment and updates its Q-values.
        """
        for episode in range(episodes):
            self.state = State()  # Reset to the start state at the beginning of each episode
            self.state.is_end = False
            self.total_rewards = 0  # Reset the reward tracker for each episode

            while not self.state.is_end:
                # Choose an action based on the current policy
                next_state, action = self.choose_action()

                # Get the reward for the current state
                reward = self.state.get_reward()
                self.total_rewards += reward

                # Update the Q-values based on the reward and the next state
                self.update_q_values(action, reward, next_state)

                # Move to the next state
                self.state = State(next_state)
                self.state.check_if_end()

            # Save the total rewards for this episode
            self.reward_history.append(self.total_rewards)

            # Copy new Q-values to the main Q-table
            self.Q = self.new_Q.copy()

    def plot_rewards(self):
        """
        Plot the cumulative rewards over episodes.
        """
        plt.plot(self.reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Rewards vs Episodes')
        plt.show()

    def display_q_values(self):
        """
        Display the Q-values for each state in the grid.
        """
        for i in range(BOARD_ROWS):
            print('-----------------------------------------------')
            row_str = '| '
            for j in range(BOARD_COLS):
                max_q_value = max(self.Q[(i, j, a)] for a in self.actions)
                row_str += f"{max_q_value:.3f}".ljust(6) + ' | '
            print(row_str)
        print('-----------------------------------------------')

if __name__ == "__main__":
    # Create an agent and perform Q-learning for 10,000 episodes
    agent = Agent()
    agent.q_learning(10000)

    # Plot the reward history and display the learned Q-values
    agent.plot_rewards()
    agent.display_q_values()
