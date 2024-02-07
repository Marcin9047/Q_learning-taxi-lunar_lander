import math
from random import choices
import random


class boltzman_strategy:
    def __init__(self, T_val):
        self.t_val = T_val
        self.name = "boltzman"

    def run_strat(self, env, agent1, state):
        possibilities = []
        sum = 0
        for q_val in agent1.q_table[str(state)]:
            value = math.exp(q_val / self.t_val)
            possibilities.append(value)
            sum += value
        possibilities = [x / sum for x in possibilities]
        actions = range(len(agent1.q_table[str(state)]))
        return choices(actions, possibilities)[0]


class changing_greedy_strategy:
    def __init__(self, epsilon, epsilon_diff, diff_step):
        self.epsilon = epsilon
        self.diff = epsilon_diff
        self.counter = 0
        self.step_size = diff_step
        self.name = "changing_greedy"

    def run_strat(self, env, agent1, state):
        self.counter += 1
        decider = random.uniform(0, 1)
        if decider < self.epsilon:
            action = env.action_space.sample()
        else:
            action = agent1.find_best(str(state))
        if self.counter == self.step_size and self.epsilon - self.diff > 0.01:
            self.counter = 0
            self.epsilon = self.epsilon - self.diff
        return action


class greedy_strategy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.name = "greedy"

    def run_strat(self, env, agent1, state):
        decider = random.uniform(0, 1)
        if decider < self.epsilon:
            action = env.action_space.sample()
        else:
            action = agent1.find_best(str(state))
        return action
