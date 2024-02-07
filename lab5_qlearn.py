import gymnasium as gym
from agent import agent
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Training_Results:
    total_reward: int
    iteration_vals: int


class Q_learning:
    def __init__(self, agent1):
        self.agent = agent1

    def learn(self, strategy, max_iter):
        agent1 = self.agent
        env = gym.make("Taxi-v3").env
        size = env.action_space.n
        total_reward = []
        iteration_vals = []
        for j in range(max_iter):
            one_step_reward = 0
            terminated = False
            state = env.reset(seed=42)[0]
            if isinstance(state, Iterable):
                state = [x // 0.05 / 10 for x in state]
            if str(state) not in agent1.q_table.keys():
                agent1.q_table[str(state)] = np.zeros(size)
            i = 0
            for z in range(30):
                i += 1
                action = strategy.run_strat(env, agent1, state)
                new_state, reward, terminated, _, _ = env.step(action)
                if isinstance(new_state, Iterable):
                    new_state = [x // 0.05 / 10 for x in new_state]
                if str(new_state) not in agent1.q_table.keys():
                    agent1.q_table[str(new_state)] = np.zeros(size)
                one_step_reward += reward
                agent1.change_q_value(str(state), str(new_state), action, reward)
                state = new_state

            total_reward.append(one_step_reward)
            iteration_vals.append(i)
        env.close()
        return Training_Results(total_reward, iteration_vals)

    def test(self, num_of_times, one_try_size):
        agent1 = self.agent
        env = gym.make("Taxi-v3", render_mode="human").env
        size = env.action_space.n
        for j in range(num_of_times):
            state = env.reset(seed=42)[0]
            terminated = False
            for i in range(one_try_size):
                if isinstance(state, Iterable):
                    state = [x // 0.05 / 10 for x in state]
                if str(state) not in agent1.q_table.keys():
                    agent1.q_table[str(state)] = np.zeros(size)
                action = agent1.find_best(str(state))
                new_state, reward, terminated, _, _ = env.step(action)
                state = new_state
        env.close()
