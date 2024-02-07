import matplotlib.pyplot as plt
from agent import agent
import numpy as np
from lab5_qlearn import Q_learning, Training_Results
from exploration_strategies import (
    greedy_strategy,
    changing_greedy_strategy,
    boltzman_strategy,
)


agent1 = agent(0.1, 0.3)


# learning_obj.test(2, 30)


agent1 = agent(0.1, 0.3)
agent2 = agent(0.5, 0.3)
agent3 = agent(0.8, 0.3)


def average_plot(
    size_average, learning_rate, discount_factor, strategy, max_num, opis=None
):
    average_result = []
    for i in range(size_average):
        agent1 = agent(learning_rate, discount_factor)
        learning_obj = Q_learning(agent1)
        results = learning_obj.learn(strategy, max_num)
        if len(average_result) != 0:
            average_result = [
                x + y for x, y in zip(average_result, results.total_reward)
            ]
        else:
            average_result = results.total_reward
    average_result = [x / size_average for x in average_result]
    if not opis:
        label1 = (
            f"strategy = {strategy.name}, learning rate = {learning_rate}, discount factor = {discount_factor}",
        )
    else:
        label1 = opis
    plt.plot([int(x) for x in range(len(average_result))], average_result, label=label1)


# learning_obj = Q_learning(agent2)
# results = learning_obj.learn(strat1, 30)
# plt.plot(
#     [x for x in range(len(results.iteration_vals))],
#     results.iteration_vals,
#     label="learning rate = 0,5",
# )

# learning_obj = Q_learning(agent3)
# results = learning_obj.learn(strat1, 30)

# plt.plot(
#     [x for x in range(len(results.iteration_vals))],
#     results.iteration_vals,
#     label="learning rate = 0,8",
# )
strat1 = changing_greedy_strategy(0.4, 0.05, 30)
opis1 = "epsilon = 0.4, epsilon diff = 0.05, changing step = 10"
strat2 = changing_greedy_strategy(0.1, 0.05, 30)
opis2 = "epsilon = 0.1, epsilon diff = 0.05, changing step = 30"

strat3 = boltzman_strategy(3)
opis3 = "T value = 3"

strat4 = boltzman_strategy(3)
opis4 = "T value = 9"

strat5 = greedy_strategy(0.4)
opis5 = "epsilon = 0.4"

strat6 = greedy_strategy(0.1)
opis6 = "epsilon = 0.1"


average_plot(50, 0.5, 0.1, strat1, 300)
average_plot(50, 0.5, 0.3, strat1, 300)
average_plot(50, 0.5, 0.5, strat1, 300)
average_plot(50, 0.5, 0.7, strat1, 300)
average_plot(50, 0.5, 0.9, strat1, 300)
# average_plot(50, 0.3, 0.3, strat5, 300, opis6)

plt.title("Taxi training > average of 50")

plt.xlabel("Iteration of training")
plt.ylabel("Average total value")

# plt.yscale("log")
plt.legend()
plt.show()
