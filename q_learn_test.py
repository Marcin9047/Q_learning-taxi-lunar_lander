import matplotlib.pyplot as plt
from agent import agent
from test_body import Q_learning
from exploration_strategies import (
    greedy_strategy,
    changing_greedy_strategy,
    boltzman_strategy,
)


agent1 = agent(0.1, 0.3)
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


strat1 = changing_greedy_strategy(0.4, 0.05, 30)
opis1 = (
    "changing_greedy_strategy: epsilon = 0.4, epsilon diff = 0.05, changing step = 10"
)
strat2 = changing_greedy_strategy(0.1, 0.05, 30)
opis2 = (
    "changing_greedy_strategy: epsilon = 0.1, epsilon diff = 0.05, changing step = 30"
)

strat3 = boltzman_strategy(3)
opis3 = "boltzman_strategy: T value = 3"

strat4 = boltzman_strategy(3)
opis4 = "boltzman_strategy: T value = 9"

strat5 = greedy_strategy(0.4)
opis5 = "greedy_strategy: epsilon = 0.4"

strat6 = greedy_strategy(0.1)
opis6 = "greedy_strategy: epsilon = 0.1"


average_plot(20, 0.5, 0.3, strat1, 300, opis1)
average_plot(20, 0.5, 0.3, strat2, 300, opis2)
average_plot(20, 0.5, 0.3, strat3, 300, opis3)
average_plot(20, 0.5, 0.3, strat4, 300, opis4)
average_plot(20, 0.5, 0.3, strat5, 300, opis5)
average_plot(20, 0.5, 0.3, strat6, 300, opis6)
plt.title("Taxi training > average of 20")
plt.xlabel("Iteration of training")
plt.ylabel("Average total value")
plt.legend()
plt.show()
