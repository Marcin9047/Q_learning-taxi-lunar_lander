class agent:
    def __init__(self, learning_rate, discount_factor):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def change_q_value(self, time, new_time, action, reward1):
        max_posible_next_q = max(self.q_table[new_time])
        value = (1 - self.learning_rate) * self.q_table[time][action]
        value += self.learning_rate * (
            (self.discount_factor * max_posible_next_q) + reward1
        )
        self.q_table[time][action] = value

    def find_best(self, time):
        maxnum = 0
        maxvalue = self.q_table[time][0]
        for num, value in enumerate(self.q_table[time]):
            if maxvalue < value:
                maxvalue = value
                maxnum = num
        return maxnum
