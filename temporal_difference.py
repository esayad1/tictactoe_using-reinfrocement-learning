import numpy as np
from tictactoe import TicTacToe

class Sarsa:
    def __init__(self, player, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.Q = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.player = player  # Player this AI represents ('X' or 'O')

    def train(self, episodes):
        for _ in range(episodes):
            game = TicTacToe()
            self.play_game(game)

    def play_game(self, game):
        state = self.initialize_state(game)
        action = self.get_action(game)
        while True:
            result = game.make_move(*action)
            next_state = self.initialize_state(game)
            reward = -1 if result and result.startswith(f"Player {self.player} wins") else 0
            next_action = self.get_action(game)
            self.Q[(state, action)] = self.Q.get((state, action), 0) + \
                self.alpha * (reward + self.gamma * self.Q.get((next_state, next_action), 0) - self.Q.get((state, action), 0))
            state, action = next_state, next_action
            if result is not None:
                break

    def get_action(self, game):
        state = self.initialize_state(game)
        if np.random.random() < self.epsilon:
            return game.get_random_move()  # Exploration
        else:
            if state not in self.Q:
                return game.get_random_move()
            else:
                return max(self.Q[state], key=self.Q[state].get)  # Exploitation

    def initialize_state(self, game):
        return str(game.board)

    def make_move(self, game):
        return self.get_action(game)


class QLearning(Sarsa):
    def play_game(self, game):
        state = self.initialize_state(game)
        while True:
            action = self.get_action(game)
            result = game.make_move(*action)
            next_state = self.initialize_state(game)
            reward = -1 if result and result.startswith(f"Player {self.player} wins") else 0
            max_next_Q_value = max(self.Q.get(next_state, {}).values(), default=0)
            self.Q[(state, action)] = self.Q.get((state, action), 0) + \
                self.alpha * (reward + self.gamma * max_next_Q_value - self.Q.get((state, action), 0))
            state = next_state
            if result is not None:
                break


class ExpectedSarsa(Sarsa):
    def play_game(self, game):
        state = self.initialize_state(game)
        while True:
            action = self.get_action(game)
            result = game.make_move(*action)
            next_state = self.initialize_state(game)
            reward = -1 if result and result.startswith(f"Player {self.player} wins") else 0
            next_Q_values = self.Q.get(next_state, {}).values()
            expected_next_Q_value = sum(next_Q_values) / len(next_Q_values) if next_Q_values else 0
            self.Q[(state, action)] = self.Q.get((state, action), 0) + \
                self.alpha * (reward + self.gamma * expected_next_Q_value - self.Q.get((state, action), 0))
            state = next_state
            if result is not None:
                break
