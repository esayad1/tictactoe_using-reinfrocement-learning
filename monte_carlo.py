import numpy as np
from tictactoe import TicTacToe

class MonteCarloES:
    def __init__(self, player):
        self.Q = {}
        self.returns = {}
        self.policy = {}
        self.player = player

    def train(self, episodes):
        for _ in range(episodes):
            game = TicTacToe()
            state_action_pairs = self.play_game(game)
            self.update_Q(state_action_pairs)
        self.update_policy()

    def play_game(self, game):
        state_action_pairs = []
        while True:
            state = str(game.board)
            action = self.get_action(game)
            state_action_pairs.append((state, action))
            result = game.make_move(*action)
            if result is not None:
                return_value = 1 if result.startswith(f"Player {self.player} wins") else -1 if result.startswith("Player O wins") else 0
                return [(state, action, return_value) for state, action in state_action_pairs]

    def get_action(self, game):
        state = str(game.board)
        if state not in self.policy:
            return game.get_random_move()
        else:
            return self.policy[state]

    def update_Q(self, state_action_returns):
        for state, action, return_value in state_action_returns:
            if (state, action) not in self.returns:
                self.returns[(state, action)] = []
            self.returns[(state, action)].append(return_value)
            if state not in self.Q:
                self.Q[state] = {}
            self.Q[state][action] = np.mean(self.returns[(state, action)])

    def update_policy(self):
        for state in self.Q:
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def make_move(self, game):
        return self.get_action(game)


class OnPolicyMonteCarloControl(MonteCarloES):
    def __init__(self, player, epsilon):
        super().__init__(player)
        self.epsilon = epsilon

    def get_action(self, game):
        state = str(game.board)
        if np.random.random() < self.epsilon or state not in self.policy:
            return game.get_random_move()
        else:
            return self.policy[state]


class OffPolicyMonteCarloControl(MonteCarloES):
    def __init__(self, player):
        super().__init__(player)
        self.C = {}

    def play_game(self, game):
        state_action_reward_tuples = []
        while True:
            state = str(game.board)
            action = self.get_random_move(game)
            result = game.make_move(*action)
            if result is not None:
                reward = 1 if result.startswith(f"Player {self.player} wins") else -1 if result.startswith("Player O wins") else 0
                state_action_reward_tuples.append((state, action, reward))
                return state_action_reward_tuples

    def get_random_move(self, game):
        return game.get_random_move()

    def update_Q(self, state_action_reward_tuples):
        G = 0
        W = 1
        for state, action, reward in reversed(state_action_reward_tuples):
            G = reward + G
            if (state, action) not in self.C:
                self.C[(state, action)] = 0
            self.C[(state, action)] += W
            if state not in self.Q:
                self.Q[state] = {}
            if action not in self.Q[state]:
                self.Q[state][action] = 0
            self.Q[state][action] += W / self.C[(state, action)] * (G - self.Q[state][action])
            if action != self.policy.get(state, None):
                break
            W /= 1 / 9  # Probability of selecting action under behavior policy

    def update_policy(self):
        for state in self.Q:
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def make_move(self, game):
        state = str(game.board)
        return self.policy.get(state, self.get_random_move(game))
