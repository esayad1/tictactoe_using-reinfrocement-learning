import tkinter as tk
import tkinter.messagebox
from tictactoe import TicTacToe
from monte_carlo import MonteCarloES, OnPolicyMonteCarloControl, OffPolicyMonteCarloControl
from temporal_difference import Sarsa, QLearning, ExpectedSarsa

class TicTacToeGUI:
    def __init__(self, ai_opponent=None):
        self.game = TicTacToe()
        self.ai_opponent = ai_opponent
        self.current_player = 'X'
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe")
        self.buttons = [[None, None, None] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(
                    self.window,
                    text=" ",
                    command=lambda row=i, col=j: self.make_move(row, col),
                    font=('Arial', 24),
                    width=5,
                    height=2
                )
                self.buttons[i][j].grid(row=i, column=j)

    def make_move(self, row, col):
        if self.game.board[row][col] == ' ':
            self.game.board[row][col] = self.current_player
            self.buttons[row][col]['text'] = self.current_player

        if self.game.check_win():
            self.game_over(f"Player {self.current_player} wins!")
        elif self.game.check_tie():
            self.game_over("The game is a tie!")
        else:
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            if self.current_player == 'O' and self.ai_opponent is not None:
                self.make_ai_move()

    def make_ai_move(self):
        move = self.ai_opponent.make_move(self.game)
        if move is not None:
            self.make_move(*move)

    def game_over(self, result):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]['state'] = tk.DISABLED
        tk.messagebox.showinfo("Game Over", result)
        self.restart_game()

    def restart_game(self):
        self.window.destroy()
        main_menu = MainMenu()
        main_menu.run()

    def run(self):
        self.window.mainloop()


class MainMenu:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe - Main Menu")
        self.title = tk.Label(self.window, text="Tic Tac Toe", font=('Arial', 24))
        self.title.pack()
        self.play_button = tk.Button(self.window, text="Play Against Random Player", command=self.start_game, font=('Arial', 20))
        self.play_button.pack()
        self.play_mc_es_button = tk.Button(self.window, text="Play Against Monte Carlo ES", command=self.start_game_mc_es, font=('Arial', 20))
        self.play_mc_es_button.pack()
        self.play_mc_on_policy_button = tk.Button(self.window, text="Play Against On-policy MC", command=self.start_game_mc_on_policy, font=('Arial', 20))
        self.play_mc_on_policy_button.pack()
        self.play_mc_off_policy_button = tk.Button(self.window, text="Play Against Off-policy MC", command=self.start_game_mc_off_policy, font=('Arial', 20))
        self.play_mc_off_policy_button.pack()
        self.play_sarsa_button = tk.Button(self.window, text="Play Against Sarsa", command=self.start_game_sarsa, font=('Arial', 20))
        self.play_sarsa_button.pack()
        self.play_q_learning_button = tk.Button(self.window, text="Play Against Q-Learning", command=self.start_game_q_learning, font=('Arial', 20))
        self.play_q_learning_button.pack()
        self.play_expected_sarsa_button = tk.Button(self.window, text="Play Against Expected Sarsa", command=self.start_game_expected_sarsa, font=('Arial', 20))
        self.play_expected_sarsa_button.pack()

    def start_game(self):
        self.window.destroy()
        game = TicTacToeGUI()
        game.run()

    def start_game_mc_es(self):
        self.window.destroy()
        mc_es = MonteCarloES('O')
        mc_es.train(10000)  # Train on 10,000 episodes
        game = TicTacToeGUI(mc_es)
        game.run()

    def start_game_mc_on_policy(self):
        self.window.destroy()
        mc_on_policy = OnPolicyMonteCarloControl('O', 0.1)  # Use epsilon = 0.1
        mc_on_policy.train(10000)  # Train on 10,000 episodes
        game = TicTacToeGUI(mc_on_policy)
        game.run()

    def start_game_mc_off_policy(self):
        self.window.destroy()
        mc_off_policy = OffPolicyMonteCarloControl('O')
        mc_off_policy.train(10000)  # Train on 10,000 episodes
        game = TicTacToeGUI(mc_off_policy)
        game.run()

    def start_game_sarsa(self):
        self.window.destroy()
        sarsa = Sarsa('O')
        sarsa.train(10000)  # Train on 10,000 episodes
        game = TicTacToeGUI(sarsa)
        game.run()

    def start_game_q_learning(self):
        self.window.destroy()
        q_learning = QLearning('O')
        q_learning.train(10000)  # Train on 10,000 episodes
        game = TicTacToeGUI(q_learning)
        game.run()

    def start_game_expected_sarsa(self):
        self.window.destroy()
        expected_sarsa = ExpectedSarsa('O')
        expected_sarsa.train(10000)  # Train on 10,000 episodes
        game = TicTacToeGUI(expected_sarsa)
        game.run()

    def run(self):
        self.window.mainloop()


def main():
    main_menu = MainMenu()
    main_menu.run()


if __name__ == "__main__":
    main()