import random
class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            if self.check_win():
                return f'Player {self.current_player} wins!'
            elif self.check_tie():
                return 'The game is a tie!'
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
        return None

    def check_win(self):
        for player in ['X', 'O']:  # Check for both 'X' and 'O' as winners
            for i in range(3):
                if self.board[i][0] == self.board[i][1] == self.board[i][2] == player:
                    return True
                if self.board[0][i] == self.board[1][i] == self.board[2][i] == player:
                    return True
            if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
                return True
            if self.board[0][2] == self.board[1][1] == self.board[2][0] == player:
                return True
        return False

    def check_tie(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def get_random_move(self):
        available_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    available_moves.append((i, j))
        if available_moves:
            return random.choice(available_moves)
        return None
