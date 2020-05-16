"""
The game of TicTacToe
"""

import numpy as np
import utils

CIRCLE = 1
CROSS = -1


class TicTacToeBoard:
    """
    The board the holds the tic-tac-toe game
    """
    continuous_pieces_to_win = 3

    def __init__(self, board: np.ndarray):
        """
        Initailize  a board
        :param board: A 3 by numpy ndarray, with elements 0 and 1
        """

        self.board_values = board

        # Our board always has length == width
        self.board_width, self.board_length = board.shape

    @property
    def board_size(self):
        return self.board_width

    def put_circle(self, x: int, y: int):
        """
        Put the 1 chess piece at the given position of the board
        :param x: Horizontal index
        :param y: Vertical index
        :return:
        """
        self.board_values[x, y] = CIRCLE

    def put_cross(self, x: int, y: int):
        """
        Put chess piece -1 at the given position of the board
        :param x: Horizontal index
        :param y: Vertical index
        :return:
        """
        self.board_values[x, y] = CROSS

    def check_status(self, x: int, y: int, piece: int) -> int:
        """
        Check whether game is over after put a $piece at the
        position
        :return: -1 if game continues, 1 game ends and piece 1 winds, 0 game ends and piece 0 wins
        """
        winnable = piece * self.continuous_pieces_to_win

        # Horizontally continuous pieces
        if np.sum(self.board_values[x, :]) == winnable:
            return piece

        # Vertically continuous pieces
        if np.sum(self.board_values[:, y]) == winnable:
            return piece

        # Diagonally continuous pieces
        if x == y:
            upper_diagonal_values = np.sum([self.board_values[self.board_length - 1 - i, i] for i in range(self.board_width)])
            lower_diagonal_values = np.sum([self.board_values[i, i] for i in range(self.board_width)])
            if upper_diagonal_values == winnable or lower_diagonal_values == winnable:
                return piece

        # Game is not over
        return 0


    def copy(self):
        """
        Copy the board
        :return:  A new instance of the board
        """
        data = self.board_values.copy()
        return type(self)(data)

    @classmethod
    def create_empty_board(cls, size=3):
        board_value = np.array([[0]*3 for _ in range(size)])
        return cls(board_value)

    def __str__(self):
        string_representation = utils.get_string_grid(self.board_width, self.board_length, self.board_values)
        return string_representation


class TicTacToeGame:
    """
    The tic tac toe game class, determining who is the next
    player and the status of a game
    """
    def __init__(self, game_board: TicTacToeBoard, first_player=1):
        self.game_board = game_board
        self.current_to_play = first_player

        self.num_steps_played = 0

        self.game_running = False

    def next_player(self):
        if self.current_to_play == CROSS:
            return CIRCLE

        return CROSS

    def check_tied(self):
        board_size = self.game_board.board_size
        if board_size ** 2 == self.num_steps_played:
            return True

        return False

    def put_piece(self, x: int, y: int):
        """
        Put the piece of current player at the given position

        :param x: Row index of the chess
        :param y: Column index of the chess
        :return:
        """
        if self.game_board.board_values[x, y] != 0:
            raise ValueError("You must select a empty slot!")

        self.game_board.board_values[x, y] = self.current_to_play

    def start_game(self):
        print("Game started!")
        self.game_running = True
        while self.game_running:
            pass




