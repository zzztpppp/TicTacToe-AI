"""
The game of TicTacToe
"""

import numpy as np
import utils


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

        self.board = board

        # Our board always has length == width
        self.board_width, self.board_length = board.shape

    def put_0(self, x: int, y: int):
        """
        Put the 0 chess piece at the given position of the board
        :param x: Horizontal index
        :param y: Vertical index
        :return:
        """
        self.board[x, y] = 0

    def check_status(self, x: int, y: int, piece: int) -> int:
        """
        Check whether game is over after put a $piece at the
        position
        :return: -1 if game continues, 1 game ends and piece 1 winds, 0 game ends and piece 0 wins
        """
        winnable = piece * self.continuous_pieces_to_win

        # Horizontally continuous pieces
        if np.sum(self.board[x, :]) == winnable:
            return piece

        # Vertically continuous pieces
        if np.sum(self.board[:, y]) == winnable:
            return piece

        # Diagonally continuous pieces
        if x == y:
            upper_diagonal_values = np.sum([self.board[self.board_length - 1 - i, i] for i in range(self.board_width)])
            lower_diagonal_values = np.sum([self.board[i, i] for i in range(self.board_width)])
            if upper_diagonal_values == winnable or lower_diagonal_values == winnable:
                return piece

        # Game is not over
        return -1

    def __str__(self):
        string_representation = utils.get_string_grid(self.board_width, self.board_length, self.board)
        return string_representation

    
