import pytest
import numpy as np
from tic_tac_toe import TicTacToeBoard


class TestGameBoard:
    data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    board = TicTacToeBoard(data)

    data2 = np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]])
    board2 = TicTacToeBoard(data2)

    def test_put(self):
        self.board.put_1(0, 1)
        assert self.board.board[0, 1] == 1

        self.board.put_neg_1(1, 1)
        assert self.board.board[1, 1] == -1

    def test_copy(self):
        dup_board = self.board.copy()

        # Test two board are different instance
        assert dup_board is not self.board

        # Test two board have the same content
        assert np.all(self.board.board == dup_board.board)

    def test_check_status(self):
        assert self.board2.check_status(0, 2, 1) == 1
        assert self.board2.check_status(2, 2, -1) == -1
        assert self.board2.check_status(1, 1, 1) == 0

