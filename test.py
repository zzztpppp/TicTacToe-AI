import pytest
import numpy as np
import utils


class TestGameBoard:
    import tic_tac_toe
    data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    board = tic_tac_toe.TicTacToeBoard(data)

    data2 = np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]])
    board2 = tic_tac_toe.TicTacToeBoard(data2)

    def test_put(self):
        self.board.put_circle(0, 1)
        assert self.board.board_values[0, 1] == 1

        self.board.put_cross(1, 1)
        assert self.board.board_values[1, 1] == -1

    def test_copy(self):
        dup_board = self.board.copy()

        # Test two board are different instance
        assert dup_board is not self.board

        # Test two board have the same content
        assert np.all(self.board.board_values == dup_board.board_values)

    def test_check_status(self):
        assert self.board2.check_status(0, 2, 1) == 1
        assert self.board2.check_status(2, 2, -1) == -1
        assert self.board2.check_status(1, 1, 1) == 0


class TestUtil:

    def test_index2coordinate(self):
        assert utils.index2coordinate(8, 3) == (2, 2)
        assert utils.index2coordinate(0, 3) == (0, 0)
        assert utils.index2coordinate(3, 3) == (1, 0)


# class Test:
#     import tic_tac_toe
#     def test_check_tie(self):
