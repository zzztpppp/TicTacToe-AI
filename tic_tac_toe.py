"""
The game of TicTacToe
"""

import numpy as np
import utils
import typing
import players

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
    def size(self):
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
        if x == y or (x == self.size - y - 1):
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

    def clean(self):
        """
        Set the values of the board to be zero
        :return:
        """
        self.board_values = np.zeros((self.size, self.size))

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
    def __init__(self, game_board: TicTacToeBoard, first_player=CROSS):
        self.game_board = game_board
        self.current_piece = first_player  # The piece to put for this move

        self.num_steps_played = 0

        self.game_running = False

    def switch_player(self):
        if self.current_piece == CROSS:
            self.current_piece = CIRCLE
        else:
            self.current_piece = CROSS

    def check_tied(self):
        board_size = self.game_board.size
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

        self.game_board.board_values[x, y] = self.current_piece

    def check_status(self, x: int, y: int) -> typing.Union[int, None]:
        """
        Check if game is over.It may alter
        :param x:
        :param y:
        :param piece:
        :return: 1 if CROSS wins, -1 if CIRCLE wins; 0 if game tied; None if game continues
        """
        if self.check_tied():
            return 0

        # If game is not tied, check if any player wins the game
        winner = self.game_board.check_status(x, y, self.current_piece)
        if winner != 0:
            return winner
        else:
            return None

    def show_board(self):
        """
        Print the game board onto console
        :return:
        """
        print(self.game_board)

    @utils.try_until_success
    def play(self, player: players.Player) -> int:
        """
        Method that allows the player to look at the game  board
        and decide where to put his chess.

        :param player:
        :return:
        """
        # A player first look at the board
        player.peek(self.game_board, self.current_piece)
        position = player.play()
        row, col = utils.index2coordinate(position, self.game_board.size)
        self.put_piece(row, col)

        # Check status after move
        status_after_play = self.check_status(row, col)

        return status_after_play

    def start_game(self, first_player: players.Player, second_player: players.Player) -> int:
        from itertools import cycle
        self.game_running = True
        players_queue = cycle([first_player, second_player])
        status_after_play = 0

        # Player moves one by one
        for player in players_queue:
            if not self.game_running:
                break

            status_after_play = self.play(player)

            self.num_steps_played += 1

            if status_after_play is not None:
                self.game_running = False

            self.switch_player()

        self.show_board()

        return status_after_play


# Unit test
if __name__ == "__main__":
    player_1 = players.HumanPlayer()
    player_2 = players.HumanPlayer()

    game = TicTacToeGame(TicTacToeBoard.create_empty_board())
    print("Game started")

    game_result = game.start_game(player_1, player_2)
    if game_result == CROSS:
        print("Player CROSS wins!")
    elif game_result == CIRCLE:
        print("Player CIRCLE wins!")
    else:
        print("It's a tied game")


