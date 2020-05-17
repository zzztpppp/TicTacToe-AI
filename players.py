"""
File contains Player's abstract class
and the subclass for AI-player, human-player.
"""

from typing import Union

from abc import ABC, abstractmethod
from tic_tac_toe import TicTacToeBoard


class Player(ABC):
    """
    The abstract base class for all
    kinds of players
    """

    def __init__(self):
        self._name = None  # Name of this player

    @property
    def name(self) -> Union[None, str]:
        """
        Get the name of this player
        :return: Name of this player, None if player is not named
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        Name this player

        :param name:
        :return:
        """
        self._name = name

    @abstractmethod
    def peek(self, board: TicTacToeBoard, piece_type: int):
        """
        Peek the game board, pre-process necessary information
        for play.

        :param piece_type: Which piece does the player need to put
        :param board: The game board
        :return:
        """
        pass

    @abstractmethod
    def play(self) -> int:
        """
        Decide where to put my piece
        :return: Position to put my piece
        """
        pass


class AIPlayer(Player):

    def __init__(self):
        super().__init__()
        pass

    def peek(self, board: TicTacToeBoard, piece_type: int):
        pass

    def play(self) -> int:
        pass


class HumanPlayer(Player):

    def __init__(self):
        super().__init__()
        pass

    def peek(self, board: TicTacToeBoard, piece_type: int):
        """
        For human players to peek the game board, just simply print the
        board onto console

        :param board:
        :param piece_type:
        :return:
        """
        print(board)

    def play(self) -> int:
        """
        For human player to play, just simply take the input from console
        and put the specified chess onto the input position
        :return:
        """

        chess_position = input("Please input where you want to put the chess at")
        return int(chess_position)
