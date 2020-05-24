"""
File contains Player's abstract class
and the subclass for AI-player, human-player.
"""

from typing import Union

from abc import ABC, abstractmethod


class Player(ABC):
    """
    The abstract base class for all
    kinds of players
    """

    def __init__(self):
        self._name = None  # Name of this player
        self._game = None

    @property
    def name(self) -> Union[None, str]:
        """
        Get the name of this player
        :return: Name of this player, None if player is not named
        """
        return self._name

    @property
    def game(self):
        return self._game

    @game.setter
    def game(self, game):
        self._game = game

    @name.setter
    def name(self, name: str):
        """
        Name this player

        :param name:
        :return:
        """
        self._name = name

    @abstractmethod
    def peek(self, game):
        """
        Peek the game board, pre-process necessary information
        for play.

        :param game: The game board
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

    def peek(self, board, piece_type: int):
        pass

    def play(self) -> int:
        pass


class HumanPlayer(Player):

    def __init__(self):
        super().__init__()

    def peek(self, game):
        """
        For human players to peek the game board, just simply print the
        board onto console

        :param game:
        :return:
        """
        print(game.game_board)

    def play(self) -> int:
        """
        For human player to play, just simply take the input from console
        and put the specified chess onto the input position
        :return:
        """

        chess_position = input("Please input where you want to put the chess at")
        return int(chess_position)
