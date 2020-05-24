"""
File contains Player's abstract class
and the subclass for AI-player, human-player.
"""

import numpy as np
import utils
from typing import Union, List

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


class MonteCarloPlayer(Player):
    def __init__(self, num_simulations=1000):
        super().__init__()
        self._num_simulations = num_simulations

    def peek(self, game):
        self.game = game.copy()

    def play(self):
        move_list = self.game.game_board.get_empty_slots()
        current_piece = self.game.current_piece
        winning_probabilities = []
        for move in move_list:
            pseudo_game = self.game.copy()
            row, col = utils.index2coordinate(move, pseudo_game.game_board.size)
            pseudo_game.put_piece(row, col)
            pseudo_game.switch_player()
            simulation_result = self._simulate(pseudo_game)
            winning_probabilities.append(np.sum([x == current_piece for x in simulation_result]) / self._num_simulations)

        winning_probabilities_each_move = zip(move_list, winning_probabilities)
        return max(winning_probabilities_each_move, key=lambda x: x[1])[0]

    def _simulate(self, game) -> List[int]:
        simulation_result = []
        for _ in range(self._num_simulations):
            game_dup = game.copy()

            # Monte carlo simulation depends on random play
            simulation_result.append(game_dup.start_game(RandomPlayer(), RandomPlayer()))

        return simulation_result


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


class RandomPlayer(Player):
    """
    A player that plays the game randomly
    """

    def __int__(self, game):
        super().__init__()

    def peek(self, game):
        """
        A random player peeks the board for all empty slots

        :param game:
        :return:
        """
        self.game = game.copy()

    def play(self) -> int:
        """
        A random player uniformly randomly choose from all possible moves
        :return: A selected move
        """
        return np.random.choice(self.game.game_board.get_empty_slots())


# AI player engines. It would have two kinds of AI engines: Monte carlo based one and
# the reinforcement learning based one.
class AIEngine(ABC):
    """
    Abstract base class for all sorts of AI engines.
    """

    @abstractmethod
    def train(self):
        """
        Train the AI decision engine

        :return:
        """
        pass

    @abstractmethod
    def decide(self, piece):
        """
        Check the board status and decide where to
        put the current piece

        :return:
        """
        pass


class MonteCarlo(AIEngine):

    def __init__(self, game, num_simulations=1000):
        """
        Initialize a monte carlo simulation engine

        :param move_list:
        :param num_simulations:
        """

        self.num_simulations = num_simulations
        self.game = game

    def train(self):
        """
        Monte carlo method doesn't
        depend on training
        :return:
        """

        return

    def decide(self, piece) -> float:
        """
        The decision engine returns the quality of a move

        :param piece:
        :return:
        """
        simulation_result = self._simulate()
        return np.sum([x == piece for x in simulation_result]) / self.num_simulations

    def _simulate(self) -> List[int]:
        """
        Run one round of monte carlo simulation and return the result

        :return:
        """
        simulation_result = []
        for _ in range(self.num_simulations):
            simulation_result.append(self.game.pla())

        return simulation_result







