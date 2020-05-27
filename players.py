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
            prior_player = DumbPlayer(move)
            status = pseudo_game.play(prior_player)

            # If game ends at first move, then no need to simulate
            if status is not None:
                simulation_result = [status]
            else:
                pseudo_game.switch_player()
                simulation_result = self._simulate(pseudo_game)

            winning_probabilities.append(np.sum([x == current_piece for x in simulation_result]) / len(simulation_result))

        winning_probabilities_each_move = zip(move_list, winning_probabilities)

        # If no moves go into simulation, return the last move in move_list
        return max(winning_probabilities_each_move, key=lambda x: x[1])[0] or move_list[-1]

    def _simulate(self, game) -> List[int]:
        simulation_result = []
        # TODO: Consider a shrinking number of simulations with shrinking number of empty slots for better performance
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


class DumbPlayer(Player):
    """
    A player that alawys play one given position
    """
    def __init__(self, pos):
        super().__init__()
        self.position = pos

    def peek(self, game):
        return

    def play(self):
        return self.position



