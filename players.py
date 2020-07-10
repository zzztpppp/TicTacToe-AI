"""
File contains Player's abstract class
and the subclass for AI-player, human-player.
"""

import os

import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import in_game_types as t
import torch.optim as optim

import exceptions
import logging

from typing import Union, List
from itertools import cycle

from abc import ABC, abstractmethod

from config  import game_config, ai_config


# Where we store the pre_trained dqn
DQN_DIRECTORY = os.path.join('.', 'dqn.pkl')


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

            simulation_result = [utils.translate_game_status(current_piece, x) for x in simulation_result]

            winning_probabilities.append(
                np.sum(simulation_result) / len(simulation_result)
            )

        winning_probabilities_each_move = list(zip(move_list, winning_probabilities))
        print(winning_probabilities_each_move)

        # If no moves go into simulation, return the last move in move_list
        return max(winning_probabilities_each_move, key=lambda x: x[1])[0]

    def _simulate(self, game) -> List[int]:
        simulation_result = []
        # TODO: Consider a shrinking number of simulations w.r.t number of empty slots for better performance
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


class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        flat_x = torch.flatten(x)
        state = F.relu(self.state_encoder(flat_x))
        return self.state2action(state)


class RfPlayer(Player):
    """
    An AI player based on reinforcement learning method
    """

    def __init__(self, dqn: DQN, epsilon):
        super().__init__()
        self.dqn = dqn
        self.epsilon = epsilon
        self.prev_move = None   # Previous move. For model training

    def peek(self, game):
        self.game = game

    def play(self):
        game_state = torch.tensor(self.get_game_states(), dtype=torch.float)
        q_values = self.dqn(game_state)
        move = self.epsilon_greedy(q_values, self.game.game_board.get_empty_slots())
        self.prev_move = move
        return move

    def get_game_states(self):
        """
        Get the current game status according to current game player
        :return: A numpy array with values True standing for self
        and False for opponent.i.e. its a relative state
        """
        game_values = self.game.get_board_values()
        current_player = self.game.current_piece

        return game_values * current_player

    def epsilon_greedy(self, q_values, possible_exploring_positions):
        """
        With probability epsilon exploring  and (1-epsilon) exploiting

        :param possible_exploring_positions: Positions that can be randomly explored
        :param q_values: Q values for each position
        """
        is_exploring = np.random.random() <= self.epsilon
        return np.random.choice(possible_exploring_positions) if is_exploring else q_values.argmax()


class RfTrainer:
    """
    Takes the dqn and train it
    """

    def __init__(
            self, dqn: DQN,
            n_episodes: int,
            alpha=0.01,
            gamma=0.5,
            exploring_rate=0.5,
    ):
        """
        Construct a new trainer

        :param dqn: The q-function network tobe trained
        :param n_episodes: Number of  rounds to train
        :param alpha: Learning rate
        :param gamma: Discount factor when calculating reward
        """
        self.dqn = dqn
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.exploring_rate = exploring_rate
        self.optimizer = optim.SGD(self.dqn.parameters(), lr=self.alpha)

    def train(self):
        episodes_trained = 0
        from tic_tac_toe import TicTacToeGame,TicTacToeBoard

        while episodes_trained < self.n_episodes:

            new_game = TicTacToeGame(TicTacToeBoard.create_empty_board(game_config.BOARD_SIZE))
            self.run_episode(new_game)

            episodes_trained += 1

        return self.dqn

    # TODO: Maybe initialize a game instance inside the method?
    def run_episode(self, game):
        """
        Run one round of training
        :return:
        """
        # Create two AI players who share the same dqn
        player_1 = RfPlayer(self.dqn, self.exploring_rate)
        player_2 = RfPlayer(self.dqn, self.exploring_rate)
        player_1.game, player_2.game = game, game
        game.game_running = True

        players_queue = cycle([player_1, player_2])
        for player in players_queue:
            if not game.game_running:
                break

            current_state = torch.tensor(player.get_game_states(), dtype=torch.float)
            # Basically we goes through the same procedure
            # as in the game.play(). But since we need the internal state
            # in game.play() and don't want the exception being ignored,
            # we go to the ugly way and replicate the code.
            # This is a bad example of programming.
            player.peek(game)
            position = player.play()
            row, col = utils.index2coordinate(position, game.game_board.size)
            try:
                game.put_piece(row, col)
                status_after_play = game.check_status(row, col)
            except exceptions.NonEmptySlotError as e:
                logging.warning(e)
                # Special code for invalid move
                status_after_play = -10

            reward = self._get_reward(status_after_play, game)

            if (status_after_play is not None) and (status_after_play != -10):
                game.game_running = False

            # Take the not operation since next state is pivoted on the opponent
            next_state = torch.tensor(-player.get_game_states(), dtype=torch.float)
            move = player.prev_move

            # Update q-function
            self._q_learning(reward, current_state, move, next_state, game.game_running)

            # If current player chooses a invalid position. Then let him keep trying.
            if status_after_play != -10:
                game.switch_player()

        # Return the dqn trained.
        return self.dqn

    def _q_learning(self, reward, current_state, move, next_state, game_running):
        """
        Update q_function according to the reward gained and the move invoked.
        """
        with torch.no_grad():
            q_next = self.dqn(next_state)

        qmax_next = q_next.max()

        q_current_old = self.dqn(current_state)[move]

        # Since next state denotes the q value of the opponent, then less the better.
        # TODO: utilize a policy function to determine the potential consequence of current move
        q_current_new = torch.tensor(float(reward)) - (self.gamma * qmax_next if game_running else 0)

        loss = F.mse_loss(q_current_old, q_current_new)
        print(f"Current loss is: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def _get_reward(status_after_play, game):

        if status_after_play is None:
            return t.PLAIN_REWARD

        # The player wins the game
        if status_after_play == game.current_piece:
            return t.WINNING_REWARD
        # The game is a tie
        elif status_after_play == 0:
            return t.DRAWING_REWARD

        # Special code for invalid move
        elif status_after_play == -10:
            return t.INVALID_MOVE_REWARD
        # Losing the game
        else:
            return t.LOSING_REWARD


def get_dqn(from_archive=ai_config.LOAD_DQN_ARCHIVE):
    if from_archive:
        import pickle
        with open(DQN_DIRECTORY, 'rb') as f:
            dqn = pickle.load(f)
    else:
        dqn = DQN(game_config.BOARD_SIZE**2, game_config.BOARD_SIZE**2)
        dqn = RfTrainer(dqn, n_episodes=10000)

    return dqn


if __name__ == '__main__':

    pre_trained_dqn = DQN(game_config.BOARD_SIZE**2, game_config.BOARD_SIZE**2)

    dqn_trainer = RfTrainer(pre_trained_dqn, n_episodes=50000)

    pre_trained_dqn = dqn_trainer.train()

    # Dump the pre-trained dqn as pickle file.
    import pickle
    with open(DQN_DIRECTORY, 'wb') as f:
        pickle.dump(pre_trained_dqn, f, pickle.HIGHEST_PROTOCOL)
