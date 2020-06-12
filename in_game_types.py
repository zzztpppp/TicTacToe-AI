# Chess piece data type is defined in this file

from typing import NewType

ChessPiece = NewType('ChessPiece', int)

CIRCLE = ChessPiece(1)
CROSS = ChessPiece(-1)

Reward = NewType('Reward', int)

WINNING_REWARD = 10  # Reward for winning a game
DRAWING_REWARD = 5   # Reward for a tied game
LOSING_REWARD = -10  # Reward for losing a game
PLAIN_REWARD = 0     # Reward for an ordinary move
