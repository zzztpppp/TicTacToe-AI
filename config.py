"""
Game configuration
"""

import logging


class GameConfig:

    BOARD_SIZE = 3

    WINNING_STREAK = 3


game_config = GameConfig()

logger = logging.getLogger('game_logger')

