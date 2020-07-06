"""
Game configuration
"""

import logging


class GameConfig:

    BOARD_SIZE = 3

    WINNING_STREAK = 3


class AIConfig:
    LOAD_DQN_ARCHIVE = True


game_config = GameConfig()
ai_config = AIConfig()

logger = logging.getLogger('game_logger')

