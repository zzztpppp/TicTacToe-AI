"""
Utility funtions
"""
import numpy as np
from typing import Tuple
from functools import wraps
from exceptions import NonEmptySlotError


def get_string_grid(col: int, row: int, vals: np.ndarray) -> str:
    """
    Return a string representation of a rectangle grids with given
    size of n by n and width and given values filled

    :param col: Number of rows of the square
    :param row: Number of columns of the square
    :param vals: Values to fill the grids

    :return: String representation of the rectangle grid

    Example:

    >>> print(get_string_grid(3, 3, np.array([[1,2,3], [1,2,3], [1,2,3]])))

    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+

    """
    from tic_tac_toe import CROSS, CIRCLE

    if vals.shape != (row, col):
        raise ValueError("Shape of values should row by col")
    sep = '\n' + '+---' * col + '+\n'
    content = [sep]
    for r in range(row):
        row_content = ['| {v} '.format(v='X' if vals[r, c] == CROSS else ('O' if vals[r, c] == CIRCLE else ' '))
                       for c in range(col)]
        row_content.extend(['|', sep])
        content.append(''.join(row_content))
    return ''.join(content)


def index2coordinate(index: int, size: int) -> Tuple[int, int]:
    """
    Given index of the tile in a square, convert it into
    row-column coordinate

    :param index: Index of the tile in a square, starting with 0
    :param size:
    :return:

    Example:

    >>> index2coordinate(0, 3)
    (0, 0)

    >>> index2coordinate(8, 3)
    (2, 2)

    """
    row = int(index / size)
    column = index % size

    return row, column


def translate_game_status(x, y):
    """
    Kinda like logic not xor, return 1 if x == y != 0 and -1 if x != y != 0, 0 otherwise
    """
    if x == 0 or y == 0:
        return 1

    if x == y:
        return 1
    else:
        return -1

def alternate(first, second):
    return first, lambda: alternate(second, first)


# Decorators
def try_until_success(func):
    """
    A decorator that allows us to try execute the given function
    until no exceptions are raised

    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NonEmptySlotError:
            print('Invalid arguments, please try again')
            return wrapper(*args, **kwargs)

    return wrapper

