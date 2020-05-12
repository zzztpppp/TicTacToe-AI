"""
Utility funtions
"""
import numpy as np


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
    if vals.shape != (row, col):
        raise ValueError("Shape of values should row by col")
    sep = '\n' + '+---' * col + '+\n'
    content = [sep]
    for r in range(row):
        row_content = ['| {v} '.format(v=vals[r, c]) for c in range(col)]
        row_content.extend(['|', sep])
        content.append(''.join(row_content))
    return ''.join(content)
