# Chess piece data type is defined in this file

from typing import NewType

ChessPiece = NewType('ChessPiece', int)

CIRCLE = ChessPiece(1)
CROSS = ChessPiece(-1)
