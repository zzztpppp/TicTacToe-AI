# User defined excetions for our game


class NonEmptySlotError(ValueError):
    """
    Raise when a player try to put a chess at a
    non empty slot
    """
    def __int__(self, message):
        self.message = message
