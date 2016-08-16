import pandas as pd


class DataView(object):
    """This is not truely a view and will store multiply copies of the
    dataset in memory. Use at your own risk.
    """
    def __init__(self):
        self._data = None
        self._cleaned = None
        self._ordinal = None
        self._onehot = None


    def as_raw(self):
        raise NotImplementedError()

    def as_cleaned(self):
        if self._cleaned is not None:
            return self._cleaned

    def as_ordinal(self):
        if self._ordinal is not None:
            return self._ordinal

    def as_onehot(self):
        if self._onehot is not None:
            return self._onehot
