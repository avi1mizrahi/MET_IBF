import itertools
from abc import ABC
from collections.abc import Callable, Collection, Iterable

import numpy as np

import hashing

np_type = np.int64


class InvertibleBloomFilterAPI(ABC):
    def insert(self, x):
        pass

    def insert_from(self, s: Iterable):
        """
        Helper method.
        :param s: set of elements to be inserted.
        """
        for x in s:
            self.insert(x)

    def delete(self, x):
        pass

    def peel(self) -> Collection:
        """
        Return the elements that were inserted to the IBF.
        Notes:
             - The collection can be partial.
             - The operation is destructive; returned elements are deleted.
        """
        pass

    def __bool__(self):
        pass


class IBF(InvertibleBloomFilterAPI):
    def __init__(self, m: int, hasher: Callable[[int], Iterable[int]]):
        self.T = np.zeros((2, m), dtype=np_type)
        self.hasher = hasher
        self.n = 0

    @staticmethod
    def create_irregular(m: int, key2deg: Callable[[int], int]):
        return IBF(m, lambda x: hashing.hash_sample(x, range(m), key2deg(x)))

    @staticmethod
    def create(m, k):
        return IBF.create_irregular(m, lambda x: k)

    @property
    def m(self):
        return self.T.shape[1]

    def _indel(self, x, c):
        idxs = self.hasher(x)
        if len(idxs) == 0:
            return
        self.n += c
        self.T[0, idxs] += np_type(c)
        self.T[1, idxs] ^= np_type(x)
        return idxs

    def insert(self, x):
        self._indel(x, 1)

    def delete(self, x):
        self._indel(x, -1)

    def peel(self) -> Collection:
        pure_cells = {i for i, (count, xor) in enumerate(self.T.T) if count == 1}
        res = []

        while pure_cells:
            i_pure = pure_cells.pop()
            x = int(self.T[1, i_pure])
            res.append(x)
            idxs = self._indel(x, -1)
            for i in idxs:
                if self.T[0, i] == 1:
                    pure_cells.add(i)
                if self.T[0, i] == 0:
                    pure_cells.discard(i)

        return res

    def __bool__(self):
        return bool(self.T.any())


class METIBF(InvertibleBloomFilterAPI):
    def __init__(self, deg_matrix: np.ndarray, m_cells: np.ndarray, key2type: Callable[[int], int]):
        self.D = deg_matrix
        self.M = m_cells
        self.key2type = key2type
        self.tables: list[InvertibleBloomFilterAPI] = [
            IBF(m=m, hasher=self.create_table_hahser(cell_type))
            for cell_type, m in enumerate(self.M)
        ]

    def create_table_hahser(self, cell_type):
        return lambda x: hashing.hash_sample(x + cell_type,
                                             range(self.M[cell_type]),
                                             self.D[cell_type, self.key2type(x)])

    def insert(self, x):
        for table in self.tables:
            table.insert(x)

    def delete(self, x):
        for table in self.tables:
            table.delete(x)

    def _peel_once(self):
        removed_per_table = [set(table.peel()) for table in self.tables]
        removed_total = set(itertools.chain.from_iterable(removed_per_table))

        for removed_already, table in zip(removed_per_table, self.tables):
            for x in removed_total - removed_already:
                table.delete(x)

        return removed_total

    def peel(self) -> Collection:
        res = []
        while peels := self._peel_once():
            res.extend(peels)

        return res

    def __bool__(self):
        return any(map(bool, self.tables))
