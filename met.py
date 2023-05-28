import itertools
from abc import ABC
from collections.abc import Callable, Collection, Iterable

import numpy as np

import hashing

np_type = np.uint64


class InvertibleBloomFilterAPI(ABC):
    def insert(self, x):
        pass

    def delete(self, x):
        pass

    def peel(self) -> Collection:
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

    def insert(self, x):
        self._indel(x, 1)

    def delete(self, x):
        self._indel(x, -1)

    def _peel_once(self):
        to_delete = {int(xor) for count, xor in self.T.T if count == 1}

        for x in to_delete:
            self.delete(x)

        return to_delete

    def peel(self) -> Collection:
        res = []
        while peels := self._peel_once():
            res.extend(peels)

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