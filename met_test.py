import unittest
from unittest import TestCase

import numpy as np

from met import IBF, METIBF


class TestIBF(TestCase):
    k = 3
    m = 10

    def setUp(self):
        self.ibf = IBF.create(self.m, self.k)

    def test_insert(self):
        self.ibf.insert(10)
        self.ibf.insert(50)
        self.ibf.insert(100)

        l = self.ibf.peel()

        self.assertCountEqual(l, [10, 50, 100])

    def test_delete(self):
        self.ibf.insert(10)
        self.ibf.insert(50)
        self.ibf.insert(100)

        self.ibf.delete(50)

        l = self.ibf.peel()

        self.assertCountEqual(l, [10, 100])

    def test_peel(self):
        self.ibf.insert(10)
        self.ibf.insert(50)
        self.ibf.insert(100)

        l = self.ibf.peel()

        self.assertCountEqual(l, [10, 50, 100])

    def test_empty(self):
        self.assertFalse(self.ibf)

        self.ibf.insert(10)
        self.ibf.insert(1313)
        self.ibf.insert(100)

        self.assertTrue(self.ibf)

    def test_empty_after_delete(self):
        self.ibf.insert(10)
        self.ibf.insert(1313)
        self.ibf.delete(10)
        self.assertTrue(self.ibf)

        self.ibf.delete(1313)
        self.assertFalse(self.ibf)


class TestIrregularIBF(TestCase):
    m = 10

    @staticmethod
    def key2deg(x):
        return int(x % 4)

    def setUp(self):
        self.ibf = IBF.create_irregular(self.m, self.key2deg)

    def test_zero_deg_insert(self):
        self.ibf.insert(0)

        self.assertFalse(self.ibf)

    def test_zero_deg_delete(self):
        self.ibf.delete(0)

        self.assertFalse(self.ibf)

    def test_zero_deg_list(self):
        self.ibf.insert(3)
        self.ibf.insert(0)
        self.ibf.insert(7)
        self.ibf.insert(400)
        l = self.ibf.peel()

        self.assertCountEqual(l, [3, 7])


class TestMETIBFasIBF(TestIBF):

    def setUp(self):
        self.ibf = METIBF(
            deg_matrix=np.array([[self.k]]),
            m_cells=np.array([self.m]),
            key2type=lambda x: 0,
        )


class TestMETIBFasIrregularIBF(TestIrregularIBF):
    @classmethod
    def key2type(cls, x):
        return cls.key2deg(x)

    def setUp(self):
        self.ibf = METIBF(
            deg_matrix=np.expand_dims(np.arange(100), 0),
            m_cells=np.array([self.m]),
            key2type=self.key2type,
        )


class TestMETIBF(TestIBF):
    @staticmethod
    def key2type(x):
        return int(x % 4)

    def setUp(self):
        deg = np.array([
            [1, 2, 3, 0],
            [2, 2, 2, 4],
            [0, 3, 1, 1],
        ])
        m_cells = np.array([20, 16, 52])

        self.ibf = METIBF(
            deg_matrix=deg,
            m_cells=m_cells,
            key2type=self.key2type,
        )

    def test_peel2(self):
        elements = [
            1 + 11 * 4,
            2 + 23 * 4,
            0 + 24 * 4,
            3 + 73 * 4,
            2 + 10 * 4,
        ]
        for e in elements:
            self.ibf.insert(e)

        l = self.ibf.peel()

        self.assertCountEqual(l, elements)


if __name__ == '__main__':
    unittest.main()
