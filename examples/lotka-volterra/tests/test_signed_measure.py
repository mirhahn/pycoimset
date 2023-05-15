#!/usr/bin/env python3

import math
import unittest
import numpy

from numpy.polynomial import Polynomial

from lotka_volterra.similarity_class import (
    IntervalSimilarityClass,
    IntervalSimilaritySpace,
    PolynomialSignedMeasure,
)


class TestSignedMeasure(unittest.TestCase):
    def setUp(self):
        self.space = IntervalSimilaritySpace((-5, 5))
        self.measures = [
            PolynomialSignedMeasure(
                self.space,
                [
                    Polynomial([-2, -4, -1], domain=[-5, -1], window=[-5, -1]
                               ),
                    Polynomial([0, 0, 1], domain=[-1, 1], window=[-1, 1]),
                    Polynomial([-2, 4, -1], domain=[1, 5], window=[1, 5]),
                ]
            ),
        ]
        self.levels = [1, 0, -1]

        self.gt_switch = [
            [
                # level = 1
                [-3, -1, 1, 3],
                # level = 0
                [-2 - math.sqrt(2), 2 + math.sqrt(2)],
                # level = -1
                [-2 - math.sqrt(3), 2 + math.sqrt(3)]
            ]
        ]
        self.ge_switch = [
            [
                # level = 1
                [-3, -1, 1, 3],
                # level = 0
                [-2 - math.sqrt(2), 2 + math.sqrt(2)],
                # level = -1
                [-2 - math.sqrt(3), 2 + math.sqrt(3)]
            ]
        ]
        self.lt_switch = [
            [
                # level = 1
                [-5, -3, -1, 1, 3, 5],
                # level = 0
                [-5, -2 - math.sqrt(2), 2 + math.sqrt(2), 5],
                # level = -1
                [-5, -2 - math.sqrt(3), 2 + math.sqrt(3), 5]
            ]
        ]
        self.le_switch = [
            [
                # level = 1
                [-5, -3, -1, 1, 3, 5],
                # level = 0
                [-5, -2 - math.sqrt(2), 2 + math.sqrt(2), 5],
                # level = -1
                [-5, -2 - math.sqrt(3), 2 + math.sqrt(3), 5]
            ]
        ]

    def test_level_sets_gt(self):
        '''Test level sets for `>`.'''
        for i, (m, switch) in enumerate(zip(self.measures, self.gt_switch)):
            for level, expect in zip(self.levels, switch):
                with self.subTest(i=i, lvl=level):
                    expected_class = IntervalSimilarityClass(m.space, expect)
                    actual_class = m > level
                    self.assertTrue(
                        numpy.allclose(actual_class.switch_times, expect),
                        f"Level set is {repr(actual_class)}; expected "
                        f"{repr(expected_class)}"
                    )

    def test_level_sets_ge(self):
        '''Test level sets for `>=`.'''
        for i, (m, switch) in enumerate(zip(self.measures, self.ge_switch)):
            for level, expect in zip(self.levels, switch):
                with self.subTest(i=i, lvl=level):
                    expected_class = IntervalSimilarityClass(m.space, expect)
                    actual_class = m >= level
                    self.assertTrue(
                        numpy.allclose(actual_class.switch_times, expect),
                        f"Level set is {repr(actual_class)}; expected "
                        f"{repr(expected_class)}"
                    )

    def test_level_sets_lt(self):
        '''Test level sets for `<`.'''
        for i, (m, switch) in enumerate(zip(self.measures, self.lt_switch)):
            for level, expect in zip(self.levels, switch):
                with self.subTest(i=i, lvl=level):
                    expected_class = IntervalSimilarityClass(m.space, expect)
                    actual_class = m < level
                    self.assertTrue(
                        numpy.allclose(actual_class.switch_times, expect),
                        f"Level set is {repr(actual_class)}; expected "
                        f"{repr(expected_class)}"
                    )

    def test_level_sets_le(self):
        '''Test level sets for `<=`.'''
        for i, (m, switch) in enumerate(zip(self.measures, self.le_switch)):
            for level, expect in zip(self.levels, switch):
                with self.subTest(i=i, lvl=level):
                    expected_class = IntervalSimilarityClass(m.space, expect)
                    actual_class = m <= level
                    self.assertTrue(
                        numpy.allclose(actual_class.switch_times, expect),
                        f"Level set is {repr(actual_class)}; expected "
                        f"{repr(expected_class)}"
                    )


if __name__ == '__main__':
    unittest.main()
