#!/usr/bin/env python3

import unittest

import numpy

from lotka_volterra.space import (
    IntervalSimilarityClass,
    IntervalSimilaritySpace,
)


class TestSimilarityClass(unittest.TestCase):
    def setUp(self):
        self.space = IntervalSimilaritySpace((0, 12))
        self.ctor_args = [
            ([1, 2, 4, 7, 10, 11],            False),
            ([1, 2, 4, 7, 7, 10, 10, 11, 11], True),
            ([0, 3, 7, 8, 12, 13],            True),
            ([-1, 3, 7, 14],                  True),
        ]
        self.classes = [
            IntervalSimilarityClass(self.space, st, filter=flt)
            for st, flt in self.ctor_args
        ]

        self.expected_switch_times = [
            [1, 2, 4, 7, 10, 11],
            [1, 2, 4, 12],
            [0, 3, 7, 8],
            [0, 3, 7, 12],
        ]
        self.expected_measure = [5, 9, 4, 8]

        self.complement_switch_time = [
            [0, 1, 2, 4, 7, 10, 11, 12],
            [0, 1, 2, 4],
            [3, 7, 8, 12],
            [3, 7],
        ]
        self.union_switch_time = [
            [
                [1, 2, 4, 7, 10, 11],
                [1, 2, 4, 12],
                [0, 3, 4, 8, 10, 11],
                [0, 3, 4, 12],
            ],
            [
                [1, 2, 4, 12],
                [1, 2, 4, 12],
                [0, 3, 4, 12],
                [0, 3, 4, 12],
            ],
            [
                [0, 3, 4, 8, 10, 11],
                [0, 3, 4, 12],
                [0, 3, 7, 8],
                [0, 3, 7, 12],
            ],
            [
                [0, 3, 4, 12],
                [0, 3, 4, 12],
                [0, 3, 7, 12],
                [0, 3, 7, 12],
            ],
        ]
        self.isect_switch_time = [
            [
                [1, 2, 4, 7, 10, 11],
                [1, 2, 4, 7, 10, 11],
                [1, 2],
                [1, 2, 10, 11],
            ],
            [
                [1, 2, 4, 7, 10, 11],
                [1, 2, 4, 12],
                [1, 2, 7, 8],
                [1, 2, 7, 12],
            ],
            [
                [1, 2],
                [1, 2, 7, 8],
                [0, 3, 7, 8],
                [0, 3, 7, 8],
            ],
            [
                [1, 2, 10, 11],
                [1, 2, 7, 12],
                [0, 3, 7, 8],
                [0, 3, 7, 12],
            ],
        ]
        self.diff_switch_time = [
            # [1, 2, 4, 7, 10, 11]
            [
                [],
                [],
                [4, 7, 10, 11],
                [4, 7],
            ],
            # [1, 2, 4, 12]
            [
                [7, 10, 11, 12],
                [],
                [4, 7, 8, 12],
                [4, 7],
            ],
            # [0, 3, 7, 8]
            [
                [0, 1, 2, 3, 7, 8],
                [0, 1, 2, 3],
                [],
                [],
            ],
            # [0, 3, 7, 12]
            [
                [0, 1, 2, 3, 7, 10, 11, 12],
                [0, 1, 2, 3],
                [8, 12],
                [],
            ],
        ]
        self.symmdiff_switch_time = [
            # [1, 2, 4, 7, 10, 11]
            [
                [],
                [7, 10, 11, 12],
                [0, 1, 2, 3, 4, 8, 10, 11],
                [0, 1, 2, 3, 4, 10, 11, 12],
            ],
            # [1, 2, 4, 12]
            [
                [7, 10, 11, 12],
                [],
                [0, 1, 2, 3, 4, 7, 8, 12],
                [0, 1, 2, 3, 4, 7],
            ],
            # [0, 3, 7, 8]
            [
                [0, 1, 2, 3, 4, 8, 10, 11],
                [0, 1, 2, 3, 4, 7, 8, 12],
                [],
                [8, 12],
            ],
            # [0, 3, 7, 12]
            [
                [0, 1, 2, 3, 4, 10, 11, 12],
                [0, 1, 2, 3, 4, 7],
                [8, 12],
                [],
            ],
        ]

    def test_ctor_sorted(self):
        '''Construction with sorted switch time vector'''
        for i, (sc, est) in enumerate(zip(self.classes,
                                          self.expected_switch_times)):
            with self.subTest(i=i):
                self.assertTrue(
                    numpy.array_equal(sc.switch_times, est),
                    f'{sc} does not match {est}'
                )

    def test_ctor_randperm(self):
        '''Construction with randomly permuted switch time vector'''
        for i, ((st, flt), est) in enumerate(zip(self.ctor_args,
                                                 self.expected_switch_times)):
            for j in range(10):
                rst = numpy.random.permutation(st)
                with self.subTest(i=i, j=j):
                    sc = IntervalSimilarityClass(self.space, rst, sort=True,
                                                 filter=flt)
                    self.assertTrue(
                        numpy.array_equal(sc.switch_times, est),
                        f'{rst}: {sc} does not match {est}'
                    )

    def test_measure(self):
        '''Measure calculation'''
        for i, (sc, em) in enumerate(zip(self.classes,
                                         self.expected_measure)):
            with self.subTest(i=i):
                self.assertEqual(
                    sc.measure, em,
                    f'{sc}.measure() is not {em}'
                )

    def test_complement(self):
        '''Complement construction'''
        for i, (sc, st) in enumerate(zip(self.classes,
                                         self.complement_switch_time)):
            with self.subTest(i=i):
                self.assertTrue(
                    numpy.array_equal((~sc).switch_times, st),
                    f'{~sc}.switch_time is not {st}'
                )

    def test_union(self):
        '''Union construction'''
        for i, (sca, row) in enumerate(zip(self.classes,
                                           self.union_switch_time)):
            for j, (scb, st) in enumerate(zip(self.classes, row)):
                with self.subTest(i=i, j=j):
                    self.assertTrue(
                        numpy.array_equal((sca | scb).switch_times, st),
                        f'({sca | scb}).switch_time is not {st}'
                    )

    def test_intersection(self):
        '''Intersection construction'''
        for i, (sca, row) in enumerate(zip(self.classes,
                                           self.isect_switch_time)):
            for j, (scb, st) in enumerate(zip(self.classes, row)):
                with self.subTest(i=i, j=j):
                    self.assertTrue(
                        numpy.array_equal((sca & scb).switch_times, st),
                        f'{sca & scb}.switch_time is not {st}'
                    )

    def test_difference(self):
        '''Difference construction'''
        for i, (sca, row) in enumerate(zip(self.classes,
                                           self.diff_switch_time)):
            for j, (scb, st) in enumerate(zip(self.classes, row)):
                with self.subTest(i=i, j=j):
                    self.assertTrue(
                        numpy.array_equal((sca - scb).switch_times, st),
                        f'{sca - scb}.switch_time is not {st}'
                    )

    def test_symmdiff(self):
        '''Symmetric difference construction'''
        for i, (sca, row) in enumerate(zip(self.classes,
                                           self.symmdiff_switch_time)):
            for j, (scb, st) in enumerate(zip(self.classes, row)):
                with self.subTest(i=i, j=j):
                    self.assertTrue(
                        numpy.array_equal((sca ^ scb).switch_times, st),
                        f'{sca ^ scb}.switch_time is not {st}'
                    )


if __name__ == '__main__':
    unittest.main()
