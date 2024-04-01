# PyCoimset: Python library for COntinuous IMprovement of SETs
#
# Copyright 2024 Mirko Hahn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Logger for tabular progress logs.
'''

from dataclasses import dataclass
import sys
from typing import IO, Iterable, Mapping, Optional


__all__ = ['TabularLogger']


@dataclass(frozen=True, slots=True)
class ColumnDesc:
    '''
    Description of a column.
    '''
    index: int
    label: str
    format: str
    width: int


class TabularLogger:
    _cols: list[str]
    _desc: Mapping[str, ColumnDesc]
    _file: IO[str]
    _flush: bool
    _count: int
    _interval: int

    def __init__(self, cols: Iterable[str],
                 label: Optional[Mapping[str, str]] = None,
                 format: Optional[Mapping[str, str]] = None,
                 width: Optional[Mapping[str, int]] = None,
                 file: Optional[IO[str]] = None,
                 flush: bool = False,
                 interval: int = 50):
        if label is None:
            label = {}
        if format is None:
            format = {}
        if width is None:
            width = {}
        if file is None:
            file = sys.stdout

        self._cols = list(cols)
        self._desc = {
            key: ColumnDesc(
                idx,
                (item_label := label.get(key, key)),
                format.get(key, ''),
                max(len(item_label), width.get(key, 0))
            )
            for idx, key in enumerate(cols)
        }
        self._file = file
        self._flush = flush
        self._count = 0
        self._interval = interval

    def print_header(self) -> None:
        print(' | '.join([
            (desc := self._desc[key]).label.rjust(desc.width)
            for key in self._desc
        ]), file=self._file, flush=False)
        print('-+-'.join(['-' * self._desc[key].width for key in self._cols]),
              file=self._file, flush=self._flush)

    def print_line(self, *args, **kwargs) -> None:
        values = list(args) + [None] * (len(self._cols) - len(args))
        for key, value in kwargs.items():
            values[self._desc[key].index] = value
        strings = [
            format(value, (desc := self._desc[key]).format).rjust(desc.width)
            if value is not None
            else '---'.rjust(self._desc[key].width)
            for key, value in zip(self._cols, values)
        ]
        print(' | '.join(strings), file=self._file, flush=self._flush)

    def push_line(self, *args, **kwargs) -> None:
        if self._count % self._interval == 0:
            self.print_header()
            self._count = 0
        self.print_line(*args, **kwargs)
        self._count += 1
