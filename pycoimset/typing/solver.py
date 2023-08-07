'''
Static typing protocols for solver components.
'''

from typing import Optional, Protocol, TypeVar

from .space import SignedMeasure, SimilarityClass, SimilaritySpace


__all__ = [
    'UnconstrainedStepFinder',
]


Spc = TypeVar('Spc', bound=SimilaritySpace)


class UnconstrainedStepFinder(Protocol[Spc]):
    '''Protocol for unconstrained step finders.'''

    @property
    def quality(self) -> float:
        '''Return quality constant of this step finder.'''
        ...

    @property
    def gradient(self) -> Optional[SignedMeasure[Spc]]:
        '''Gradient measure.'''
        ...

    @gradient.setter
    def gradient(self, grad: Optional[SignedMeasure[Spc]]) -> None:
        '''Set gradient measure and discard all cached results.'''
        ...

    @property
    def radius(self) -> float:
        '''Trust-region radius.'''
        ...

    @radius.setter
    def radius(self, r: float) -> None:
        '''Set trust-region radius and discard cache.'''
        ...

    @property
    def tolerance(self) -> float:
        '''Step finding tolerance.'''
        ...

    @tolerance.setter
    def tolerance(self, tol: float) -> None:
        '''Set step finding tolerance and discard cached values.'''
        ...

    def get_step(self) -> tuple[SimilarityClass[Spc], float]:
        '''
        Calculate and return step.

        Implementations should use cached results wherever possible,
        but must ensure that such results satisfy the current error
        tolerance.

        :return: A tuple of step and step finding error.
        :raise ValueError: `gradient` is not set.
        '''
        ...
