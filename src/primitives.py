from enum import Enum
from typing import Callable

import numpy as np

class PersonHomeOwnership(Enum):
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"
    OWN = "OWN"
    RENT = "RENT"

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]

    def one_hot(self) -> np.ndarray:
        return one_hot(PersonHomeOwnership.values(), self.value)


class LoanIntent(Enum):
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"
    EDUCATION = "EDUCATION"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    MEDICAL = "MEDICAL"
    PERSONAL = "PERSONAL"
    VENTURE = "VENTURE"

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]

    def one_hot(self) -> np.ndarray:
        return one_hot(LoanIntent.values(), self.value)

class LoanGrade(Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'
    E = 'E'
    F = 'F'
    G = 'G'

def one_hot(variants: list[str], value: str) -> np.ndarray:
    """Build a one-hot vector marking where ``value`` sits among ``variants``.

    Args:
        variants: All possible values, in a fixed order. Must be unique —
            a repeated entry would make more than one position "correct"
            for the same value, so the result would depend on whichever
            occurrence ``list.index`` happens to find first.
        value: The value to encode. Must be contained in ``variants``,
            otherwise ``list.index`` raises ``ValueError``.

    Returns:
        np.ndarray: A vector of length ``len(variants)``, all zeros except
        a single ``1`` at ``value``'s position in ``variants``.
    """
    vector = np.zeros(len(variants), dtype=int)
    vector[variants.index(value)] = 1
    return vector