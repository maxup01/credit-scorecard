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
    def value(cls) -> list[str]:
        return [member.value for member in cls]

def one_hot(variants: list[str], value: str) -> np.ndarray:
    vector = np.zeros(len(variants), dtype=int)
    vector[variants.index(value)] = 1
    return vector