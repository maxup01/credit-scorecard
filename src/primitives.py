from enum import Enum
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
        members = list(PersonHomeOwnership)

        vector = np.zeros(len(members), dtype=int)
        vector[members.index(self)] = 1
        return vector


class LoanIntent(Enum):
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"
    EDUCATION = "EDUCATION"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    MEDICAL = "MEDICAL"
    PERSONAL = "PERSONAL"
    VENTURE = "VENTURE"