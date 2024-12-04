from abc import ABC, abstractmethod

import pandas as pd


class query_builder(ABC):
    """
    df의 한 행이 입력으로 들어오면 적절한 query를 생성해주는 builder
    """

    def __init__(self, data: pd.Series):
        self.paragraph = data.get("paragraph", "")
        self.question = data.get("question", "")
        self.choices = data.get("choices", [])
        self.question_plus = data.get("question_plus", "")

    @abstractmethod
    def build(self) -> str:
        """
        Query를 생성하는 메인 메서드.
        """
        pass
