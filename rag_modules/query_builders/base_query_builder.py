from abc import ABC, abstractmethod

import pandas as pd


class query_builder(ABC):
    """
    df의 한 행이 입력으로 들어오면 적절한 query를 생성해주는 builder
    """

    @abstractmethod
    def build(self, data: pd.Series) -> str:
        """
        query를 생성하는 메인 메서드.
        """
        pass
