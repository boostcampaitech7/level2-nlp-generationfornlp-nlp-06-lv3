from abc import ABC, abstractmethod

import pandas as pd

from utils import DataVersionManager


class BaseProcessor(ABC):
    """
    데이터 처리를 위한 기본 추상 클래스.

    이 클래스는 데이터를 처리하는 공통 기능과 데이터 버전 관리를 제공하며,
    구체적인 데이터 처리 방식은 서브클래스에서 구현해야 합니다.
    """

    def __init__(
        self, data_path: str = "data/", experiment_data_path: str = "data/experiments/", data: pd.DataFrame = None
    ):
        """
        BaseProcessor 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 데이터프레임 초기 값. 기본값은 None.

        Attributes:
            data_version_manager (DataVersionManager): 데이터 버전 관리를 위한 유틸리티 클래스.
            source_data (pd.DataFrame): 데이터 소스. 초기화 시 전달된 데이터를 저장.
        """
        self.data_version_manager = DataVersionManager(data_path=data_path, experiment_data_path=experiment_data_path)
        self.source_data: pd.DataFrame = data

    @property
    def data(self) -> pd.DataFrame:
        """
        데이터프레임 반환 프로퍼티.

        Returns:
            pd.DataFrame: 현재 소스 데이터프레임.
        """
        return self.source_data

    @abstractmethod
    def process(self) -> None:
        """
        데이터 처리 로직을 정의하는 추상 메서드.

        NOTE: 이 메서드는 서브클래스에서 반드시 구현해야 합니다.
        """
        pass

    def sample_data(self, n: int, seed: int = 1004):
        """
        데이터를 랜덤 샘플링하여 갱신하는 함수.

        Args:
            n (int): 샘플링할 데이터 개수.
            seed (int, optional): 랜덤 시드 값. 기본값은 1004.

        Example:
            processor = SomeProcessor(data=pd.DataFrame({"col1": [1, 2, 3]}))
            processor.sample_data(n=2)
        """
        self.source_data = self.source_data.sample(n=n, random_state=seed)
