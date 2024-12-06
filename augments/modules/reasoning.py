import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import REASONING
from .langchain_manager import LangchainManager


class Reasoning(BaseProcessor):
    """
    데이터의 추론 과정을 생성하는 클래스.

    LangChain을 활용하여 데이터를 처리하고, 추론 결과를 데이터프레임에 추가합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        Reasoning 클래스 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 입력 데이터프레임. 기본값은 None.

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            source_data (pd.DataFrame): 입력 데이터프레임.
            langchain_manager (LangchainManager): 추론 생성을 위한 LangChain 기반 매니저.
        """
        super().__init__(data_path, experiment_data_path, data)

        # 데이터 로드 또는 초기화
        if data is None:
            self.source_data = self.data_version_manager.get_latest_train_dataframe(major=1, minor=0, load_exp=False)
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")
        else:
            self.source_data = data

        # LangChainManager 설정 - 추론 생성을 위한 프롬프트 설정
        self.langchain_manager = LangchainManager(prompt_type="base", prompt_source="reasoning.txt", output_type="str")

    def process(self):
        """
        데이터 처리 함수.

        데이터를 LangChain을 활용하여 처리하고, 추론 결과를 데이터프레임에 추가합니다.

        Steps:
            1. id 열 삭제.
            2. LangChainManager를 통해 각 행에 대해 reasoning 열 생성.

        Example:
        ```python
            reasoning_processor = Reasoning(data=pd.DataFrame({...}))
            reasoning_processor.process()
        ```
        """
        tqdm.pandas()

        # 'id' 열 삭제
        self.source_data.drop(columns=["id"], inplace=True)

        # LangChain을 활용해 추론 결과 생성
        self.source_data[REASONING] = self.source_data.progress_apply(
            lambda row: self.langchain_manager.invoke(row), axis=1
        )

        # 결과 저장
        self.source_data.to_csv("data/experiments/reasoning.csv", index=False)
