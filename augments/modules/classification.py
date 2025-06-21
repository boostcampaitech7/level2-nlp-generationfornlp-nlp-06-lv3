import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import ANALYSIS, CATEGORY
from .langchain_manager import LangchainManager


class Classification(BaseProcessor):
    """
    데이터 분류 작업을 수행하는 클래스.

    LangchainManager를 활용하여 입력 데이터를 처리하고,
    분석 및 분류 결과를 데이터프레임에 추가합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        Classification 클래스 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 입력 데이터프레임. 기본값은 None.

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            langchain_manager (LangchainManager): LangChain 기반 데이터 처리 매니저.
        """
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")

        # LangChainManager 설정 - 프롬프트와 출력 형식 지정
        self.langchain_manager = LangchainManager(
            prompt_type="data_classification", prompt_source="information_source_with_reasoning.txt", output_type="json"
        )

    def process(self):
        """
        데이터 처리 함수.

        LangChain을 활용하여 데이터를 처리한 후, 분석 결과와 분류 결과를 데이터프레임에 추가합니다.

        Notes:
            - tqdm의 progress_apply를 활용하여 처리 상태를 시각적으로 표시.
            - 처리 결과는 source_data의 analysis 및 category 열에 저장됩니다.

        Example:
            classification = Classification(data=pd.DataFrame({"text": ["sample data"]}))
            classification.process()
        """
        tqdm.pandas()  # 진행 상황 표시를 위한 tqdm 적용
        # LangChainManager를 활용한 데이터 처리
        result = self.source_data.progress_apply(lambda row: self.langchain_manager.invoke(row), axis=1)
        # 처리 결과를 Series로 분리
        result_df = result.apply(pd.Series)
        # 결과를 source_data의 ANALYSIS, CATEGORY 열에 저장
        self.source_data[[ANALYSIS, CATEGORY]] = result_df

        # 결과 저장
        self.source_data.to_csv("data/experiments/classification.csv", index=False)
