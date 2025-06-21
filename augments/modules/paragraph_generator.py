import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import (
    CHOICES,
    CRAWLED_TEXT,
    DEFAULT_COLUMNS,
    KEYWORD_PREFIX,
    PARAGRAPH,
    QUESTION,
    QUESTION_PLUS,
    RAW_PARAGRAPH,
    REASONING,
    SUMMARY_SUFFIX,
)
from .langchain_manager import LangchainManager


class ParagraphGenerator(BaseProcessor):
    """
    파라그래프(문단) 생성을 위한 클래스.

    주어진 데이터를 기반으로 크롤링된 텍스트를 조합하여 LangChain을 활용해
    문단을 생성합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        ParagraphGenerator 클래스 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 입력 데이터프레임. 기본값은 None.

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            source_data (pd.DataFrame): 입력 데이터프레임.
            langchain_manager (LangchainManager): 문단 생성을 위한 LangChain 기반 매니저.
        """
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")

        # LangChainManager 설정 - 문단 생성을 위한 프롬프트 설정
        self.langchain_manager = LangchainManager(
            prompt_type="paragraph_generation",
            prompt_source="generation_from_wiki.txt",
            output_type="str",
        )

    def process(self):
        """
        데이터 처리 함수.

        크롤링된 텍스트를 조합하고 LangChain을 활용해 문단을 생성한 결과를
        데이터프레임에 추가합니다.

        Steps:
            1. 크롤링된 텍스트 열들을 결합하여 CRAWLED_TEXT 생성.
            2. 필요한 열(PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, CRAWLED_TEXT) 추출.
            3. LangChainManager를 사용하여 RAW_PARAGRAPH 생성.

        Example:
        ```python
            generator = ParagraphGenerator(data=pd.DataFrame({...}))
            generator.process()
        ```
        """
        tqdm.pandas()

        # 크롤링된 텍스트 결합
        columns_to_join = [KEYWORD_PREFIX + str(i) + SUMMARY_SUFFIX for i in range(1, 6)]
        self.source_data.loc[:, CRAWLED_TEXT] = self.source_data[columns_to_join].apply(
            lambda row: "\n".join(row), axis=1
        )

        # 필요한 열만 추출
        self.source_data = self.source_data[DEFAULT_COLUMNS + [REASONING, CRAWLED_TEXT]].copy()

        df = self.source_data[[PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, CRAWLED_TEXT]]

        # LangChain을 활용한 문단 생성
        self.source_data.loc[:, RAW_PARAGRAPH] = df.progress_apply(
            lambda row: self.langchain_manager.invoke(row.to_dict()), axis=1
        )

        self.source_data = self.source_data[DEFAULT_COLUMNS + [REASONING, RAW_PARAGRAPH]].copy()

        # 결과 저장
        self.source_data.to_csv("data/experiments/paragraph_generation.csv", index=False)
