import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import CHOICES, DEFAULT_COLUMNS, DOCUMENT, PARAGRAPH, QUESTION, QUESTION_PLUS, RAW_PARAGRAPH
from .langchain_manager import LangchainManager


class ParagraphTrimmer(BaseProcessor):
    """
    문단을 다듬고 최적화하는 클래스.

    LangChain을 활용하여 주어진 문단을 요약하거나 조정하고,
    결과를 데이터프레임에 업데이트합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        ParagraphTrimmer 클래스 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 입력 데이터프레임. 기본값은 None.

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            source_data (pd.DataFrame): 입력 데이터프레임.
            langchain_manager (LangchainManager): 문단 다듬기를 위한 LangChain 기반 매니저.
        """
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")

        # LangChainManager 설정 - 문단 다듬기를 위한 프롬프트 설정
        self.langchain_manager = LangchainManager(
            prompt_type="paragraph_generation",
            prompt_source="trimming_paragraph.txt",
            output_type="str",
        )

    def process(self):
        """
        데이터 처리 함수.

        주어진 문단을 다듬고 결과를 데이터프레임에 추가하거나 갱신합니다.

        Steps:
            1. 필요한 열(PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, RAW_PARAGRAPH) 추출.
            2. LangChain을 활용하여 DOCUMENT 열 생성.
            3. QUESTION_PLUS 열 업데이트: 기존 문단과 결합.
            4. PARAGRAPH 열 갱신: 다듬어진 DOCUMENT 열로 대체.
            5. 데이터프레임 열 정리.

        Example:
        ```python
            trimmer = ParagraphTrimmer(data=pd.DataFrame({...}))
            trimmer.process()
        ```
        """
        tqdm.pandas()

        # 필요한 열만 추출
        df = self.source_data[[PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, RAW_PARAGRAPH]]

        # LangChain을 활용해 문단 다듬기
        self.source_data[DOCUMENT] = df.progress_apply(lambda row: self.langchain_manager.invoke(row.to_dict()), axis=1)

        # QUESTION_PLUS 열 업데이트
        self.source_data[QUESTION_PLUS] = self.source_data.apply(
            lambda row: row[PARAGRAPH] if pd.isna(row[QUESTION_PLUS]) else row[PARAGRAPH] + "\n" + row[QUESTION_PLUS],
            axis=1,
        )

        # PARAGRAPH 열 갱신
        self.source_data[PARAGRAPH] = self.source_data[DOCUMENT]

        # 데이터프레임 열 정리
        self.source_data = self.source_data[DEFAULT_COLUMNS]

        self.source_data.to_csv("data/experiments/paragraph_trimmer.csv", index=False)
