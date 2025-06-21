import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import DEFAULT_COLUMNS, DOCUMENT, KEYWORD_PREFIX, KEYWORDS, PAGE_SUFFIX, REASONING, SUMMARY_SUFFIX
from .langchain_manager import LangchainManager


class Summarizer(BaseProcessor):
    """
    위키피디아 문서 요약을 위한 클래스.

    LangChain을 활용하여 주어진 문서를 요약하고, 결과를 데이터프레임에 추가합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        Summarizer 클래스 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 입력 데이터프레임. 기본값은 None.

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            source_data (pd.DataFrame): 입력 데이터프레임.
            langchain_manager (LangchainManager): 문서 요약을 위한 LangChain 기반 매니저.
        """
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")

        # LangChainManager 설정 - 문서 요약을 위한 프롬프트 설정
        self.langchain_manager = LangchainManager(
            prompt_type="paragraph_generation",
            prompt_source="document_summarization.txt",
            output_type="str",
        )

    def summarize(self, row: pd.Series, i: int) -> str:
        """
        위키피디아 문서를 요약하는 함수.

        Args:
            row (pd.Series): 제공된 데이터셋의 한 행(row), 문제 하나.
            i (int): 참고할 키워드 인덱스.

        Returns:
            str: 요약된 문서 내용.

        Example:
        ```python
            row = pd.Series({
                "KEYWORD_1_PAGE": "Document content...",
                "KEYWORD_1": "keyword1",
                ...
            })
            summary = summarizer.summarize(row, 1)
        ```
        """
        # 키워드 리스트 생성
        keywords_list = [row[KEYWORD_PREFIX + str(idx)] for idx in range(1, 6)]
        keywords = ", ".join(keywords_list)

        # 문서가 없는 경우 빈 문자열 반환
        if pd.isna(row[KEYWORD_PREFIX + str(i) + PAGE_SUFFIX]):
            return ""
        document = row[KEYWORD_PREFIX + str(i) + PAGE_SUFFIX]

        # LangChainManager를 활용한 요약 생성
        response = self.langchain_manager.invoke(
            {
                KEYWORDS: keywords,
                DOCUMENT: document,
            },
        )
        return response

    def process(self):
        """
        데이터 처리 함수.

        각 키워드에 대한 문서를 요약하고, 결과를 데이터프레임에 추가합니다.

        Steps:
            1. 각 키워드 인덱스(1~5)에 대해 문서를 요약.
            2. 요약 결과를 KEYWORD_PREFIX + SUMMARY_SUFFIX 열에 저장.

        Example:
            summarizer = Summarizer(data=pd.DataFrame({...}))
            summarizer.process()
        """
        tqdm.pandas()

        # 각 키워드 인덱스에 대해 문서 요약 생성
        for i in range(1, 6):
            self.source_data[KEYWORD_PREFIX + str(i) + SUMMARY_SUFFIX] = self.source_data.progress_apply(
                lambda row: self.summarize(row, i), axis=1
            )

        self.source_data = self.source_data[
            DEFAULT_COLUMNS + [KEYWORD_PREFIX + str(i) + SUMMARY_SUFFIX for i in range(1, 6)] + [REASONING]
        ]

        # 결과 저장
        self.source_data.to_csv("data/experiments/summarization.csv", index=False)
