import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import CATEGORY, DEFAULT_COLUMNS, KEYWORD_PREFIX, KEYWORDS, NEED_KNOWLEDGE, REASONING
from .langchain_manager import LangchainManager


class KeywordsExtractor(BaseProcessor):
    """
    키워드 추출을 위한 클래스.

    LangchainManager를 활용하여 문제 해결에 필요한 키워드를 추출하고,
    데이터프레임을 업데이트합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        KeywordsExtractor 클래스 초기화 함수.

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
            prompt_type="keyword_extraction",
            prompt_source="wikipedia_search_keyword_with_reasoning.txt",
            output_type="str",
        )

    def extract_keyword(self, data: pd.Series) -> list[str]:
        """
        문제 해결에 필요한 키워드 5개를 추출하고 결과를 배열로 반환하는 함수.

        Args:
            data (pd.Series): 제공된 데이터셋의 한 행(row), 문제 하나.

        Returns:
            list[str]: 추출된 5개 키워드.
        """
        response = self.langchain_manager.invoke(data.to_dict())
        response_split = response.split(",")
        keywords = [keyword.strip() for keyword in response_split]
        return keywords

    def process(self):
        """
        데이터 처리 함수.

        `외적 추론` 카테고리에 해당하는 데이터를 필터링하고, 키워드를 추출하여
        키워드별 칼럼으로 데이터프레임에 추가합니다.

        Steps:
            1. category가 외적 추론인 데이터 필터링.
            2. 주요 열만 유지(paragraph, question, question_plus, choices, answer, reasoning).
            3. extract_keyword 메서드를 통해 키워드 추출.
            4. 키워드를 각 칼럼으로 분리하여 데이터프레임에 병합.

        Example:
            extractor = KeywordsExtractor(data=pd.DataFrame({...}))
            extractor.process()
        """
        tqdm.pandas()

        # 외적 추론 카테고리에 해당하는 데이터 필터링
        self.source_data = self.source_data[self.source_data[CATEGORY] == NEED_KNOWLEDGE]

        # 주요 열만 유지
        self.source_data = self.source_data[DEFAULT_COLUMNS + [REASONING]]

        # extract_keyword 함수로 키워드 추출
        self.source_data[KEYWORDS] = self.source_data.progress_apply(self.extract_keyword, axis=1)

        # 키워드를 유용하게 다루기 위해 각 키워드를 별도의 칼럼으로 분리
        keywords_expanded = pd.DataFrame(self.source_data[KEYWORDS].tolist(), index=self.source_data.index)
        keywords_expanded.columns = [KEYWORD_PREFIX + str(i + 1) for i in range(keywords_expanded.shape[1])]

        # 기존 데이터를 키워드 칼럼들과 병합
        self.source_data = pd.concat([self.source_data.drop(columns=[KEYWORDS]), keywords_expanded], axis=1)

        # 결과 저장
        self.source_data.to_csv("data/experiments/keywords.csv", index=False)
