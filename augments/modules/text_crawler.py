import os

import pandas as pd
import wikipediaapi as wk
from dotenv import load_dotenv
from tqdm import tqdm

from .constants import EXISTS_SUFFIX, KEYWORD_PREFIX, PAGE_SUFFIX


class WikipediaCrawler:
    """
    위키피디아에서 텍스트를 크롤링하는 모듈.

    주어진 키워드 리스트를 기반으로 위키피디아 문서 존재 여부를 확인하고,
    해당 문서의 내용을 데이터프레임에 저장합니다.
    """

    def __init__(self, data: pd.DataFrame, languange="ko"):
        """
        WikipediaCrawler 클래스 초기화 함수.

        Args:
            data (pd.DataFrame): 입력 데이터프레임. `keyword_i` 열을 포함해야 함.
            languange (str, optional): 위키피디아 언어를 설정. 기본값은 "ko".

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            source_data (pd.DataFrame): 입력 데이터프레임.
            wiki (wk.Wikipedia): Wikipedia API 객체.
        """
        load_dotenv()
        USER_AGENT = os.getenv("WIKI_USER_AGENT")
        self.wiki = wk.Wikipedia(user_agent=USER_AGENT, language=languange)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")

    @property
    def data(self) -> pd.DataFrame:
        """
        데이터프레임 반환 프로퍼티.

        Returns:
            pd.DataFrame: 크롤링된 데이터프레임.
        """
        return self.source_data

    def crawl(self):
        """
        위키피디아 문서 존재 여부를 판별하고 내용을 데이터프레임에 추가하는 함수.

        Steps:
            1. `keyword_i_exists` 및 `keyword_i_page` 열 초기화.
            2. 각 키워드에 대해 위키피디아 페이지 존재 여부와 내용을 확인.
            3. 결과를 데이터프레임에 저장.

        Returns:
            pd.DataFrame: 위키피디아 문서 존재 여부와 문서 텍스트 내용이 저장된 데이터.

        Example:
        ```python
            crawler = WikipediaCrawler(data=pd.DataFrame({...}))
            crawled_data = crawler.crawl()
        ```
        """
        # 초기화: 존재 여부와 페이지 내용 열 생성
        for i in range(1, 6):
            self.source_data[KEYWORD_PREFIX + str(i) + EXISTS_SUFFIX] = False
            self.source_data[KEYWORD_PREFIX + str(i) + PAGE_SUFFIX] = ""

        # 키워드별 위키피디아 문서 존재 여부 및 내용 확인
        for idx, row in tqdm(self.source_data.iterrows(), total=len(self.source_data), desc="Crawling Wikipedia"):
            for j in range(1, 6):
                page = self.wiki.page(row[KEYWORD_PREFIX + str(j)])
                self.source_data.at[idx, KEYWORD_PREFIX + str(j) + EXISTS_SUFFIX] = page.exists()
                self.source_data.at[idx, KEYWORD_PREFIX + str(j) + PAGE_SUFFIX] = page.text

        # 결과 저장
        self.source_data.to_csv("data/experiments/wikipedia_documents.csv", index=False)
