from typing import List

import pandas as pd

from .base_query_builder import query_builder


class original_key_query_builder(query_builder):
    """
    주어진 row에서 추가적인 처리 없이 단일 key 값을 추출하는 쿼리 빌더.

    Attributes:
        key (str): row에서 추출할 key.
    """

    def __init__(self, key):
        """
        주어진 key로 query_builder를 초기화합니다.

        Args:
            key (str, optional): row에서 추출할 key.
        """
        self.key = key

    def build(self, row: pd.Series) -> str:
        """
        주어진 row에서 key의 값을 추출하여 query를 생성합니다.

        Args:
            row (pd.Series): 값을 추출할 행.

        Raises:
            KeyError: 지정된 key가 row에 존재하지 않을 경우 발생.

        Returns:
            str: row에서 key를 통해 추출된 값
        """
        if self.key not in row.index:
            raise KeyError(f"The key '{self.key}' does not exist in the provided row.")
        return row.get(self.key)


class original_keywords_query_builder(original_key_query_builder):
    """
    주어진 row에서 keywords들을 전부 concat해 반환하는 query builder
    """

    def __init__(self):
        super().__init__(key="keywords")

    def build(self, row: pd.Series) -> str:
        keywords = super().build(row)
        return " ".join(keyword for keyword in keywords)


class original_exist_keywords_query_builder(original_key_query_builder):
    """
    주어진 row에서 keywords_exists가 true인 keywords들을 전부 concat해 반환하는 query builder
    """

    def __init__(self):
        super().__init__(key="keywords")

    def build(self, row: pd.Series) -> str:
        keywords = super().build(row)

        self.key = "keywords_exists"
        keywords_exists = super().build(row)
        return " ".join(keyword for keyword, exists in zip(keywords, keywords_exists) if exists == 1)


class combined_key_query_builder(query_builder):
    def __init__(self, keys: List[str]):
        self.keys = keys

    def build(self, row: pd.Series) -> str:
        query = ""
        for key in self.keys:
            if key not in row.index:
                raise KeyError(f"The key '{key}' does not exist in the provided row.")
            query = query + " " + row.get(key)
        return query
