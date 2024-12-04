import pandas as pd

from typing import List
from .base_query_builder import query_builder


class original_key_query_builder(query_builder):
    def __init__(self, key: str = ""):
        self.key = key

    def build(self, row: pd.Series) -> str:
        if self.key not in row.index:
            raise KeyError(f"The key '{self.key}' does not exist in the provided row.")
        return row.get(self.key)


class original_keywords_query_builder(original_key_query_builder):
    def __init__(self):
        super().__init__(key="keywords")

    def build(self, row: pd.Series) -> str:
        keywords = super().build(row)
        return " ".join(keyword for keyword in keywords)


class original_exist_keywords_query_builder(original_key_query_builder):
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
