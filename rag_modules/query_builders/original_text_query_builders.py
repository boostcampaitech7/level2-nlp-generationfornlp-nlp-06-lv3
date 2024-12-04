import pandas as pd

from .base_query_builder import query_builder


class original_paragraph_query_builder(query_builder):
    def build(self, row: pd.Series) -> str:
        return row.get("paragraph", "")


class original_question_query_builder(query_builder):
    def build(self, row: pd.Series) -> str:
        return row.get("question", "")


class original_choices_query_builder(query_builder):
    def build(self, row: pd.Series) -> str:
        return row.get("choices_text", "")


class original_question_plus_query_builder(query_builder):
    def build(self, row: pd.Series) -> str:
        return row.get("question_plus", "")
