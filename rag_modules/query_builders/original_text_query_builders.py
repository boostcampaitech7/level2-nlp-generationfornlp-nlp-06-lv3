from ast import literal_eval

import pandas as pd

from .base_query_builder import query_builder


class original_paragraph_query_builder(query_builder):
    def build(self, data: pd.Series) -> str:
        return data.get("paragraph", "")


class original_question_query_builder(query_builder):
    def build(self, data: pd.Series) -> str:
        return data.get("question", "")


class original_choices_query_builder(query_builder):
    def build(self, data: pd.Series) -> str:
        choices = literal_eval(data.get("choices", ""))
        choices_string = "\n".join(choices)
        return choices_string


class original_question_plus_query_builder(query_builder):
    def build(self, data: pd.Series) -> str:
        return data.get("question_plus", "")
