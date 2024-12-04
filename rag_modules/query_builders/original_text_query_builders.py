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


class original_keywords_query_builder(query_builder):
    def build(self, row: pd.Series) -> str:
        try:
            # 'keywords'이 df.columns에 없으면 KeyError가 발생
            return " ".join(keyword for keyword in row["keywords"])
        except KeyError:
            print("Error: keywords column이 DataFrame에 존재하지 않습니다.")


class original_exist_keywords_query_builder(query_builder):
    def build(self, row: pd.Series) -> str:
        try:
            # 'keywords'이 df.columns에 없으면 KeyError가 발생
            return " ".join(keyword for keyword, exists in zip(row["keywords"], row["keywords_exists"]) if exists == 1)
        except KeyError:
            print("Error: keywords column이 DataFrame에 존재하지 않습니다.")
