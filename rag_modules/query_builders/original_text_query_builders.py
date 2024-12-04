import pandas as pd

from .base_query_builder import query_builder


class original_query_key_builder(query_builder):
    def __init__(self, key: str = ""):
        self.key = key

    def build(self, row: pd.Series) -> str:
        if self.key not in row.index:
            raise KeyError(f"The key '{self.key}' does not exist in the provided row.")
        return row.get(self.key)


class original_paragraph_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="paragraph")


class original_question_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="question")


class original_choices_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="choices_text")


class original_question_plus_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="question_plus")


class original_summarization_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="summarization")


class original_keywords_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="keywords")

    def build(self, row: pd.Series) -> str:
        keywords = super().build(row)
        return " ".join(keyword for keyword in keywords)


class original_exist_keywords_query_builder(original_query_key_builder):
    def __init__(self):
        super().__init__(key="keywords")

    def build(self, row: pd.Series) -> str:
        keywords = super().build(row)

        self.key = "keywords_exists"
        keywords_exists = super().build(row)
        return " ".join(keyword for keyword, exists in zip(keywords, keywords_exists) if exists == 1)
