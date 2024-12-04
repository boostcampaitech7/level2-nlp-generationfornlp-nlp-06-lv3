from query_builder import query_builder


class original_paragraph_query_builder(query_builder):
    def build(self) -> str:
        return self.paragraph


class original_question_query_builder(query_builder):
    def build(self) -> str:
        return self.question


class original_choices_query_builder(query_builder):
    def build(self) -> str:
        return self.choices


class original_question_plus_query_builder(query_builder):
    def build(self) -> str:
        return self.question_plus
