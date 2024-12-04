from .advanced_query_builders import TFIDFQueryBuilder
from .base_query_builder import QueryBuilder
from .original_text_query_builders import (
    CombinedKeyQueryBuilder,
    OriginalExistKeywordsQueryBuilder,
    OriginalKeywordsQueryBuilder,
)


def set_query_builder_from_config(query_builder_type: str) -> QueryBuilder:
    """
    query builder type에 따라 query building 시 사용되는 query builder를 반환합니다.

    Args:
        query_builder_type (str): 사전 정의된 query builder type str

    Returns:
        QueryBuilder: query_builder_type에 따른 QueryBuilder
    """
    match query_builder_type:
        case "OriginalKeywordsQueryBuilder":
            return OriginalKeywordsQueryBuilder()
        case "OriginalExistKeywordsQueryBuilder":
            return OriginalExistKeywordsQueryBuilder()
        case "CombinedKeyQueryBuilder_pqc":
            return CombinedKeyQueryBuilder(["paragraph", "full_question", "choices_text"])
        case "CombinedKeyQueryBuilder_pq":
            return CombinedKeyQueryBuilder(["paragraph", "full_question"])
        case "CombinedKeyQueryBuilder_p":
            return CombinedKeyQueryBuilder(["paragraph"])
        case "CombinedKeyQueryBuilder_sqc":
            return CombinedKeyQueryBuilder(["summarization", "full_question", "choices_text"])
        case "CombinedKeyQueryBuilder_sq":
            return CombinedKeyQueryBuilder(["summarization", "full_question"])
        case "CombinedKeyQueryBuilder_s":
            return CombinedKeyQueryBuilder(["summarization"])
        case "TFIDFQueryBuilder":
            return TFIDFQueryBuilder()
