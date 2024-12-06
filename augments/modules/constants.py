# pd.DataFrame의 column 이름으로 사용되는 상수입니다.
PARAGRAPH = "paragraph"
QUESTION = "question"
CHOICES = "choices"
ANSWER = "answer"
QUESTION_PLUS = "question_plus"
DEFAULT_COLUMNS = ["paragraph", "question", "choices", "answer", "question_plus"]


REASONING = "reasoning"
ANALYSIS = "analysis"
CATEGORY = "category"
VALID = "is_valid"

# 조합하여 column 이름으로 사용되는 상수입니다. ex) keyword_1_exists
KEYWORD_PREFIX = "keyword_"
EXISTS_SUFFIX = "_exists"
PAGE_SUFFIX = "_page"
SUMMARY_SUFFIX = "_summary"

# 프롬프트에 사용되는 변수 명입니다.
KEYWORDS = "keywords"
DOCUMENT = "document"
CRAWLED_TEXT = "crawled_text"
RAW_PARAGRAPH = "raw_paragraph"

# 카테고리 명
NEED_KNOWLEDGE = "외적 추론"
