import re
from typing import List, Union

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_query_builder import query_builder

kiwi = Kiwi()


# Kiwi 기반 커스텀 토크나이저 정의 (명사, 미분류)
def kiwi_tokenizer(text):
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    tokens = kiwi.tokenize(cleaned_text)

    # tokenize 결과 중 일반명사, 고유명사만 추출
    nouns = [token.form for token in tokens if token.tag in ["NNG", "NNP"]]

    # 형태소로 분류되지 않은 것 중 길이가 2 이상인 단어 포함
    additional_candidates = [token.form for token in tokens if len(token.form) >= 2 and token.tag in ["UN"]]
    return list(nouns + additional_candidates)


class tf_idf_query_builder(query_builder):
    def __init__(self, columns: Union[str, List[str]], top_k: int = 5):
        """
        :param top_k: TF-IDF 상위 키워드 개수
        """
        self.columns = columns
        self.top_k = top_k  # 키워드 상위 개수

    def _extract_keywords(self, text: str) -> list[str]:
        """
        TF-IDF로 중요한 키워드 추출.
        :param text: 분석할 텍스트
        :return: 상위 키워드 리스트
        """
        # TF-IDF 모델 생성
        vectorizer = TfidfVectorizer(max_features=50, tokenizer=kiwi_tokenizer)
        tfidf_matrix = vectorizer.fit_transform([text])  # 텍스트 벡터화

        # TF-IDF 점수 추출
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()

        # 점수가 높은 상위 키워드 추출
        top_indices = np.argsort(scores)[-self.top_k :][::-1]  # 높은 점수 순서대로 정렬
        return [feature_names[i] for i in top_indices]

    def build(self, row: pd.Series) -> str:
        """
        TF-IDF 키워드를 활용해 Query 생성.
        """
        # 지문에서 키워드 추출
        text = "".join(map(str, row[self.columns]))
        keywords = self._extract_keywords(text)
        print(text)
        keywords_text = " ".join(keywords)

        return keywords_text
