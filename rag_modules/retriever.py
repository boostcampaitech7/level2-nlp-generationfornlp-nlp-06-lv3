from ast import literal_eval

import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore

from utils import get_pinecone_index


def get_retriever(embedding_model: str = "BAAI/bge-m3", k: int = 5, minimal_score: float = 0.4):
    """
    Pinecone에 연결된 Retriever를 생성하고 반환합니다.

    Args:
        embedding_model (str): 임베딩에 사용할 허깅페이스 모델 이름
        k (int): similarity search로 불러올 문서 개수
        minimal_score (float): 유사도 임계값(유사도가 임계값보다 큰 벡터들만 가져오게 됩니다)

    Returns:
        langchain retriever
    """

    # Pinecone index 연결
    pinecone_index, index_name = get_pinecone_index()

    # HuggingFace Embeddings 모델 초기화
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)

    # LangChain의 Pinecone VectorStore 생성
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key="text")

    # Retriever 반환
    return vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": minimal_score}
    )


def read_csv_for_rag_query(file_path: str) -> pd.DataFrame:
    """
    csv 파일을 결측치 처리, str의 List화 등의 전처리를 진행해 query building이 가능한 DataFrame으로 읽어옵니다.

    Args:
        file_path (str): csv 파일을 읽어올 경로

    Returns:
        pd.DataFrame: query building 관련 전처리가 완료된 DataFrame
    """
    df = pd.read_csv(file_path)

    # 결측치 처리 및 List로 type 변환
    df["question_plus"] = df["question_plus"].fillna("")
    df["choices"] = df["choices"].apply(literal_eval)

    # choices를 한 문장으로 연결한 choices_text 생성
    df["choices_text"] = df["choices"].apply(lambda x: " ".join(x))

    # question과 <보기> 를 합친 full question 생성
    df["full_question"] = df.apply(
        lambda row: row["question"] + row["question_plus"] if row["question_plus"] else row["question"],
        axis=1,
    )

    # 파일 내에 keyword 관련 컬럼이 있을 경우, 해당 컬럼들을 전부 엮어 list로 저장
    if "keyword_1" in df.columns:
        df["keywords"] = df[["keyword_1", "keyword_2", "keyword_3", "keyword_4", "keyword_5"]].apply(
            lambda row: row.tolist(), axis=1
        )

    if "keyword_1_exists" in df.columns:
        df["keywords_exists"] = df[
            ["keyword_1_exists", "keyword_2_exists", "keyword_3_exists", "keyword_4_exists", "keyword_5_exists"]
        ].apply(lambda row: row.astype(int).tolist(), axis=1)

    return df


def set_columns_from_config(query_builder_type: int) -> list[str]:
    """
    query builder type에 따라 query building 시 사용되는 column 명 리스트를 반환합니다.

    Args:
        query_builder_type (int): 사전 정의된 query builder type 번호

    Returns:
        list[str]: query builder type에 따른 사용 column명
    """
    match query_builder_type:
        case 1:
            return ["paragraph"]
        case 2:
            return ["paragraph", "full_question"]
        case 3:
            return ["paragraph", "full_question", "choices_text"]
        case 4:
            return ["summarization"]
        case 5:
            return ["summarization", "full_question"]
        case _:
            return ["summarization", "full_question", "choices_text"]
