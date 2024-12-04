import os

from dotenv import load_dotenv
from pinecone import Pinecone


def get_pinecone_index():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key or not environment or not index_name:
        raise ValueError("Pinecone API_KEY, ENVIRONMENT, or INDEX name is missing from environment variables.")

    # Initialize Pinecone and connect to the existing index
    pc = Pinecone(api_key=api_key, environment=environment)
    pinecone_index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")

    return pinecone_index, index_name


def check_valid_score(valid_df, result_df):
    # answer 열이 일치하는 경우 카운트
    matches = (result_df["answer"] == valid_df["answer"]).sum()

    # 정확도 계산 (일치한 수 / 전체 데이터 수)
    accuracy = matches / len(valid_df)

    print(f"valid accuracy: {accuracy * 100:.2f}%")
    return accuracy


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)
