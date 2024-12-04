import json
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


def record_right_answer(valid_df, result_df):
    # match: answer 값이 일치하는 id 리스트
    match_ids = valid_df.loc[valid_df["answer"] == result_df["answer"], "id"].tolist()
    # not match: answer 값이 일치하지 않는 id 리스트
    not_match_ids = valid_df.loc[valid_df["answer"] != result_df["answer"], "id"].tolist()

    # 결과를 딕셔너리 형태로 저장
    result_dict = {"match": match_ids, "not_match": not_match_ids}
    save_to_json(result_dict, "rag_data/valid_result/result.json")


def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)
