import argparse
from ast import literal_eval

import dotenv
import pandas as pd
from tqdm import tqdm

from configs import Config, create_rag_config
from rag_modules import (
    combined_key_query_builder,
    create_chain,
    get_retriever,
    read_csv_for_rag_query,
    set_columns_from_config,
)
from utils import check_valid_score, format_docs, record_right_answer, set_seed


def run_rag_pipeline(data_path: str, config: Config, valid_flag: bool = False):
    """
    전체 RAG pipeline을 실행하는 함수입니다.

    """

    retriever = get_retriever(embedding_model="BAAI/bge-m3", k=5)
    chain = create_chain(model_id="google/gemma-2-2b-it", max_new_tokens=256)

    # config 내 query builder type을 통해 query로 사용할 columns setting
    rag_config = create_rag_config(config.rag)
    columns = set_columns_from_config(rag_config.query_builder_type)
    query_builder = combined_key_query_builder(columns)

    # CSV 파일 로드
    df = read_csv_for_rag_query(file_path=data_path)

    # 결과 저장용 리스트
    results = []

    # Inference
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):

        question_plus_string = ("\n\n<보기>:\n" + row["question_plus"]) if row["question_plus"] else ""
        question = row["question"] + question_plus_string
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

        query = query_builder.build(row)

        retrieved_docs = retriever.invoke(query)

        support = format_docs(retrieved_docs)

        inputs = {
            "len_choice": len(row["choices"]),
            "question": question,
            "paragraph": row["paragraph"],
            "choices": choices_string,
            "support": support,
        }

        try:
            result = chain.invoke(inputs)
            # 결과 추가
            results.append({"id": row["id"], "answer": int(result["answer"])})
        except Exception as e:
            print("Error:", str(e))  # 에러 메시지 출력
            print("in trouble..")
            results.append({"id": row["id"], "answer": -1})

    result_df = pd.DataFrame(results)

    if valid_flag:
        # valid accuracy 확인
        check_valid_score(valid_df=df, result_df=result_df)
        # 정답, 오답 index 정보 저장
        record_right_answer(valid_df=df, result_df=result_df)

        result_df.to_csv("data/valid_output.csv", index=False)
    else:
        result_df.to_csv("data/output.csv", index=False)


if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    try:
        config = Config(args.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {args.config_file}")
        print("Run with default config: config.yaml\n")
        config = Config()

    set_seed(config.common.seed)

    data_path = "data/test_v1.0.2.csv"
    run_rag_pipeline(data_path=data_path, config=config, valid_flag=False)
