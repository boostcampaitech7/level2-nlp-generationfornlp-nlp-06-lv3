from ast import literal_eval

import pandas as pd
from tqdm import tqdm

from rag_modules import create_chain, get_retriever
from utils import check_valid_score, format_docs, record_right_answer, set_seed


def run_rag_pipeline():
    """
    전체 RAG pipeline을 실행하는 함수입니다.

    """

    retriever = get_retriever(embedding_model="BAAI/bge-m3", k=5)
    chain = create_chain(model_id="google/gemma-2-2b-it", max_new_tokens=256)

    # CSV 파일 로드
    df = pd.read_csv("data/valid_v2.0.1.csv")

    df["choices"] = df["choices"].apply(literal_eval)
    df["question_plus"] = df["question_plus"].fillna("")

    # 결과 저장용 리스트
    results = []

    # Inference
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):

        question_plus_string = ("\n\n<보기>:\n" + row["question_plus"]) if row["question_plus"] else ""
        question = row["question"] + question_plus_string
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

        retrieved_docs = retriever.invoke(row["paragraph"])

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

    # valid accuracy 확인
    check_valid_score(valid_df=df, result_df=result_df)
    # 정답, 오답 index 정보 저장
    record_right_answer(valid_df=df, result_df=result_df)

    result_df.to_csv("data/valid_output.csv", index=False)


if __name__ == "__main__":
    set_seed(42)
    run_rag_pipeline()
