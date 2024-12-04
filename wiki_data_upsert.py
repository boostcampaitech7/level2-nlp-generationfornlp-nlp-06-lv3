import csv
import os
import sys
import uuid

from rag_preprocess import generate_embeddings, load_document, split_into_chunks
from tqdm import tqdm

from utils import get_pinecone_index


# 제외할 소제목 이전만 남기는 함수
def clean_dic(data_dict, excluded_subtitles):
    text = data_dict.get("text", "")  # 'text' 필드 가져오기
    for keyword in excluded_subtitles:
        if keyword in text:
            text = text.split(keyword)[0]  # 키워드를 기준으로 자르고 첫 번째 부분만 남김
    data_dict["text"] = text.strip()  # 공백 제거 후 업데이트
    return data_dict


def rag_preprocessing(page_text_path, page_title, page_index):
    """
    텍스트 파일로 생성된 위키 문서를 벡터화하고 pinecone 벡터 db에 업로드 합니다.
    meta data로 설정해줄 제목과 인덱스를 인자로 받습니다.
    Args:
        page_text_path (_str_): 페이지 본문 내용이 들어간 텍스트 파일 경로
        page_title (_str_): meta data로 들어갈 페이지 제목
        page_index (_str_): meta data로 들어갈 페이지 인덱스
    """
    # Load and process document
    documents = load_document(page_text_path)
    chunks = split_into_chunks(documents)
    # print(chunks)
    embeddings = generate_embeddings(chunks, embedding_type="huggingface")

    # Prepare and upsert embeddings with unique IDs
    upsert_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        unique_id = f"{chunk.metadata.get('document_id', 'unknown')}_{i}_{uuid.uuid4()}"

        upsert_data.append(
            {
                "id": unique_id,  # 고유한 ID 생성
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    "document_id": page_title,
                    "chunk_index": i,
                    "page_index": page_index,
                },
            }
        )

    # Upload embeddings

    pinecone_index.upsert(vectors=upsert_data)

    # print(f"Upserted {len(embeddings)} embeddings with metadata to Pinecone index '{index_name}'.")


if __name__ == "__main__":  # noqa: C901
    csv.field_size_limit(sys.maxsize)
    # CSV 파일 경로
    file_path = "data/wiki_contents_categories_from_test_exp_v1.0.3.csv"

    # 제외할 소제목 키워드 정의
    excluded_subtitles = ["각주", "같이 보기", "참고 자료", "외부 링크"]

    # CSV를 딕셔너리로 읽기
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # 각 dict값의 본문에서 필요없는 소제목을 삭제한 뒤 리스트에 추가
            cleaned_row = clean_dic(dict(row), excluded_subtitles)
            data.append(cleaned_row)

    # 소제목 커팅 결과 확인
    for row in data[:5]:  # 첫 5개 데이터 출력
        print("===============")
        print(row["text"])

    for i in range(len(data)):
        data[i]["page_index"] = i

    # 총 읽은 문서 수 확인
    print(data[-1]["page_index"])

    sys.path.append("..")

    raw_data = data

    # Pinecone index에 연결
    pinecone_index, index_name = get_pinecone_index()

    save_path = "data"
    continue_flag = 0

    for i in tqdm(raw_data):
        for key in list(i.keys()):
            if not i[key]:
                continue_flag = 1
                break
        if continue_flag == 1:
            continue_flag = 0
            continue

        text = i["text"]
        page_index = i["page_index"]
        page_title = i["title"]
        file_name = "wiki_text_upsert.txt"
        os.makedirs(save_path, exist_ok=True)  # 디렉터리 없으면 생성
        file_path = os.path.join(save_path, file_name)
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            file.write(text)

        rag_preprocessing(file_path, page_title, page_index)
