import csv
import os
import sys
import uuid

import wikipediaapi
from tqdm import tqdm

from rag_preprocess import generate_embeddings, load_document, split_into_chunks
from utils import get_pinecone_index

csv.field_size_limit(sys.maxsize)
# CSV 파일 경로
file_path = "data/wiki_contents_categories_from_test_exp_v1.0.3.csv"

# CSV를 딕셔너리로 읽기
data = []
with open(file_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(dict(row))

for i in range(len(data)):
    data[i]["page_index"] = i
    print(data[i]["page_index"])


sys.path.append("..")


raw_data = data


def rag_preprocessing(page_title, page_index):
    # Pinecone index에 연결
    pinecone_index, index_name = get_pinecone_index()

    # Load and process document
    file_path = "data/wiki_text_upsert.txt"
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    print(chunks)
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


save_path = "data"
continue_flag = 0

for i in tqdm(raw_data[400:800], desc="400~800"):
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

    rag_preprocessing(page_title, page_index)

for i in tqdm(raw_data[:400], desc="~400"):
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

    rag_preprocessing(page_title, page_index)

for i in tqdm(raw_data[800:1200], desc="800~1200"):
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

    rag_preprocessing(page_title, page_index)

for i in tqdm(raw_data[1200:1600], desc="1200~1600"):
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

    rag_preprocessing(page_title, page_index)

for i in tqdm(raw_data[1600:], desc="1600~"):
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

    rag_preprocessing(page_title, page_index)
