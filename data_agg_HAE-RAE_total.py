import re

import pandas as pd
from tqdm import tqdm


# 저장할 파일명을 선언해준다
def get_category_initials(name):
    name_split = name.split("_")
    initials = ""
    for i in name_split:
        initials += i[0]
    return initials


# 정규표현식으로 특정 패턴의 텍스트를 추출하는 함수를 선언해준다
def extract_before(text, keyword):
    pattern = f"(.*){re.escape(keyword)}"  # 키워드 이전 모든 문자열 매칭
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 키워드 이전 문자열 반환
    return None  # 키워드가 없으면 None 반환


def extract_after(text, keyword):
    pattern = f"{re.escape(keyword)}(.*)"  # 키워드 이후 모든 문자열 매칭
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 키워드 이후 문자열 반환
    return None  # 키워드가 없으면 None 반환


def extract_between(text, start_keyword, end_keyword):
    # 시작과 끝 키워드 사이의 텍스트 추출
    pattern = f"{re.escape(start_keyword)}(.*?){re.escape(end_keyword)}"
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL은 줄바꿈 포함 매칭
    if match:
        return match.group(1).strip()
    return None


# 알파벳을 숫자로 맵핑하는 함수를 만들어준다 (answer 칼럼용)
def convert_string_to_number(s):
    # 문자열-숫자 매핑 딕셔너리
    mapping = {"(A)": 1, "(B)": 2, "(C)": 3, "(D)": 4, "(E)": 5}
    # 매핑에 없는 문자열은 None 반환
    return mapping.get(s, None)


# HAERAE 데이터를 모두 넣어줄 빈 데이터셋을 선언한다
train_agg = pd.read_csv("./data/agg_other_benchmarks/train_agg.csv")
data_haerae_agg = pd.DataFrame(columns=train_agg.columns)


# HAERAE 데이터셋 선별 데이터 취합 (agg = aggregation)
agg_categories = [
    "reading_comprehension",
    "standard_nomenclature",
    "correct_definition_matching",
    "general_knowledge",
    "history",
    "rare_words",
]  # 취합할 데이터셋에 맞춰 수정할 것

if __name__ == "__main__":  # noqa: C901
    datasets_paths = []
    for i in agg_categories:
        default_path = "./data/agg_other_benchmarks/to_be_agg/"
        dataset_file_name = "HAE-RAE_" + i + ".csv"
        dataset_path = default_path + dataset_file_name
        datasets_paths.append(dataset_path)

    ids = []
    for i in agg_categories:
        default_name = i.replace("_", "-")
        id = "haerae-" + default_name + "-"
        ids.append(id)

    for path_index in range(len(datasets_paths)):
        dataset = pd.read_csv(datasets_paths[path_index])
        dataset_name = agg_categories[path_index]
        count = 0
        for i in tqdm(range(len(dataset)), desc=f"{dataset_name}"):
            id = f"{ids[path_index]}{count}"
            query = dataset.iloc[i]["query"]

            # HAE-RAE의 각 데이터셋 특성에 맞게 가공해준다.
            if dataset_name == "reading_comprehension":
                passage = extract_between(query, "### 지문:", "### 질문:")
                question = extract_between(query, "### 질문:", "### 선택지:")

            elif dataset_name == "correct_definition_matching":
                passage = extract_between(query, "### 문장:", "### 선택지:")
                question = extract_before(query, "### 문장:")
                question = question.replace("다음", "위")

            elif dataset_name == "general_knowledge":
                passage = extract_between(query, "### 질문:", "### 참고:")
                question = extract_before(query, "### 질문:")
                question = question.replace("다음", "위")
                question_plus = extract_between(query, "### 참고:", "### 선택지:")

            else:
                passage = extract_between(query, "### 질문:", "### 선택지:")
                question = extract_before(query, "### 질문:")
                question = question.replace("다음", "위")

            choices = str(dataset.iloc[i]["options"])
            answer = convert_string_to_number(dataset.iloc[i]["answer"])

            if dataset_name == "general_knowledge":
                df_agg = pd.DataFrame(
                    [
                        {
                            "id": id,
                            "paragraph": passage,
                            "question": question,
                            "choices": choices,
                            "answer": answer,
                            "question_plus": question_plus,
                        }
                    ]
                )
            else:
                df_agg = pd.DataFrame(
                    [
                        {
                            "id": id,
                            "paragraph": passage,
                            "question": question,
                            "choices": choices,
                            "answer": answer,
                            "question_plus": None,
                        }
                    ]
                )

            data_haerae_agg = pd.concat([data_haerae_agg, df_agg])
            print(len(data_haerae_agg))
            count += 1

    final_dataset = pd.concat([train_agg, data_haerae_agg])

    final_file_name = "train_agg_haerae"

    for i in agg_categories:
        final_file_name += "_" + get_category_initials(i)

    # 파일을 생성한다
    final_file_name += ".csv"
    final_dataset.to_csv(final_file_name, index=False)
