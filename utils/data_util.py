from ast import literal_eval

import pandas as pd


def sample_valid_dataset(input_file: str, output_valid_file: str, output_train_file: str, sample_size: int = 400):
    """
    샘플링을 수행하여 train 데이터셋을 분리합니다.

    Args:
        input_file (str): 입력 CSV 파일 경로
        output_valid_file (str): 샘플링된 validation 데이터를 저장할 CSV 파일 경로
        output_train_file (str): 샘플링 후 남은 train 데이터를 저장할 CSV 파일 경로
        sample_size (int): 샘플링할 데이터의 행 수 (기본값 400)
    """
    # Load the train dataset
    df = pd.read_csv(input_file)
    df["choices"] = df["choices"].apply(literal_eval)
    df["question_plus"] = df["question_plus"].fillna("")

    sampled_rows = []
    while len(sampled_rows) < sample_size and not df.empty:
        # 1개 샘플링
        sampled_row = df.sample(n=1, random_state=42)

        # 샘플링한 "paragraph" 값
        paragraph_value = sampled_row.iloc[0]["paragraph"]

        # 동일한 "paragraph" 값을 가진 행들을 모두 추출
        matching_rows = df[df["paragraph"] == paragraph_value]

        # 추출된 행들을 sampled_rows에 추가
        sampled_rows.extend(matching_rows.to_dict("records"))

        # df에서 추출된 행 제거
        df = df[~df["paragraph"].isin(matching_rows["paragraph"])]

    # 샘플링된 데이터와 나머지 데이터 저장
    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df.to_csv(output_valid_file, index=False)
    df.to_csv(output_train_file, index=False)
