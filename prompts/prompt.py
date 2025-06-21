import os

import pandas as pd


def make_prompt(row: pd.Series, template_type: str = "base") -> str:
    """
    주어진 데이터와 템플릿 유형에 따라 프롬프트를 생성합니다.

    Args:
        row (pd.Series):
            프롬프트를 생성하는 데 필요한 데이터를 포함한 pandas Series 객체.
            예: "paragraph", "question", "choices", "question_plus" 키 포함.
        template_type (str):
            템플릿이 위치한 디렉토리의 유형을 나타내는 문자열.

    Returns:
        str:
            생성된 프롬프트 문자열.
    """
    template_name = "question_plus.txt" if row["question_plus"] else "no_question_plus.txt"
    template = load_template(template_name, template_type)

    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

    return template.format(
        paragraph=row["paragraph"],
        question=row["question"],
        choices=choices_string,
        question_plus=row.get("question_plus", ""),
        support=row.get("support", ""),
        len_choice=len(row["choices"]),
    )


def load_template(file_name: str, template_type: str) -> str:
    """
    지정된 파일 이름과 템플릿 유형에 따라 템플릿 파일을 로드합니다.

    Args:
        file_name (str):
            로드할 템플릿 파일의 이름 (예: "question_plus.txt").
        template_type (str):
            템플릿이 위치한 디렉토리 유형을 나타내는 문자열.

    Raises:
        FileNotFoundError:
            지정된 파일이 존재하지 않을 경우 발생.

    Returns:
        str:
            템플릿 파일의 내용을 문자열로 반환.
    """
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "templates", template_type, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Template file '{file_name}' not found.")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()
