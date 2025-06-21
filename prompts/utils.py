import os
import re

import pandas as pd

from .prompt import load_template


def check_prompt_path(prompt_type: str, file_name: str):
    """
    프롬프트 경로를 검사하는 함수.

    Args:
        prompt_type (str): 프롬프트 종류. `prompts/templates/` 디렉토리 아래의 서브 디렉토리 이름.
        file_name (str): 프롬프트 파일명. `prompts/templates/{prompt_type}/` 디렉토리 안의 .txt 파일 이름.

    Raises:
        FileNotFoundError: 지정한 `prompt_type` 디렉토리나 `file_name` 파일이 존재하지 않을 경우 예외를 발생시킴.

    Example:
        check_prompt_path(prompt_type="example_type", file_name="example.txt")
    """
    prompt_root_dir = os.path.join(os.path.dirname(__file__), "templates")
    prompt_type_dir = os.path.join(prompt_root_dir, prompt_type)
    prompt_file = os.path.join(prompt_type_dir, file_name)

    # 프롬프트 타입 디렉토리 존재 여부 확인
    if not os.path.exists(prompt_type_dir):
        raise FileNotFoundError(f"{prompt_type} 디렉토리가 {prompt_root_dir} 경로에 존재하지 않습니다.")

    # 프롬프트 파일 존재 여부 확인
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"{prompt_type_dir} 경로에 {file_name} 파일이 존재하지 않습니다.")


def _detect_input_variables(prompt_type: str, file_name: str) -> list[str]:
    """
    프롬프트에서 입력 변수를 감지하는 함수.
    중괄호(`{}`)로 감싸진 변수명을 추출합니다.

    NOTE: 중괄호 내부의 값이 영문자, 숫자, 또는 밑줄(_)인 경우만 감지됩니다.

    Args:
        prompt_type (str): 프롬프트 종류. `prompts/templates/` 디렉토리 아래의 서브 디렉토리 이름.
        file_name (str): 프롬프트 파일명. `prompts/templates/{prompt_type}/` 디렉토리 안의 .txt 파일 이름.

    Returns:
        list[str]: 감지된 변수명 리스트.

    Example:
        _detect_input_variables(prompt_type="example_type", file_name="example.txt")
        # Output: ['variable1', 'variable2']
    """
    # 템플릿 파일 로드
    prompt = load_template(file_name=file_name, template_type=prompt_type)
    # 중괄호 내부의 변수명 감지
    matches = re.findall(r"\{([a-zA-Z0-9_]+)}(?![^\{]*\})", prompt)
    return matches


def parse_input(data: pd.Series, prompt_type: str, file_name: str) -> dict[str, str]:
    """
    데이터 시리즈와 프롬프트 파일을 기반으로 입력 변수를 파싱하는 함수.

    Args:
        data (pd.Series): 데이터 입력. 변수명과 값의 매핑이 포함된 판다스 시리즈.
        prompt_type (str): 프롬프트 종류. `prompts/templates/` 디렉토리 아래의 서브 디렉토리 이름.
        file_name (str): 프롬프트 파일명. `prompts/templates/{prompt_type}/` 디렉토리 안의 .txt 파일 이름.

    Returns:
        dict[str, str]: 감지된 변수명과 입력 데이터 값을 매핑한 딕셔너리.

    Example:
        data = pd.Series({"variable1": "value1", "variable2": "value2"})
        parse_input(data, prompt_type="example_type", file_name="example.txt")
        # Output: {"variable1": "value1", "variable2": "value2"}
    """
    # 프롬프트에서 필요한 변수 추출
    variables = _detect_input_variables(prompt_type=prompt_type, file_name=file_name)
    # 데이터에서 변수명에 해당하는 값 추출
    return {variable: data[variable] for variable in variables}
