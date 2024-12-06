import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import ANSWER, CHOICES, PARAGRAPH, QUESTION, VALID
from .langchain_manager import LangchainManager


class KoBESTAugment(BaseProcessor):
    """
    KoBEST 데이터 증강을 위한 클래스.

    Hugging Face Hub에서 데이터를 로드하여 문제 및 선택지를 생성하고,
    LangchainManager를 활용하여 생성된 데이터를 검증한 후 유효한 데이터만 유지합니다.
    """

    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        """
        KoBESTAugment 클래스 초기화 함수.

        Args:
            data_path (str): 기본 데이터 경로. 기본값은 "data/".
            experiment_data_path (str): 실험 데이터 저장 경로. 기본값은 "data/experiments/".
            data (pd.DataFrame, optional): 입력 데이터프레임. 기본값은 None.
                데이터가 None인 경우 Hugging Face Hub에서 데이터를 로드.

        Raises:
            TypeError: 데이터가 pandas.DataFrame 형식이 아닌 경우 예외를 발생시킴.

        Attributes:
            source_data (pd.DataFrame): 로드되거나 전달된 소스 데이터.
            langchain_manager (LangchainManager): 문제 및 선택지 생성을 위한 LangChain 기반 매니저.
            validator (LangchainManager): 생성된 데이터를 검증하기 위한 LangChain 기반 매니저.
        """
        super().__init__(data_path, experiment_data_path)

        if data is None:
            # HuggingFace Hub에서 KoBEST 데이터셋 로드
            # https://huggingface.co/datasets/skt/kobest_v1
            dataset = load_dataset("skt/kobest_v1", "boolq")
            train_df = pd.DataFrame(dataset["train"])
            valid_df = pd.DataFrame(dataset["validation"])
            test_df = pd.DataFrame(dataset["test"])
            self.source_data = pd.concat([train_df, valid_df, test_df])
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data는 pandas.DataFrame 형식이어야 합니다.")
        else:
            self.source_data = data

        # LangChainManager 설정 - 문제 및 선택지 생성을 위한 프롬프트 설정
        self.langchain_manager = LangchainManager(
            prompt_type="question_choices_generation",
            prompt_source="question_choices_generation.txt",
            output_type="json",
        )

        # LangChainManager 설정 - 생성된 데이터 검증을 위한 프롬프트 설정
        self.validator = LangchainManager(
            prompt_type="question_choices_generation", prompt_source="validation.txt", output_type="bool"
        )

    def generate_problem(self, data: pd.Series):
        """
        입력 데이터에서 문제, 선택지, 정답을 생성하는 함수.

        Args:
            data (pd.Series): 데이터프레임의 한 행(row).

        Returns:
            tuple: 생성된 문제(question), 선택지(choices), 정답(answer).

        Example:
        ```python
            data = pd.Series({"PARAGRAPH": "This is a sample paragraph."})
            question, choices, answer = augmentor.generate_problem(data)
        ```
        """
        response = self.langchain_manager.invoke({PARAGRAPH: data[PARAGRAPH]})
        return response[QUESTION], response[CHOICES], response[ANSWER]

    def process(self):
        """
        데이터 처리 함수.

        문제 및 선택지를 생성하고, 검증을 통해 유효한 데이터만 필터링합니다.

        Steps:
            1. LangChain을 활용하여 문제, 선택지, 정답 생성.
            2. 생성된 결과를 데이터프레임에 추가.
            3. 생성 데이터 검증 및 유효 데이터 필터링.
            4. 불필요한 열 삭제.

        Example:
        ```python
            augmentor = KoBESTAugment()
            augmentor.process()
        ```
        """
        tqdm.pandas()

        # 문제, 선택지, 정답 생성
        self.source_data["result"] = self.source_data.progress_apply(self.generate_problem, axis=1)

        # 생성된 결과 분리하여 열 추가
        self.source_data[QUESTION] = self.source_data["result"].apply(lambda x: x[0])
        self.source_data[CHOICES] = self.source_data["result"].apply(lambda x: x[1])
        self.source_data[ANSWER] = self.source_data["result"].apply(lambda x: x[2])

        # 임시 열 삭제
        self.source_data.drop(columns=["result", "label"], inplace=True)

        # 생성 데이터 검증
        self.source_data[VALID] = self.source_data.progress_apply(
            lambda row: self.validator.invoke(row.to_dict()), axis=1
        )

        # 유효한 데이터만 유지
        self.source_data = self.source_data[self.source_data[VALID]]
