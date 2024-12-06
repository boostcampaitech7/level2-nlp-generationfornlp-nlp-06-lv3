from abc import ABC, abstractmethod
from ast import literal_eval

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from configs import Config
from prompts import make_prompt


class BaseDataLoader(ABC):
    def __init__(self, data_path: str, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.config = config

        df = self.read_csv(data_path)

        self.dataset = self.preprocess_dataset(df)

    def read_csv(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        df["choices"] = df["choices"].apply(literal_eval)
        df["question_plus"] = df["question_plus"].fillna("")
        return df

    def preprocess_dataset(self, df: pd.DataFrame) -> Dataset:
        processed_dataset = []
        for i, row in df.iterrows():
            user_prompt = make_prompt(row, self.config.common.prompt_template)

            processed_dataset.append(self.build_single_data(row, user_prompt))
        return Dataset.from_list(processed_dataset)

    @abstractmethod
    def build_single_data(self, data: pd.Series, user_prompt: str):
        """
        각 행의 데이터를 처리하는 추상 메서드. 서브클래스에서 구현 필요.

        Args:
            data (pd.Series): 데이터프레임의 행 데이터.
            user_prompt (str): 사용자 프롬프트.

        Returns:
            dict: 처리된 데이터.
        """
        pass
