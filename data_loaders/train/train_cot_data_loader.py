import json

import pandas as pd
from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.train.train_data_loader import TrainDataLoader


class TrainCotDataLoader(TrainDataLoader):
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        super().__init__(config, tokenizer)

    def build_single_data(self, data: pd.Series, user_prompt: str):
        len_choices = len(data["choices"])

        response = json.dumps(
            {
                "reasoning": data["reasoning"],
                "answer": data["answer"],
            },
            ensure_ascii=False,
        )

        return {
            "id": data["id"],
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            "len_choices": len_choices,
        }
