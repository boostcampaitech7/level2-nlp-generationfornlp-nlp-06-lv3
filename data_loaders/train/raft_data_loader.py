import json

import pandas as pd
from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.train.train_data_loader import TrainDataLoader


class RAFTDataLoader(TrainDataLoader):
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        super().__init__(config, tokenizer)

    def build_single_data(self, data: pd.Series, user_prompt: str):
        len_choices = len(data["choices"])

        return {
            "id": data["id"],
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"answer": str(data["answer"])}, ensure_ascii=False)},
            ],
            # "label": data["answer"], # train, eval 둘다 안쓰임..
            "len_choices": len_choices,
        }
