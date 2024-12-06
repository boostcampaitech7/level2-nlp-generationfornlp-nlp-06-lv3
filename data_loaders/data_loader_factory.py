from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.inference.inference_data_loader import InferenceDataLoader
from data_loaders.train.raft_data_loader import RAFTDataLoader
from data_loaders.train.train_cot_data_loader import TrainCotDataLoader
from data_loaders.train.train_data_loader import TrainDataLoader


def build_data_loader(type: str, tokenizer: PreTrainedTokenizerFast, config: Config):
    if type == "train":
        if config.common.cot_on:
            return TrainCotDataLoader(config, tokenizer)
        elif config.rag.raft_on:
            return RAFTDataLoader(config=config, tokenizer=tokenizer)
        else:
            return TrainDataLoader(config, tokenizer)
    elif type == "validation":
        return InferenceDataLoader(config.train.valid_data_path, config, tokenizer)
    elif type == "inference":
        return InferenceDataLoader(config.inference.data_path, config, tokenizer)
    else:
        raise ValueError(f"Invalid data loader type: {type}")
