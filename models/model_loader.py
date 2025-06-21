from unsloth import FastLanguageModel

from models.tokenizer_utils import set_chat_template
from utils import str_to_dtype


def load_model_and_tokenizer(model_name_or_path, config):
    """
    모델과 토크나이저를 로드하는 함수입니다.

    Args:
        model_name_or_path (str): 사전 학습된 모델의 경로 또는 이름.
        config (Config): 설정 객체.

    Returns:
        model: 로드된 모델.
        tokenizer: 로드된 토크나이저.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=config.sft.max_seq_length,
        dtype=str_to_dtype(config.model.torch_dtype),
        load_in_4bit=config.bnb.load_in_4bit,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = set_chat_template(tokenizer)

    return model, tokenizer
