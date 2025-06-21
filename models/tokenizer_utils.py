from transformers import PreTrainedTokenizerFast
from trl import DataCollatorForCompletionOnlyLM


def get_data_collator(tokenizer: PreTrainedTokenizerFast, response_template: str):
    """
    데이터를 모델에 입력하기 위해 처리하는 데이터 콜레이터를 반환합니다.

    응답 템플릿을 사용하여 모델이 생성해야 할 응답 부분만을 학습합니다.

    Args:
        tokenizer (PreTrainedTokenizerFast): 텍스트 데이터를 토큰화할 토크나이저.

    Returns:
        DataCollatorForLanguageModeling: 언어 모델링용 데이터 콜레이터.
    """
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )


def set_chat_template(tokenizer: PreTrainedTokenizerFast):
    """
    토크나이저에 채팅 템플릿을 설정하는 함수입니다.

    Args:
        tokenizer: 업데이트할 토크나이저.

    Returns:
        tokenizer: 업데이트된 토크나이저.
    """
    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        # chat_template 속성이 없거나 비어 있으면 기본 템플릿 설정
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
            "{% if system_message is defined %}{{ system_message }}{% endif %}"
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
        )

    return tokenizer
