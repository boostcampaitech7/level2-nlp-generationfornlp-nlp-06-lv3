def preprocess_logits_for_metrics(logits, labels, logit_idx: list[int]):
    """
    모델의 logits를 조정하여 정답 토큰 부분만 출력하도록 설정하는 함수.

    Args:
        logits: 모델의 출력 logits.
        labels: 실제 레이블 (사용하지 않음).
        tokenizer: 토크나이저 객체.

    Returns:
        조정된 logits 텐서.
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits
