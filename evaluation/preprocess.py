import numpy as np
import torch
from transformers import PreTrainedTokenizerFast


def preprocess_logits_for_metrics(logits, labels, logit_idx: list[int]):
    """
    모델의 logits를 조정하여 정답 토큰 부분만 출력하도록 설정하는 함수.

    Args:
        logits: 모델의 출력 logits.
        tokenizer: 토크나이저 객체.

    Returns:
        조정된 logits 텐서.
    """
    # 모델에 따라 로짓이 튜플인 경우 첫 번째 요소인 logits만 선택
    logits = logits if not isinstance(logits, tuple) else logits[0]

    # 평가에 필요한 로짓만 선택 (예: 마지막에서 두 번째 토큰의 특정 인덱스)
    logits = logits[:, -3, logit_idx]  # -3: answer token, -2: end token, -1: eos token
    return logits


def logits_to_predictions(logits):
    """
    logits를 확률로 변환하는 함수.

    Args:
        logits: 모델의 출력 logits(1, 2, 3, 4, 5에 대해서만 존재).

    Returns:
        최종 예측된 예측값(1 ~ 5).
    """
    # logits를 소프트맥스 함수로 확률 값으로 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

    # 가장 높은 확률 값을 가진 인덱스를 추출
    predicted_indices = np.argmax(probs, axis=-1).tolist()

    # 인덱스 값(0~4)을 문자열 형태의 예측값(1~5)으로 변환
    prediction_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    predictions = [prediction_map[idx] for idx in predicted_indices]
    return predictions


def preprocess_labels(labels, tokenizer: PreTrainedTokenizerFast):
    """
    레이블 데이터를 전처리하는 함수.

    Args:
        labels: 레이블 데이터.
        tokenizer: 토크나이저 객체.

    Returns:
        디코딩된 레이블 데이터.
    """
    # -100 값을 tokenizer의 pad_token_id로 변경
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 토큰 ID를 문자열로 디코딩
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 불필요한 공백 제거
    labels = [x.strip() for x in labels]
    return labels
