import evaluate
from transformers import PreTrainedTokenizerFast

from .preprocess import logits_to_predictions, preprocess_labels

# metric 로드
acc_metric = evaluate.load("accuracy")


def compute_metrics(evaluation_result, tokenizer: PreTrainedTokenizerFast):
    """
    평가 지표를 계산하는 함수.

    Args:
        evaluation_result: (logits, labels) 튜플.
        tokenizer: 토크나이저 객체.

    Returns:
        accuracy metric 딕셔너리.
    """
    logits, labels = evaluation_result

    predictions = logits_to_predictions(logits)
    labels = preprocess_labels(labels, tokenizer)

    # 정확도 계산
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc
