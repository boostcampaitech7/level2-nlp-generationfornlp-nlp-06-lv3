import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutput


def predict(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, dataset: Dataset):
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]
            label = data.get("label")

            outputs: CausalLMOutput = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

            probs = (
                torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            result = {"id": _id, "answer": predict_value}
            if label is not None:
                result["label"] = label
            infer_results.append(result)

    return pd.DataFrame(infer_results)
