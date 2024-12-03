import argparse
import os
import shutil

import dotenv
import wandb
from trl import SFTTrainer

from configs import Config, create_peft_config, create_sft_config
from data_loaders import build_data_loader
from evaluation import compute_metrics, preprocess_logits_for_metrics
from models import get_data_collator, load_model_and_tokenizer
from utils import CustomEarlyStoppingCallback, set_seed


def train(config: Config):
    # wandb 초기화
    wandb.init(
        project=config.wandb.project,
        name=(
            f"{config.model.name_or_path.split('/')[1]}/"
            f"{'4bit' if config.bnb.load_in_4bit else ('8bit' if config.bnb.load_in_8bit else 'no_q')}/"
            f"ep{config.sft.num_train_epochs}/"
            f"r{config.peft.r}/al{config.peft.lora_alpha}/"
            f"MSL{config.sft.max_seq_length}/"
            f"lr{config.sft.learning_rate}/"
            f"data-{config.train.data_path.split('_')[1].split('.csv')[0]}"
        ),
        config=config.raw_config,
    )

    model, tokenizer = load_model_and_tokenizer(config.model.name_or_path, config)

    data_loader = build_data_loader("train", tokenizer, config)

    peft_config = create_peft_config(config.peft)
    sft_config = create_sft_config(config.sft)

    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"],
    ]

    early_stopping_callback = CustomEarlyStoppingCallback(
        monitor=config.earlystop.metric_for_best_model,
        patience=config.earlystop.early_stopping_patience,
        threshold=config.earlystop.early_stopping_threshold,
        greater_is_better=config.earlystop.greater_is_better,
    )

    def compute_metrics_fn(eval_res):
        return compute_metrics(eval_res, tokenizer)

    if config.common.cot_on:
        conditional_metrics_fn = None
    elif config.rag.raft_on:
        conditional_metrics_fn = None
    else:
        conditional_metrics_fn = compute_metrics_fn

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        data_collator=get_data_collator(tokenizer, config.model.response_template),
        tokenizer=tokenizer,
        compute_metrics=conditional_metrics_fn,
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, logit_idx),
        peft_config=peft_config,
        args=sft_config,
        callbacks=[early_stopping_callback],
    )
    trainer.train()

    # 학습 종료 후 체크포인트 삭제
    if os.path.exists(sft_config.output_dir):
        print(f"Deleting checkpoints in {sft_config.output_dir}...")
        shutil.rmtree(sft_config.output_dir)  # 체크포인트 디렉토리 삭제

    # 최종 모델 저장
    save_path = f"outputs/{config.model.name_or_path.split('/')[1]}"
    print(f"Saving final model to {save_path}...")
    trainer.save_model(save_path)


if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    try:
        config = Config(args.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {args.config_file}")
        print("Run with default config: config.yaml\n")
        config = Config()

    set_seed(config.common.seed)

    train(config)
