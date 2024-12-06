import argparse
import json
from ast import literal_eval

import dotenv
import pandas as pd
from langchain.output_parsers import OutputFixingParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import pipeline

from configs import Config
from models import load_model_and_tokenizer
from prompts import load_template
from utils import set_seed


def inference(config: Config, validation: bool):
    if validation:
        df = pd.read_csv(config.train.valid_data_path)
    else:
        df = pd.read_csv(config.inference.data_path)
    df["choices"] = df["choices"].apply(literal_eval)
    df["question_plus"] = df["question_plus"].fillna("")

    train_df = pd.read_csv(config.train.data_path)
    train_df["choices"] = train_df["choices"].apply(literal_eval)
    train_df["question_plus"] = train_df["question_plus"].fillna("")

    chat_prompt_template: ChatPromptTemplate = build_chat_prompt_template(train_df, config)

    model, tokenizer = load_model_and_tokenizer(config.inference.model_path, config)

    llm = HuggingFacePipeline(
        pipeline=pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024 if config.common.cot_on else 32,
            temperature=0.5,
            top_p=0.7,
            repetition_penalty=1.15,
            do_sample=True,
            return_full_text=False,
        )
    )
    chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
    parser = JsonOutputParser()
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    chain = chat_prompt_template | chat_model | fixing_parser

    for i, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True):
        try:
            result = process_row(chain, row)
            reasoning, predict = result["reasoning"], result["predict"]
        except Exception as e:
            print(f"Error: {row['id']} - {str(e)}")
            reasoning, predict = "", config.inference.default_answer
        df.loc[i, "reasoning"] = reasoning
        df.loc[i, "predict"] = predict

    df["predict"] = df["predict"].astype(int)
    df.to_csv(config.inference.raw_output_path, index=False)

    if validation:
        df[["id", "predict"]].rename(columns={"predict": "answer"}).to_csv(config.train.valid_output_path, index=False)
        accuracy = accuracy_score(df["answer"].astype(str), df["predict"].astype(str))
        print("\nFinal Validation results:")
        print(f"Accuracy: {accuracy:4f}")
    else:
        df[["id", "predict"]].rename(columns={"predict": "answer"}).to_csv(config.inference.output_path, index=False)


def build_chat_prompt_template(train_df: pd.DataFrame, config: Config):
    template: str = load_template("no_question_plus.txt", config.common.prompt_template)

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                content=template.format(
                    **build_input(
                        paragraph=train_df.iloc[0]["paragraph"],
                        question=train_df.iloc[0]["question"],
                        question_plus=train_df.iloc[0]["question_plus"],
                        choices=train_df.iloc[0]["choices"],
                    )
                )
            ),
            AIMessage(content=create_ai_response(train_df.iloc[0], config)),
            HumanMessage(
                content=template.format(
                    **build_input(
                        paragraph=train_df.iloc[1]["paragraph"],
                        question=train_df.iloc[1]["question"],
                        question_plus=train_df.iloc[1]["question_plus"],
                        choices=train_df.iloc[1]["choices"],
                    )
                )
            ),
            AIMessage(content=create_ai_response(train_df.iloc[1], config)),
            HumanMessagePromptTemplate.from_template(template),
        ]
    )


def build_input(paragraph, question, choices, question_plus=""):
    question_plus_string = f"\n\n<보기>:\n{question_plus}" if question_plus else ""
    question = f"{question}{question_plus_string}"
    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
    return {
        "paragraph": paragraph,
        "question": question,
        "choices": choices_string,
    }


def create_ai_response(series: pd.Series, config: Config):
    if config.common.cot_on:
        return json.dumps(
            {
                "reasoning": series["reasoning"],
                "answer": str(series["answer"]),
            },
            ensure_ascii=False,
        )
    else:
        return json.dumps(
            {
                "answer": str(series["answer"]),
            },
            ensure_ascii=False,
        )


def process_row(chain: RunnableSerializable, row):
    result = chain.invoke(
        build_input(
            row["paragraph"],
            row["question"],
            row["choices"],
            row["question_plus"],
        )
    )
    answer = int(result["answer"])
    if answer < 1 or len(row["choices"]) < answer:
        raise ValueError(result)
    return {"reasoning": result.get("reasoning", ""), "predict": answer}


if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.yaml")
    parser.add_argument("-v", "--validation", action="store_true")
    args = parser.parse_args()

    try:
        config = Config(args.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {args.config_file}")
        print("Run with default config: config.yaml\n")
        config = Config()

    set_seed(config.common.seed)

    inference(config, validation=args.validation)
