import torch
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# template 전역변수로 설정
template = (
    "당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. "
    "당신의 임무는 주어진 지문(Paragraph)과 선택지(Choices)를 바탕으로 질문(Question)에 가장 적합한 답을 선택하는 것입니다.\n"
    "선택지는 총 {len_choice}개이며, 각 선택지에는 고유한 번호가 있습니다. "
    "답변 형태는 반드시 선택지 번호(1~{len_choice}) 중 하나로만 출력하세요. (예시: 4)\n\n"
    "# Question:\n"
    "{question}\n\n"
    "# Paragraph:\n"
    "{paragraph}\n\n"
    "# Choices:\n"
    "{choices}\n\n"
    "# Support:\n"
    "{support}\n\n"
    "# Answer:\n"
    "# FORMAT:\n"
    "{format}\n"
)


# 출력 형태를 지정해주기 위해 필요합니다.
class Output(BaseModel):
    answer: int = Field(description="The answer to the question. Only single Number (1 ~ 5).")


def create_chain(model_id: str = "google/gemma-2-2b-it", max_new_tokens: int = 256):
    """
    Create and return a chain that combines prompt template, LLM, and output parser.

    Args:
        model_id (str): HuggingFace 모델 ID (e.g., "google/gemma-2-2b-it").
        max_new_tokens (int): LLM의 최대 토큰 수.

    Returns:
        Chain object ready for inference.
    """

    # Initialize HuggingFace LLM
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=gen)

    parser = JsonOutputParser(pydantic_object=Output)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    chat = [{"role": "user", "content": template}]

    prompt_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Create prompt template
    prompt = PromptTemplate.from_template(
        template=prompt_template, partial_variables={"format": parser.get_format_instructions()}
    )

    # Combine into chain
    return prompt | llm | fixing_parser
