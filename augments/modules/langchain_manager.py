import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import BooleanOutputParser, OutputFixingParser
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from prompts import check_prompt_path, load_template, parse_input

# 출력 유형 상수 정의
STR = "str"
JSON = "json"
BOOL = "bool"


class LangchainManager:
    """
    LangChain 기반 데이터 처리 매니저.

    다양한 출력 형식(str, json, bool)에 따라 체인을 구성하고 데이터를 처리합니다.
    """

    VALID_OUTPUT_TYPES = {STR, JSON, BOOL}

    def __init__(self, prompt_type: str, prompt_source: str, output_type: str):
        """
        LangchainManager 초기화 함수.

        Args:
            prompt_type (str): 프롬프트 종류. `prompts/templates/` 디렉토리 아래의 서브 디렉토리 이름.
            prompt_source (str): 프롬프트 파일명. `prompts/templates/{prompt_type}/` 디렉토리 안의 .txt 파일 이름.
            output_type (str): 출력 형식. "str", "json", "bool" 중 하나.

        Raises:
            ValueError: 출력 형식이 유효하지 않은 경우 예외를 발생시킴.

        Attributes:
            prompt_type (str): 프롬프트 종류.
            prompt_source (str): 프롬프트 파일명.
            output_type (str): 출력 형식.
            llm (ChatOpenAI): LLM 모델 인스턴스.
            chain (RunnableSerializable): 구성된 체인 객체.
        """
        # 환경 변수 로드
        load_dotenv()

        # 프롬프트 경로 및 출력 유형 검증
        check_prompt_path(prompt_type=prompt_type, file_name=prompt_source)
        if output_type not in self.VALID_OUTPUT_TYPES:
            raise ValueError(f"Invalid output_type. Must be one of {self.VALID_OUTPUT_TYPES}")

        # 속성 초기화
        self.prompt_type = prompt_type
        self.prompt_source = prompt_source
        self.output_type = output_type
        self.llm = self._initialize_model("gpt-4o-mini", 0, None, None, 2)

        # 체인 설정
        self.chain = self._build_chain()

    @staticmethod
    def _initialize_model(model, temperature, max_tokens, timeout, max_retries) -> ChatOpenAI:
        """
        LLM 모델 초기화.

        Args:
            model (str): 모델 이름.
            temperature (float): 샘플링 온도.
            max_tokens (int): 최대 토큰 수.
            timeout (int): 요청 제한 시간.
            max_retries (int): 최대 재시도 횟수.

        Returns:
            ChatOpenAI: 초기화된 LLM 인스턴스.
        """
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_chain(self) -> RunnableSerializable:
        """
        체인 빌드 함수.

        프롬프트 템플릿과 출력 파서를 조합하여 체인을 구성합니다.

        Returns:
            RunnableSerializable: 구성된 체인 객체.
        """
        prompt_template = PromptTemplate.from_template(
            template=load_template(file_name=self.prompt_source, template_type=self.prompt_type)
        )

        # 출력 유형에 따른 체인 구성
        parser = self._get_output_parser()
        return prompt_template | self.llm | parser if parser else prompt_template | self.llm

    def _get_output_parser(self):
        """
        출력 유형에 따른 파서 반환.

        Returns:
            OutputFixingParser | None: 출력 형식에 따른 파서 객체. 문자열 출력일 경우 None 반환.
        """
        if self.output_type == JSON:
            return OutputFixingParser.from_llm(parser=JsonOutputParser(), llm=self.llm)
        elif self.output_type == BOOL:
            return OutputFixingParser.from_llm(parser=BooleanOutputParser(), llm=self.llm)
        return None

    def invoke(self, data: pd.Series) -> bool | dict[str, str] | str:
        """
        입력 데이터를 체인을 통해 처리하고 결과 반환.

        Args:
            data (pd.Series): 입력 데이터. 처리에 필요한 데이터를 포함.

        Returns:
            bool | dict[str, str] | str: 체인 처리 결과. 출력 형식에 따라 반환값이 달라짐.

        Raises:
            TypeError: 출력 결과가 기대한 형식과 다른 경우 예외를 발생시킴.
        """
        input_contents = parse_input(data=data, prompt_type=self.prompt_type, file_name=self.prompt_source)
        self.response: bool | dict[str, str] | BaseMessage = self.chain.invoke(input=input_contents)
        self._check_chain_output()
        return self.response.content if self.output_type == STR else self.response

    def _check_chain_output(self):
        """
        체인의 출력 결과 형식 검증.

        Raises:
            TypeError: 출력 결과가 기대한 형식과 다른 경우 예외를 발생시킴.
        """
        if (
            (self.output_type == STR and not isinstance(self.response, BaseMessage))
            or (self.output_type == JSON and not isinstance(self.response, dict))
            or (self.output_type == BOOL and not isinstance(self.response, bool))
        ):
            raise TypeError(f"Wrong type: {type(self.response)} is invoked from chain")
