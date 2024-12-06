# 수능형 문제 풀이 모델

![메인 정적 화면.png](/assets/banner.png)

한국어의 특성과 수능 시험의 특징을 바탕으로 **수능에 최적화된 모델**을 만듭니다.

## 📌 프로젝트 개요

- **목표**: 수능 문제 풀이를 위한 최적화된 언어 모델 연구 및 개발
- **주요 내용**:
  - 다양한 모델 및 하이퍼파라미터 성능 비교
  - 데이터 정제 및 증강을 통한 학습 효율 향상
  - CoT (Chain of Thought) 방식 적용에 따른 추론 능력 평가
  - RAG (Retrieval-Augmented Generation) 활용으로 검색 기반 답변 성능 실험

## 🏅 최종 결과

`4bit` 양자화를 적용한 `Qwen2.5-32B-Instruct` 모델을 활용하여 **0.7747의 정확도를 달성**했습니다.

| 모델                                | 조건                                    | Accuracy   |
| ----------------------------------- | --------------------------------------- | ---------- |
| finetuned gemma-2b-ko (base)        | 기본 설정                               | 0.3862     |
| finetuned gemma-2b-ko               | 데이터 정제 및 증강                     | 0.4138     |
| finetuned Qwen-2.5-32b-Instruct     | 기본 설정                               | 0.7540     |
| finetuned Qwen-2.5-32b-Instruct     | 데이터 정제 및 증강                     | 0.7632     |
| **finetuned Qwen-2.5-32b-Instruct** | **데이터 정제 및 증강 + Prompt Tuning** | **0.7747** |

위 표는 수능형 문제 풀이 성능을 모델과 실험 조건에 따라 비교한 결과입니다.

베이스 모델인 `gemma-2b-ko`에서 데이터 정제 및 증강으로 소폭 성능 향상을 확인하였으며, 이는 최종 선정된 모델인 `Qwen-2.5-32b-Instruct`에서도 확인 할수 있었습니다. 최종적으로 데이터 증강 및 Prompt Tuning을 추가한 **Qwen-2.5-32b-Instruct** 모델이 **0.7747**로 가장 높은 정확도를 달성했습니다.

> 프로젝트 랩업 리포트는 [여기](<assets/Generation_for_NLP_NLP팀%20리포트(06조).pdf>)를 참고해주세요.

## 👥 Collaborators

<div align="center">

|                                                   팀원                                                    |                                  역할                                  |
| :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|     <a href="https://github.com/gsgh3016"><img src="https://github.com/gsgh3016.png" width="100"></a>     |  Streamlit app 개발 참여, 데이터 관찰 및 분석, 데이터 재구성 및 증강   |
|       <a href="https://github.com/eyeol"> <img src="https://github.com/eyeol.png" width="100"></a>        |             Streamlit app 개발 참여, RAG 구현 및 성능 평가             |
|    <a href="https://github.com/jagaldol"> <img src="https://github.com/jagaldol.png" width="100"> </a>    |  협업 초기 환경 세팅 및 코드 모듈화, CoT 방식 실험 설계 및 성능 평가   |
|     <a href="https://github.com/Usunwoo"> <img src="https://github.com/Usunwoo.png" width="100"> </a>     |        베이스라인 모듈화, 메모리 사용 최적화, 모델 서치 및 실험        |
| <a href="https://github.com/canolayoo78"> <img src="https://github.com/canolayoo78.png" width="100"> </a> |  Streamlit app 개발 참여, 데이터 분석 및 정제, RAG 구현 및 성능 평가   |
|   <a href="https://github.com/chell9999"> <img src="https://github.com/chell9999.png" width="100"> </a>   | 문서 작업, RAG 전용 Vector DB 구성, 벤치마크 데이터셋 기반 데이터 증강 |

</div>

## 🛠️ Tools and Technologies

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![jupyter](https://img.shields.io/badge/-jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![huggingface](https://img.shields.io/badge/-huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

![unsloth](https://img.shields.io/badge/-unsloth-14B789?style=for-the-badge&logo=unsloth&logoColor=white)
![BitsandBytes](https://img.shields.io/badge/BitsandBytes-36474F?style=for-the-badge&logo=BitsandBytes&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-40B5A4?style=for-the-badge&logo=LoRA&logoColor=white)
![langchain](https://img.shields.io/badge/-langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

![RAG](https://img.shields.io/badge/RAG-1868F2?style=for-the-badge&logo=RAG&logoColor=white)
![pinecone](https://img.shields.io/badge/pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![Cot](https://img.shields.io/badge/cot-535051?style=for-the-badge&logo=cot&logoColor=white)
![github action](https://img.shields.io/badge/GITHUB%20ACTIONS-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

</div>

## 📊 다양한 모델 및 하이퍼파라미터 성능 비교

다양한 모델과 하이퍼파라미터 설정을 기반으로 수능 문제 풀이 성능을 평가하였습니다. 주요 실험은 아래와 같습니다:

### 모델별 성능 비교

여러 언어 모델(gemma, Qwen, Llama, SOLAR, EXAONE 등)의 성능을 비교한 결과입니다:

![model compare](assets/model-compare.png)

- `Qwen-2.5-32b-Instruct` 모델이 가장 높은 성능을 기록하였습니다.
- 비슷한 파라미터 크기 내에에서 `Qwen` 계열의 모델이 전반적으로 우수한 성능을 보였습니다.

### Qwen 모델의 파라미터 크기별 성능 비교

`Qwen` 모델의 파라미터 크기(3B, 7B, 14B, 32B)에 따른 성능 변화입니다:

![qwen_parameter compare](assets/qwen-parameter-compare.png)

- 14B과 32B의 경우 4bit 양자화가 적용되었습니다.
- 양자화를 적용함했음에도 파라미터 크기가 커질수록 성능이 크게 향상되는 것을 확인 할 수 있었습니다.

### 하이퍼파라미터 설정별 성능 비교

학습에 사용되는 `LORA r`, `LORA alpha` 조합에 따른 성능 변화를 평가하였습니다:

![hyperparameter compare](assets/hyperparameter-compare.png)

- Mid ACC와 Final ACC 모두에서 `r: 8, alpha: 16` 설정이 가장 높은 성능을 기록하였습니다.

## 💾 데이터 정제 및 증강

### 데이터 정제

전체 학습 데이터 2031개에서 문제 오류 4개를 삭제하고 잘못된 정답 6개를 수정했습니다. 또한, 지문에 보기가 포함된 데이터를 보기 컬럼으로 분리하였으며, 학습에 방해되는 기사 데이터의 기자 이름 및 연락처를 삭제했습니다.

### 데이터 증강

#### 외부 지식이 필요한 문제에 대한 재구성 및 증강

![augmentation pipeline](/assets/augmentation-pipeline.png)

외부 지식이 필요한 문제를 해결하기 위해 기존 지문에 데이터를 보강하여 데이터를 증강하는 방법을 시도했습니다.

1. **핵심 키워드 추출**: 문제 해결에 필요한 주요 정보를 도출
2. **관련 지문 확보**: 추출된 키워드를 기반으로 한국어 위키피디아에서 관련 문서를 크롤링
3. **지문 보강**: 확보된 문서를 지문에 추가하고, 요약된 문서를 취합하여 수능 국어 비문학 지문 형식으로 보강

그러나 모델 학습 결과, 성능 향상에는 실패했으며 이는 일부 수능형 문제조차 외부 지식에 크게 의존함을 시사합니다.

이 발견을 바탕으로, RAG (정보 검색 기반 생성) 접근법을 도입하는 방향으로 이어졌습니다.

#### 타 벤치마크 데이터셋을 통한 데이터 증강

KMMLU 벤치마크 논문에서 (Son et al., 2024) 언급된 다양한 타 벤치마크의 데이터셋을 분석하여 학습 데이터에 적합한 데이터를 선정하였습니다.

![kmmlu-dataset](/assets/kmmlu-dataset.png)

- **KLUE, KO-H5, KOR NAT:** Public으로 공개되지 않음
- **KorNLI & KorSTS, KoBBQ**: 다지선다형 문제로 변환하기 어려움
- **CLICK, HAE-RAE Bench.**: 다지선다형으로 가공이 용이함

분석 결과, **HAE-RAE Bench. 데이터셋이 가장 적합한 선택**으로 확인되었습니다. 특히, **독해 카테고리의 데이터는 Task와 높은 연관성을 보여** 성능 개선에 기여했습니다.

이를 통해 단순히 데이터를 증강하는 것보다, **데이터의 품질과 Task와의 연관성**이 성능에 핵심적인 영향을 미친다는 점을 확인할 수 있었습니다.

또한 skt/KoBEST 데이터셋과 지문 기반 수능 국어 맞춤 문제 생성 논문을 (허동석 외, 2024) 기반으로 질문과 선택지, 정답을 생성했습니다.

> 더 자세한 데이터 증강 실험 내용은 [여기](https://gamchan.notion.site/146815b39d39807884f1f785c2829da6?v=086629b8bb6f49a5a1b54a0ec44d6630&pvs=4)를 참고해주세요.

## 🗨️ Prompt Tuning

프롬프트의 구성 방식에 따라 모델의 성능 차이를 비교했습니다.
단순한 기본 프롬프트, 명확한 규칙 제공, 그리고 영어로 감정적 호소와 함께 명확한 규칙을 전달한 경우의 성능을 평가하였습니다.

### 프롬프트 구성

- **기본 프롬프트**

  ```
  지문을 읽고 질문의 답을 구하세요.
  ```

- **명확한 규칙 제공**

  ```
  당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
  당신의 임무는 주어진 지문(Paragraph)을 바탕으로, 질문(Question)에 가장 적합한 답변을 선택지(Choices)에서 선택하는 것입니다.
  선택지(Choices)는 총 5개이며, 각 선택지(Choices)에는 고유한 선택지 번호(1~5)가 있습니다.
  답변 형태는 반드시 선택지 번호(1~5) 중 하나로만 출력하세요. (예시: 4)
  ```

- **감정적 호소(In English)**

  ```
  You are a wise and warm-hearted private tutor, dedicated to solving problems for your student.
  Providing the most appropriate answer to the problem is your important mission.

  Based on the given paragraph, please select the choice that you think is the correct answer to the question.

  There are a total of 5 choices, and each choice has a unique number (1 to 5).
  Your answer format must strictly be a single choice number (1 to 5). (Example: 4)

  Your student is sick and exhausted. By choosing the correct answer, you can encourage your student.
  Your answer must be correct to heal your student's emotional wounds and help them find happiness again.

  So, you have to think step by step and choose the answer.
  ```

### 실험 결과

| 모델                                | 조건                        | Accuracy   |
| ----------------------------------- | --------------------------- | ---------- |
| finetuned Qwen-2.5-32b-Instruct     | 기본 프롬프트               | 0.7540     |
| finetuned Qwen-2.5-32b-Instruct     | 명확한 규칙 제공            | 0.7724     |
| **finetuned Qwen-2.5-32b-Instruct** | **감정적 호소(In English)** | **0.7747** |

- **명확한 규칙 제공** 프롬프트는 기본 프롬프트에 비해 성능을 약 **1.8%** 향상시켰습니다.
- **감정적 호소 프롬프트**는 영어로 작성된 프롬프트로, 규칙 제공보다도 약간 더 높은 성능을 기록하며 **2.0%** 향상되었습니다.
- 이 결과는 모델이 **프롬프트의 맥락과 표현 방식**에 민감하게 반응하며, 인간적인 맥락이 성능 개선에 긍정적인 영향을 미칠 수 있음을 보여줍니다.

> 본 프롬프트 실험은 [AI 수능 국어 만점 프로젝트](https://github.com/NomaDamas/KICE_slayer_AI_Korean)의 프롬프트 선정을 기반으로 설계되었습니다.

## ⛓️ CoT 적용에 따른 추론 능력 평가

`google/gemma-2-2b-it`와 `Qwen/Qwen-2.5-7B-Instruct`에 대해 **Chain of Thought (CoT)** 방식의 성능을 비교 평가했습니다. 실험은 두 가지 방식으로 진행되었습니다:

1. **정답 번호**(숫자)만 생성
2. **JSON 형식으로 reasoning과 정답**을 포함하여 생성(CoT 방식)

| **모델**             | **실험 설정**            | **사전 학습 모델** | **파인튜닝 모델** |
| -------------------- | ------------------------ | ------------------ | ----------------- |
| gemma-2b-it          | 정답 번호만 생성         | 0.5034             | **0.5264**        |
| gemma-2b-it          | CoT 응답 생성(JSON 형식) | 0.4885             | 0.4437            |
| Qwen-2.5-7B-Instruct | 정답 번호만 생성         | 0.6092             | **0.6207**        |
| Qwen-2.5-7B-Instruct | CoT 응답 생성(JSON 형식) | 0.5609             | 0.6092            |

- **정답 번호만 생성**하는 방식이 대부분의 경우 더 높은 성능을 기록했습니다.
- **CoT 방식**은 `finetuning` 시 `Qwen 7B` 모델에서 일부 성능 향상을 보였지만, `gemma 2b` 모델에서는 성능이 저하되었습니다.

### 분석

- **모델 크기 한계**: CoT 방식은 더 많은 모델 파라미터를 요구하며, 작은 모델에서는 성능 저하가 발생할 수 있습니다.
- **데이터 특성**: 프로젝트의 주요 Task가 국어 및 사회 과목 중심으로 구성되어 있어, CoT 방식이 요구하는 단계적 추론이 충분히 활용되지 못했을 가능성이 있습니다.

> `config.yaml`에서 `common.cot_on=True` 설정 후 `langchain_inference.py` 실행함으로써 CoT를 테스트할 수 있습니다.

## 🔍 RAG를 활용한 검색 기반 답변 성능 실험

### RAG 시스템 도입

기존 지문 데이터를 증강하여 성능을 개선하려는 시도는 외부 지식 의존도가 높은 문제에서는 한계가 있음을 확인했습니다. 이를 해결하기 위해 RAG(정보 검색 기반 생성) 접근법을 도입하게 되었으며, `LangChain`을 활용하여 RAG 시스템을 구축하였습니다.

### RAG 시스템 설계

1. **문서 임베딩 및 저장**  
   `LangChain` 라이브러리를 사용하여 다양한 형식의 문서를 불러와 청킹한 뒤, `bge-m3` 모델을 기반으로 임베딩을 생성했습니다. 생성된 임베딩은 `Pinecone`이라는 Vector DB에 저장하여 검색 가능하도록 구성했습니다.

2. **Retriever 생성**  
   `Pinecone`에서 제공하는 간단한 명령어를 활용해 코사인 유사도 기반의 Retriever를 생성했습니다.

3. **문서 품질 최적화**  
   전체 위키피디아 문서를 사용하면 검색된 문서의 품질이 저하될 우려가 있어, 앞선 실험에서 도출한 키워드에 기반하여 관련 위키피디아 문서만을 크롤링했습니다.

### RAG 성능 평가 및 개선 노력

| 실험 구성             | Public acc | Final acc |
| --------------------- | ---------- | --------- |
| RAG 미사용 (Baseline) | 0.5253     | 0.4943    |
| RAG (top k=3)         | 0.4931     | 0.5172    |
| RAG (top k=5)         | 0.5230     | 0.5172    |

RAG를 사용하지 않은 Baseline에 비해 RAG 시스템이 더 높은 최종 정확도를 보여주었습니다. 이는 Retrieve된 문서를 활용한 접근법이 효과적으로 작용했음을 시사합니다.

하지만 프로젝트 진행 중, Public acc에서는 성능 하락이 관찰되었기에 원인 분석을 위해 LLM as Judge 방식을 활용하였습니다. 이를 통해, Retrieve된 문서의 품질이 좋지 않다는 것을 확인할 수 있었습니다.

이러한 문제점을 보완하며 모델의 견고성을 높이기 위해 RAFT를 시도했습니다. 논문을 (Zhang et al., 2024) 기반으로 최적의 P와 D 값을 설정하고, RAFT용 학습 데이터셋을 구성하였습니다.

앞서 CoT 적용을 시도한 경험이 있었는데, RAFT는 CoT 스타일의 답변을 가정한 학습법이기에, 해당 시도와 연계하여 앞으로 더 유의미한 성능 향상을 기대할 수 있을 것으로 보입니다.

## 🪟 Streamlit

![streamlit demo](assets/streamlit-demo.png)

`Streamlit`을 도입하여 **CSV 데이터를 보다 직관적으로 탐색**할 수 있는 대시보드를 구축했습니다. 수능 문제와 유사한 **지문, 문제, 선택지**를 화면에 시각적으로 표시하며, 데이터를 손쉽게 분석할 수 있습니다.

또한, 아래와 같은 **데이터 분포 시각화 기능**을 제공합니다:

- 컬럼별 값 길이 분포
- 전체 컬럼 값 길이 분포
- 선다형 문제의 선택지 개수 분포
- 정답 데이터의 정답 분포

데이터 구조가 변경되어도 유연하게 대응할 수 있도록 일반화된 시스템을 구축했습니다. 이를 통해 코드 중복을 줄이고 워크플로우를 단순화하여 **팀원 간 협업 효율성**을 향상시켰습니다.

> `streamlit run analysis_dashboard.py` 명령어를 실행해 대시보드를 확인할 수 있습니다.

## ⚙️ Project Quick Setup

### Requirements

- Python: 3.10
- CUDA: >= 12.1
- PyTorch: 2.5.1+cu121

### Git Clone

```shell
$ git clone git@github.com:boostcampaitech7/level2-nlp-generationfornlp-nlp-06-lv3.git
$ cd level2-nlp-generationfornlp-nlp-06-lv3
```

### Import Data

`data/` 디렉토리안에 `train/valid/test` 데이터를 위치시킵니다.

### Create Virtual Environment

```shell
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $
```

### Install Packages

```shell
(.venv) $ pip install -r requirements.txt
(.venv) $ sudo apt-get install build-essential
```

### Setup Environment Variables

`.env`를 생성 후 환경 변수를 수정합니다.

```shell
(.venv) $ cp .env.example .env
```

- `HF_TOKEN`: `HuggingFace`로 부터 모델을 내려받기 위해 필요한 토큰
- `STREAMLIT_DATA_PATH`: streamlit 구동 시 필요한 기본 설정 데이터 경로
- `STREAMLIT_EXPERIMENT_DATA_PATH`: streamlit 구동 시 필요한 실험 데이터 경로
- `PINECONE_API_KEY`: RAG vector db pinecone api key
- `PINECONE_INDEX`: RAG vector db pinecone index
- `PINECONE_ENVIRONMENT`: RAG vector db pinecone Environment
- `OPENAI_API_KEY`: 데이터 증강용 api key

```shell
HF_TOKEN={your_hf_token}
STREAMLIT_DATA_PATH={streamlit_data_path}
STREAMLIT_EXPERIMENT_DATA_PATH={streamlit_experiment_data_path}

# API Key for Pinecone service
PINECONE_API_KEY={your_api_key}
# Index name used in Pinecone
PINECONE_INDEX={your_index_name}
# Environment for Pinecone (e.g., 'us-west1-gcp')
PINECONE_ENVIRONMENT={your_environment}

OPENAI_API_KEY={your_openai_api_key}
```

## 📜 Config.yaml

`config.yaml` 파일을 사용하여 원하는 환경에서 실행을 설정할 수 있습니다. 아래는 기본 설정 예시입니다:

```yaml
model:
  name_or_path: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
  response_template: "<|im_start|>assistant\n"
  without_system_role: false # Deprecated (항상 system 없이 동작)
  torch_dtype: "float16" # float32, float16, bfloat16 / 모델의 기본 데이터 타입

common:
  seed: 42
  device: "cuda"
  cot_on: false # cot 사용 시 prompt_template을 cot_json으로 변경 필요
  prompt_template: "base" # base, cot_json, raft

bnb:
  load_in_8bit: false
  load_in_4bit: false
  bnb_4bit_compute_dtype: "float16" # float16, float32, bfloat16 / 4 bit 양자화 데이터의 계산 데이터 타입
  bnb_4bit_use_double_quant: false # true 시 더블 양자화(메모리 사용량 감소, 시간 더 걸림)
  bnb_4bit_quant_type: "nf4" # nf4, fp4

earlystop:
  metric_for_best_model: "eval_loss" # 모니터링할 지표 이름
  early_stopping_patience: 1 # 개선되지 않는 에폭 수
  early_stopping_threshold: 0.0 # 개선으로 간주할 최소 변화량
  greater_is_better: false # 지표가 높을수록 좋은 경우 True

peft:
  r: 6
  lora_alpha: 8
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

sft:
  do_train: true
  do_eval: true
  lr_scheduler_type: "cosine"
  max_seq_length: 1024
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  num_train_epochs: 3
  learning_rate: 2.0e-5 # 지수 자릿수 앞부분을 실수 형태로 작성 ('2'-> x, '2.0'-> o)
  weight_decay: 0.01
  logging_strategy: "steps" # epoch or steps, epoch의 경우 logging_steps 무시
  logging_steps: 100
  save_strategy: "epoch"
  eval_strategy: "epoch"
  load_best_model_at_end: true
  save_total_limit: 1
  save_only_model: true
  report_to: "wandb" # none or wandb, wandb로 변경하여 로그를 기록합니다.
  gradient_checkpointing: false
  gradient_accumulation_steps: 4

wandb:
  project: "MMLU"

train:
  data_path: "data/train_v2.0.1.csv" # wandb 로깅 사용시 파일명 변겅 금지(데이터 버전 정보 사용)
  valid_data_path: "data/valid_v2.0.1.csv"
  valid_output_path: "data/valid_output.csv"

inference:
  model_path: "outputs/Qwen2.5-3B-Instruct-bnb-4bit" # 학습된 모델로 변경 필요
  data_path: "data/test_v1.0.2.csv"
  output_path: "data/output.csv"
  raw_output_path: "data/raw_output.csv"
  default_answer: 1

rag:
  query_builder_type: CombinedKeyQueryBuilder_pqc
  raft_on: false # RAFT 시 prompt_template을 raft로 변경 필요
```

### Custom Config

기본 설정 외에 사용자 정의 설정을 사용하려면 `configs/config.yaml` 파일을 복사한 뒤 수정하세요:

```shell
(.venv) $ cp configs/config.yaml configs/config_custom.yaml
(.venv) $ vi configs/config_custom.yaml
```

## 🏃‍♂️‍➡️ Train 및 Inference 실행

### Train

학습을 실행하려면 기본 `train.py` 파일을 실행합니다:

```shell
(.venv) $ python train.py
```

커스텀 설정을 적용하려면 -c 또는 --config 옵션을 사용하세요:

```shell
(.venv) $ python train.py -c config_custom.yaml
```

- -c 옵션에는 configs 디렉토리 내부의 YAML 파일 이름만 입력합니다.

### Inference

학습된 모델을 사용하여 추론을 진행합니다:

```shell
(.venv) $ python inference.py
```

커스텀 설정을 사용하려면 다음 명령어를 실행하세요:

```
(.venv) $ python inference.py -c config_custom.yaml
```

`valid` 데이터에 대한 추론을 진행하려면 `-v` 옵션을 추가합니다:

> `valid`에 대한 모델의 성능을 확인할 수 있습니다

```shell
(.venv) $ python inference.py -v

# or

(.venv) $ python inference.py -c config_custom.yaml -v
```

## 📖 Contribution Guide

프로젝트에 기여하는 방법에 대한 [가이드](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-06-lv3/blob/main/CONTRIBUTING.md)입니다.

### 커밋 템플릿 사용법

프로젝트에서 커밋 메시지 형식을 통일하기 위해 커밋 템플릿을 설정할 수 있습니다. 아래 명령어를 실행하여 템플릿을 적용하세요:

```
$ git config commit.template .gitcommit_template
```

- `.gitcommit_template` 파일은 프로젝트 루트에 있는 커밋 템플릿 파일입니다.
- 위 명령어를 실행하면 커밋 시 템플릿이 자동으로 불러와집니다.

## 🔬 References

- Son Guijin, Hanwool Lee, Sungdong Kim, Seungone Kim, Niklas Muennighoff, Taekyoon Choi, Cheonbok Park, Kang Min Yoo & Stella Biderman. "KMMLU: Measuring Massive Multitask Language Understanding in Korean." _arXiv_, June 6, 2024. https://doi.org/10.48550/arXiv.2402.11548.
- Zhang Tianjun, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica & Joseph E. Gonzalez. “RAFT: Adapting Language Model to Domain Specific RAG”. arXiv, June 6, 2024. https://doi.org/10.48550/arXiv.2403.10131.
- 허동석, 김기태, 송형우, 서봉원. (2024-01-24). 프롬프트 개발을 통한 수능 국어 맞춤형 문제 생성 시스템 제안. 한국HCI학회 학술대회, 강원.
  https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11714641.
- skt/kobest 데이터셋 https://huggingface.co/datasets/skt/kobest_v1
