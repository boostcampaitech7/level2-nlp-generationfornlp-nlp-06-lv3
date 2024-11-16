- [Contribution](#contributing)

  - [Before development](#before-development)
  - [How to make an Issue](#how-to-make-an-issue)
  - [Development guildlines](#development-guidelines)

    - [기본 철학](#기본-철학)
    - [커밋 메시지 작성 가이드](#커밋-메시지-작성-가이드)
    - [브랜치 네이밍 규칙](#브랜치-이름-규칙)
    - [코드 스타일 및 코드 리뷰 가이드](#코드-스타일-및-코드-리뷰-가이드)

# Contributing

이 문서는 프로젝트에 기여하는 방법과 지켜야 할 규칙들을 안내합니다.

## Before development

1. [issues](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-06-lv3/issues)를 통해 변경점(또는 개선사항)에 대해 논의해주세요.

2. [개발 지침](#development-guidelines)을 읽고 프로젝트의 진행 방식을 숙지해주세요.

## How to make an Issue

### New Feature Request

### Bug Report

### Discussion

### Experiment Results

## Development guidelines

### 기본 철학

- 🛠️ **팀을 위한 코드 작성**: 다른 팀원이 쉽게 이해하고 유지보수할 수 있는 코드를 작성하세요.

- 📐 **일관성 유지**: 아래 가이드를 따라 코드베이스를 깨끗하고 체계적으로 유지합시다.

- 🌟 **코드 개선은 언제나 환영**: 작은 개선이라도 프로젝트의 품질을 높이는 데 기여합니다.

---

### 커밋 메시지 작성 가이드

프로젝트의 기록을 일관되고 명확하게 유지하기 위해 아래 규칙에 따라 커밋 메시지를 작성해주세요.

#### 커밋 작성 기본 원칙

1. **하나의 커밋에는 하나의 작업만 포함**하세요.

   - 여러 작업을 하나의 커밋에 묶지 말고, 작업별로 커밋을 나누어 작성합니다.

2. **메시지는 간결하고 명확하게** 작성하세요.

   - 커밋의 목적과 내용을 명확히 알 수 있도록 작성합니다.

3. **의미 있는 커밋을 남기세요.**

   - 불필요한 커밋이나 임시 저장용 커밋(예: `save` 등)은 피하세요.

#### 커밋 태그 (Prefix)

커밋 메시지 앞에 태그를 추가하여 작업의 목적을 명확히 나타냅니다.

1. **기능과 버그 관련 태그**

   - `feat`: 새로운 기능 추가
   - `fix`: 버그 수정
   - `hotfix`: 긴급한 수정

2. **코드 품질과 구조 개선 태그**

   - `refactor`: 코드 리팩토링 (기능 변경 없음)
   - `style`: 코드 스타일 수정 (예: 포맷팅, 세미콜론 누락)

3. **문서 및 설정 관련 태그**

   - `docs`: 문서 추가 및 수정
   - `chore`: 설정, 빌드, 패키지 작업 (코드 변경 없음)

4. **테스트 관련 태그**
   - `test`: 테스트 코드 추가 또는 수정

더 자세한 커밋 작성 방법은 [커밋 템플릿](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-06-lv3/blob/main/.gitcommit_template)을 참고해주세요.

---

### 브랜치 네이밍 규칙

브랜치 이름은 다음 두 가지 유형 중 하나를 따릅니다.

#### 1. `feature` 브랜치

- 새로운 기능 개발에 사용합니다.
- **형식**: `feature/{issue-number}-short-description`
  - `{issue-number}`: 해당 작업의 이슈 번호
  - `short-description`: 기능을 간단히 설명하는 이름
- **예시**:
  - `feature/123-user-login`
  - `feature/45-add-item-to-cart`

#### 2. `hotfix` 브랜치

- 긴급한 수정 작업에 사용합니다.
- **형식**: `hotfix-short-description`
  - `short-description`: 수정할 문제를 간단히 설명
- **예시**:
  - `hotfix-fix-login-error`
  - `hotfix-correct-typo`

#### 📝 네이밍 관련 설명

- 구분자 사용
  - `feature` 뒤에는 `/`를 사용해 폴더 구조처럼 구분합니다.
  - 기능 이름은 `-`로 단어를 연결합니다.
- 브랜치 이름은 **작업 내용을 명확히 표현**하도록 간결하게 작성합니다.
- 이슈 번호와 연동하여 작업 내역을 추적합니다.

대부분의 경우에는 feature를 사용하시면 됩니다.

---

### 코드 스타일 및 코드 리뷰 가이드

코드의 품질과 일관성을 유지하기 위해 아래 규칙을 따라주세요.

#### 코드 스타일

1. **불변성 선호 (Immutability)**

   - 가능하면 변수와 객체를 변경하지 않고 새로 생성하여 사용하세요.
   - 불변성을 유지하면 디버깅이 더 쉬워지고, 예기치 않은 부작용(Side Effects)을 줄일 수 있습니다.

2. **단일 책임 원칙 (Single Responsibility Principle)**

   - 함수와 클래스는 하나의 명확한 목적만 가지도록 작성합니다.
   - 하나의 함수가 여러 가지 일을 하고 있다면 분리하세요.

3. **명확한 네이밍**

   - 변수명과 함수명은 해당 역할과 목적을 명확히 나타내야 합니다.
   - **좋은 예**: `calculateTotalPrice`, `isUserLoggedIn`
   - **나쁜 예**: `doSomething`, `data`

4. **코드 자체로 의도를 설명**

   - 주석 대신 코드를 명확하게 작성하세요. 복잡한 로직은 함수로 분리하여 의도를 드러냅니다.
   - **예시**:

     ```python
     # 나쁜 코드: 주석으로 설명 필요
     total_price = sum(item.price for item in cart if item.on_sale)

     # 좋은 코드: 함수로 의도를 명확히
     def calculate_total_price(cart):
         return sum(item.price for item in cart if item.on_sale)
     ```

5. **공식 문서 및 레퍼런스 사용**
   - GPT나 비공식적인 자료보다는 **공식 문서** 또는 **신뢰할 수 있는 자료**를 기반으로 코드 작성 및 리뷰를 진행하세요.

### 코드 리뷰 가이드

1. **핵심 로직 검토**

   - 구현된 로직이 명세(처음 기획한 의도)에 맞게 작동하는지 확인합니다.
   - 불필요하거나 과도한 변경이 없는지 검토합니다.

2. **코드 스타일 준수 확인**

   - 위의 코드 스타일 규칙을 잘 따랐는지 확인합니다.
   - 변수명, 함수명이 명확한지, 불변성을 유지했는지 체크하세요.

3. **효율성과 확장성 고려**

   - 코드가 효율적으로 작성되었는지, 향후 확장 가능성이 있는지 검토합니다.
   - 불필요한 복잡성은 없는지 확인합니다.

4. **명확한 피드백 제공**

   - 코드 개선 제안 시 **명확한 이유와 대안**을 제시하세요.
   - 피드백은 건설적으로 제공하며, 개인이 아닌 코드에 집중합니다.

<!-- 테스트 관련 항목은 프로젝트 말미에 추가할 예정입니다. -->
<!--
5. **테스트 케이스 검토 (선택 사항)**
   - 작성된 코드가 테스트되었는지 확인합니다.
   - 테스트가 없는 경우, 기능 동작을 확인할 수 있는 최소한의 테스트를 추가하도록 권장합니다.
 -->