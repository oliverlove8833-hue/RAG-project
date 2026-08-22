from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    RemoveMessage
)
from ai.state import AgentState
from ai.retriever import get_retriever
from ai.text2sql import get_text2sql_engine
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List


load_dotenv()

llm = init_chat_model("gpt-5.4-mini")

_retriever = None
_text2sql_engine = None


# =========================================================
# Structured Output
# =========================================================

class VectorSearchQuery(BaseModel):
    """IoT 문서 벡터 검색을 위한 쿼리 분석 결과"""

    optimized_query: str = Field(
        description=(
            "IoT 하드웨어 설계 및 보안 문서 검색에 최적화된 쿼리. "
            "MCU, GPIO, I2C, SPI, UART, JTAG, PCB, 메모리, 전원, "
            "인증, 암호화 등 필요한 핵심 용어를 포함."
        )
    )

    categories: Optional[List[str]] = Field(
        default=None,
        description=(
            "선택된 카테고리 리스트 (1~2개). "
            "명확하게 관련 있는 카테고리만 선택하고 "
            "애매하면 null 반환. 가능한 값: "
            "회로_전원_신호설계, "
            "PCB_배선_기판설계, "
            "MCU_메모리_부품설계, "
            "인터페이스_통신설계, "
            "하드웨어_물리보안, "
            "인증_접근통제, "
            "암호화_데이터보호, "
            "펌웨어_플랫폼보안"
        )
    )


# =========================================================
# Cached Resources
# =========================================================

def get_cached_retriever():
    """캐시된 retriever 인스턴스 반환"""

    global _retriever

    if _retriever is None:
        _retriever = get_retriever()

    return _retriever


def get_cached_text2sql_engine():
    """캐시된 Text2SQL 엔진 반환"""

    global _text2sql_engine

    if _text2sql_engine is None:
        _text2sql_engine = get_text2sql_engine()

    return _text2sql_engine


# =========================================================
# 1. 질문 분석
# =========================================================

def analyze_question(state: AgentState) -> AgentState:
    """
    이전 대화 맥락을 반영하여
    현재 질문을 완전한 IoT 하드웨어·보안 질문으로 재구성
    """

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No messages provided")

    question = (
        messages[-1].content
        if hasattr(messages[-1], "content")
        else str(messages[-1])
    )

    # 첫 질문이면 그대로 사용
    if len(messages) <= 1:
        return {
            "question": question,
            "analyzed_question": question
        }

    system_prompt = """
당신은 IoT 디바이스 하드웨어 설계와 보안 질문을 분석하는 전문가입니다.

이전 대화 내용을 고려하여 현재 질문을
그 자체로 의미가 통하는 완전한 질문으로 재구성하세요.

예시:

- 이전:
  "ESP32-H2의 GPIO 기능 알려줘"
  현재:
  "보안은?"
  재구성:
  "ESP32-H2의 GPIO 설계와 함께 적용할 수 있는 보안 기능은 무엇인가요?"

- 이전:
  "스마트 도어락을 만들려고 해"
  현재:
  "연결은 어떻게 해?"
  재구성:
  "스마트 도어락을 만들 때 MCU와 센서 등 하드웨어를 어떻게 연결해야 하나요?"

- 이전:
  "JTAG가 뭐야?"
  현재:
  "제품 만들고 나서도 열어놔?"
  재구성:
  "IoT 제품을 완성한 뒤에도 JTAG 디버그 인터페이스를 활성화해 두어야 하나요?"

- 이전:
  "I2C 알려줘"
  현재:
  "보안칩 연결 가능?"
  재구성:
  "I2C를 사용하여 MCU와 하드웨어 보안 모듈을 연결할 수 있나요?"

규칙:
- 질문의 원래 의미를 바꾸지 마세요.
- 없는 조건이나 부품을 임의로 추가하지 마세요.
- 완전한 질문만 반환하세요.
- 설명은 포함하지 마세요.
- 이미 완전한 질문이면 그대로 반환하세요.
"""

    conversation = [
        SystemMessage(content=system_prompt)
    ] + messages

    response = llm.invoke(conversation)

    analyzed = response.content.strip()

    print(f"[질문 분석] {question} → {analyzed}")

    return {
        "question": question,
        "analyzed_question": analyzed
    }


# =========================================================
# 2. 의도 분류
# =========================================================

def classify_intent(state: AgentState) -> AgentState:
    """
    질문을 4가지 의도로 분류

    general
    iot_consult
    database
    hybrid
    """

    question = state.get("analyzed_question", "")

    system_prompt = """
당신은 IoT 디바이스 하드웨어 설계와 보안 질문을 분류하는 전문가입니다.

질문을 다음 4가지 중 하나로 분류하세요.


1. general

일반적인 대화, 인사, 서비스 설명처럼
IoT 전문 자료 검색이 필요하지 않은 질문입니다.

예:
- "안녕하세요"
- "고마워"
- "너는 뭘 할 수 있어?"
- "어떤 도움을 줄 수 있어?"


2. iot_consult

PDF 기술문서나 보안 가이드를 바탕으로
IoT 하드웨어 설계 원리, 보안 원리, 점검 방법,
적용 방법 등을 설명해야 하는 질문입니다.

예:
- "JTAG를 왜 막아야 하나요?"
- "IoT 제품에서 내부 포트는 어떻게 보호해야 하나요?"
- "PCB를 설계할 때 보안상 무엇을 고려해야 하나요?"
- "MCU 메모리는 어떻게 보호하나요?"
- "I2C는 어떤 방식으로 장치를 연결하나요?"


3. database

구조화된 데이터베이스에서
기능 목록, 항목, 분류, 페이지, 개수, 비교 등
정확한 데이터를 조회하는 것이 중심인 질문입니다.

예:
- "ESP32-H2의 하드웨어 기능을 모두 보여줘"
- "STM32에서 JTAG와 관련된 기능은 뭐가 있어?"
- "ESP32-H2의 보안 기능은 몇 개야?"
- "하드웨어 관련 기능을 페이지 순서대로 알려줘"
- "공통 가이드별 연결 항목 수를 알려줘"


4. hybrid

PDF의 구체적인 설명과
DB의 구조화된 기능 정보를 모두 사용하는 것이 좋은 질문입니다.

특히 실제 IoT 제품을
'만드는 방법', '설계 방법', '점검 방법'처럼
하드웨어 설계와 보안을 함께 요구하면 hybrid를 우선 선택하세요.

예:
- "스마트 도어락 제작 방법 알려줘"
- "IoT 기기를 만들 때 부품은 어떻게 연결하고 보안은 어떻게 적용해?"
- "ESP32-H2로 제품을 설계할 때 하드웨어와 보안을 같이 알려줘"
- "센서를 MCU에 어떻게 연결하고 해킹도 막으려면 어떻게 해야 해?"
- "STM32로 IoT 장치를 만들 때 설계와 보안을 같이 점검해줘"


판단 기준:

- 단순 인사/대화 → general
- 기술 원리 또는 보안/하드웨어 설명 → iot_consult
- 목록·개수·페이지·정렬·비교 등 정확한 데이터 조회 → database
- 제품 제작·설계·점검처럼 하드웨어 + 보안을 함께 요구 → hybrid


중요:
"스마트 도어락 제작 방법 알려줘"
같은 질문은 반드시 hybrid로 분류하세요.

반드시 다음 네 값 중 하나만 반환하세요:

general
iot_consult
database
hybrid

다른 설명은 하지 마세요.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)

    intent = response.content.strip().lower()

    valid_intents = [
        "general",
        "iot_consult",
        "database",
        "hybrid"
    ]

    if intent not in valid_intents:
        intent = "general"

    print(f"[의도 분류] {intent}")

    return {
        "intent": intent
    }


# =========================================================
# 3. 일반 답변
# =========================================================

def general_answer(state: AgentState) -> AgentState:
    """일반 대화 및 에이전트 소개"""

    messages = state.get("messages", [])

    system_prompt = """
당신은 IoT 디바이스의 하드웨어 설계와 보안을 함께 도와주는 AI 에이전트입니다.

사용자가 일반적인 질문이나 서비스 소개를 요청하면
쉽고 자연스럽게 답변하세요.

이 시스템은 다음과 같은 도움을 줄 수 있습니다.

- IoT 기기의 MCU, GPIO, 메모리, 전원 등 하드웨어 설계 정보
- UART, SPI, I2C 등 부품 연결과 통신 방식
- PCB 및 디버그 포트 설계 시 고려사항
- JTAG, UART 등 물리적 인터페이스 보안
- 인증 및 접근 통제
- 데이터 및 메모리 암호화
- 펌웨어 및 플랫폼 보호
- STM32 및 ESP32-H2 관련 기술 정보
- 홈·가전 IoT 제품의 하드웨어·보안 점검

전문용어는 최대한 쉽게 설명하세요.

사용자가 처음 IoT를 배우는 사람이라고 생각하고 답변하세요.
"""

    conversation = [
        SystemMessage(content=system_prompt)
    ] + messages

    response = llm.invoke(conversation)

    answer = response.content

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


# =========================================================
# 4. PDF Vector Search
# =========================================================

def vector_search(state: AgentState) -> AgentState:
    """
    Qdrant PDF 검색

    1. 질문 검색어 최적화
    2. IoT 카테고리 선택
    3. 벡터 검색
    """

    original_query = (
        state.get("rewritten_query")
        or state.get("analyzed_question", "")
    )

    system_prompt = """
당신은 IoT 디바이스 하드웨어 설계 및 보안 문서의
검색 쿼리를 최적화하는 전문가입니다.

사용자의 일상적인 질문을
기술 문서에서 검색하기 좋은 표현으로 바꾸세요.


사용 가능한 카테고리:


1. 회로_전원_신호설계

전원, 리셋, 클록, 신호 안정성, 저전력,
전압 감지, GPIO 신호 등과 관련된 하드웨어 설계


2. PCB_배선_기판설계

PCB 배선, 통신선 내층 배치,
테스트 포인트, 실크인쇄,
개발용 PCB와 양산용 PCB 구분 등


3. MCU_메모리_부품설계

MCU, 프로세서, Flash, RAM,
외부 메모리, eFuse,
하드웨어 보안 모듈 등


4. 인터페이스_통신설계

GPIO, UART, SPI, I2C,
USB, JTAG, SWD 등
부품 연결 및 통신 인터페이스


5. 하드웨어_물리보안

외부·내부 포트 보호,
JTAG/UART 차단,
분해 방지,
디버그 인터페이스 보호,
역공학 방지 등


6. 인증_접근통제

사용자 인증,
장치 간 인증,
접근권한 관리,
Permission Control 등


7. 암호화_데이터보호

AES, RSA, HMAC, SHA,
XTS-AES, 암호키,
중요 데이터 보호,
메모리 암호화 등


8. 펌웨어_플랫폼보안

Secure Boot,
펌웨어 보호,
안전한 업데이트,
코드 보호,
RDP, PCROP 등


카테고리 선택 규칙:

1. 질문과 직접 관련 있는 카테고리를 1~2개 선택하세요.
2. 하드웨어와 보안이 함께 필요한 질문이면 각각 하나씩 선택할 수 있습니다.
3. 애매한 경우 억지로 선택하지 말고 null을 반환하세요.
4. 반드시 위 카테고리 이름을 정확히 사용하세요.


검색어 최적화 예:

"기기 부품은 어떻게 연결해?"
→
"MCU GPIO UART SPI I2C 주변장치 연결 인터페이스 설계"

"도어락을 뜯어서 해킹 못하게 하려면?"
→
"디지털 도어락 물리적 보안 분해 방지 내부 포트 JTAG UART 비활성화"

"프로그램 못 빼가게 하려면?"
→
"MCU Flash 코드 보호 RDP PCROP 메모리 보호 디버그 접근 차단"

"스마트 도어락 만드는 방법"
→
"디지털 도어락 MCU GPIO 센서 통신 인터페이스 PCB 하드웨어 설계 물리적 보안 인증 암호화"


출력:
- optimized_query
- categories

Structured Output 형식에 맞춰 반환하세요.
"""

    user_prompt = f"""
다음 질문을 분석해주세요:

{original_query}
"""

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    structured_llm = llm.with_structured_output(
        VectorSearchQuery
    )

    query_analysis = structured_llm.invoke(
        llm_messages
    )

    optimized_query = query_analysis.optimized_query
    categories = query_analysis.categories

    print("[벡터 검색 쿼리 분석]")
    print(f"  원본 쿼리: {original_query}")
    print(f"  최적화된 쿼리: {optimized_query}")
    print(f"  선택된 카테고리: {categories}")

    retriever = get_cached_retriever()

    results = retriever.search(
        optimized_query,
        k=4,
        score_threshold=0.5,
        categories=categories
    )

    return {
        "vector_results": results
    }


# =========================================================
# 5. 검색 문서 평가
# =========================================================

def grade_documents(state: AgentState) -> AgentState:
    """검색 문서가 질문에 답하기 충분한지 평가"""

    question = state.get(
        "analyzed_question",
        ""
    )

    results = state.get(
        "vector_results",
        []
    )

    vector_retry = state.get(
        "vector_retry",
        0
    )

    if not results:
        print(
            "[문서 평가] not enough "
            "(검색 결과 없음)"
        )

        return {
            "document_grade": "not enough"
        }

    # 재시도 제한
    if vector_retry >= 2:
        print(
            "[문서 평가] relevant "
            "(재시도 한도 도달)"
        )

        return {
            "document_grade": "relevant"
        }

    doc_summary = "\n\n".join(
        [
            f"[문서 {i}]\n"
            f"{doc.page_content[:600]}"
            for i, doc in enumerate(results, 1)
        ]
    )

    system_prompt = """
당신은 IoT 하드웨어 설계 및 보안 문서의
검색 결과를 평가하는 전문가입니다.

검색된 자료만으로 사용자의 질문에
유용한 답변을 만들 수 있는지 판단하세요.


평가 기준:

1. 질문과 직접 관련된 내용이 있는가?

2. 사용자가 하드웨어 설계를 물었다면
   MCU, 메모리, GPIO, UART, SPI, I2C,
   PCB, 전원, 클록, 리셋, 인터페이스 등
   설계에 사용할 정보가 있는가?

3. 사용자가 보안을 물었다면
   포트 보호, 인증, 접근 통제,
   암호화, 메모리 보호,
   펌웨어 보호 등의 정보가 있는가?

4. 사용자가 하드웨어 + 보안을 함께 물었다면
   가능한 한 두 관점의 내용이 모두 있는가?

5. 문서 내용이 질문에 대한 근거로 사용할 수 있는가?


판단:

relevant
- 질문에 답하기에 충분한 관련 정보가 있음

not enough
- 검색 결과가 질문과 거의 관련 없거나
  필요한 정보가 매우 부족함


반드시

relevant

또는

not enough

중 하나만 반환하세요.
"""

    user_prompt = f"""
질문:
{question}

검색된 문서:
{doc_summary}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    grade = response.content.strip().lower()

    if grade not in [
        "relevant",
        "not enough"
    ]:
        grade = "relevant"

    print(f"[문서 평가] {grade}")

    return {
        "document_grade": grade
    }


# =========================================================
# 6. 검색 쿼리 재작성
# =========================================================

def rewrite_query(state: AgentState) -> AgentState:
    """검색 결과가 부족하면 기술 문서 검색어로 재작성"""

    question = state.get(
        "analyzed_question",
        ""
    )

    previous = state.get(
        "rewritten_query"
    )

    system_prompt = """
당신은 IoT 하드웨어 및 보안 기술문서
검색 쿼리 최적화 전문가입니다.

이전 검색으로 충분한 자료를 찾지 못했습니다.

사용자의 쉬운 표현을
기술 문서에서 실제로 검색하기 좋은
하드웨어·보안 용어로 다시 작성하세요.


주요 하드웨어 검색어:

- MCU
- GPIO
- I2C
- SPI
- UART
- JTAG
- SWD
- PCB
- Flash
- RAM
- eFuse
- Clock
- Reset
- Low Power
- Power Supply


주요 보안 검색어:

- 물리적 인터페이스 차단
- 디버그 포트 보호
- 인증 및 접근통제
- 암호화
- 암호키 관리
- 메모리 보호
- RDP
- PCROP
- Secure Boot
- 펌웨어 보호
- Tamper Proofing


예:

"부품 연결 어떻게 해?"
→
"MCU GPIO I2C SPI UART 주변장치 인터페이스 연결 설계"

"도어락 해킹 막으려면?"
→
"디지털 도어락 내부 포트 JTAG UART 물리적 보안 인증 암호화 분해 방지"

"스마트 도어락 제작 방법"
→
"디지털 도어락 MCU GPIO 센서 통신 인터페이스 PCB 설계 JTAG 물리적 보안 인증 데이터 보호"


규칙:

- 질문의 의미를 바꾸지 마세요.
- 관련 기술용어와 동의어를 추가하세요.
- 하드웨어와 보안을 함께 묻는 질문이면
  두 분야의 검색어를 모두 포함하세요.
- 재작성된 검색어만 반환하세요.
- 설명은 하지 마세요.
"""

    user_prompt = f"""
원본 질문:
{question}
"""

    if previous:
        user_prompt += f"""
이미 사용한 검색어:
{previous}

위 검색어와 다른 방식으로 작성하세요.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    rewritten = response.content.strip()

    print(f"[쿼리 재작성] {rewritten}")

    return {
        "rewritten_query": rewritten,
        "vector_retry": (
            state.get("vector_retry", 0) + 1
        )
    }


# =========================================================
# 7. Text2SQL DB 조회
# =========================================================

def database_query(state: AgentState) -> AgentState:
    """IoT 구조화 데이터 Text2SQL 조회"""

    question = state.get(
        "analyzed_question",
        ""
    )

    previous_error = state.get(
        "error"
    )

    text2sql_engine = (
        get_cached_text2sql_engine()
    )

    result = text2sql_engine.query(
        question,
        previous_error=previous_error
    )

    print(
        f"[SQL 생성] "
        f"{result['sql_query']}"
    )

    return {
        "sql_query": result["sql_query"],
        "db_results": result["result"],
        "error": result["error"],
        "db_retry": (
            state.get("db_retry", 0) + 1
        )
    }


# =========================================================
# 8. DB 결과 검증
# =========================================================

def validate_db_result(
    state: AgentState
) -> AgentState:
    """Text2SQL 실행 결과 검증"""

    error = state.get("error")
    result = state.get("db_results")

    db_retry = state.get(
        "db_retry",
        0
    )

    text2sql_engine = (
        get_cached_text2sql_engine()
    )

    if (
        error
        or not result
        or text2sql_engine.is_empty_result(result)
    ):

        if db_retry >= 2:
            print(
                "[DB 검증] valid "
                "(재시도 한도 도달)"
            )

            return {
                "db_valid": "valid"
            }

        print(
            f"[DB 검증] not valid "
            f"({'오류' if error else '결과 없음'})"
        )

        return {
            "db_valid": "not valid"
        }

    print("[DB 검증] valid")

    return {
        "db_valid": "valid"
    }


# =========================================================
# 9. PDF + DB Context 통합
# =========================================================

def combine_context(
    state: AgentState
) -> AgentState:
    """
    PDF 검색 결과와
    IoT 구조화 DB 조회 결과 통합
    """

    context_parts = []

    # -----------------------------------------
    # PDF
    # -----------------------------------------

    if state.get("vector_results"):

        docs = state["vector_results"]

        context_parts.append(
            "=== IoT 기술 문서 검색 결과 ==="
        )

        for i, doc in enumerate(docs, 1):

            source = doc.metadata.get(
                "source",
                "알 수 없음"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            category = doc.metadata.get(
                "category",
                ""
            )

            source_info = (
                f"출처: {source}, "
                f"페이지: {page}"
            )

            if category:
                source_info += (
                    f", 카테고리: {category}"
                )

            context_parts.append(
                f"\n[문서 {i}]\n"
                f"{source_info}\n"
                f"{doc.page_content}"
            )

    # -----------------------------------------
    # DB
    # -----------------------------------------

    if state.get("db_results"):

        context_parts.append(
            "\n\n=== IoT 구조화 데이터 조회 결과 ==="
        )

        context_parts.append(
            str(state["db_results"])
        )

        if state.get("sql_query"):

            context_parts.append(
                "\n참고용 실행 SQL:\n"
                f"{state['sql_query']}"
            )

    combined = (
        "\n".join(context_parts)
        if context_parts
        else "(참고 정보 없음)"
    )

    print(
        f"[컨텍스트 통합] "
        f"문서 "
        f"{len(state.get('vector_results') or [])}건 / "
        f"DB "
        f"{'있음' if state.get('db_results') else '없음'}"
    )

    return {
        "combined_context": combined
    }


# =========================================================
# 10. 최종 답변 생성
# =========================================================

def generate_answer(
    state: AgentState
) -> AgentState:
    """
    IoT 하드웨어 설계 + 보안 통합 답변 생성
    """

    messages = state.get(
        "messages",
        []
    )

    context = state.get(
        "combined_context",
        "(참고 정보 없음)"
    )

    question = state.get(
        "analyzed_question",
        ""
    )

    system_prompt = f"""
당신은 IoT 디바이스의
하드웨어 설계와 보안을 함께 점검하는 전문가입니다.

하지만 사용자는 전문 엔지니어가 아닐 수도 있으므로
답변은 최대한 쉽고 자연스러운 한국어로 작성하세요.


사용자 질문:

{question}


아래 검색 결과만 근거로 답변하세요.

<context>

{context}

</context>


==================================================
가장 중요한 원칙
==================================================

이 시스템의 목적은

"하드웨어 설계"

와

"보안"

을 따로 설명하는 것이 아니라

하나의 IoT 제품 설계 과정에서
두 요소를 함께 설명하는 것입니다.


사용자가 제품 제작, 설계, 구성 방법을 질문했다면
보안 이야기만 하지 마세요.

가능한 경우 전체 답변의 약 절반은
하드웨어 구성과 연결·설계 설명에 사용하고,

나머지 절반은
그 하드웨어에 필요한 보안 설명에 사용하세요.


==================================================
하드웨어 설명 방법
==================================================

하드웨어 이름만 나열하지 마세요.

예:

나쁜 답변:

"GPIO, SPI, I2C를 사용합니다."


좋은 답변:

"버튼이나 센서의 신호는 GPIO를 통해 MCU에 연결할 수 있습니다.
여러 센서나 주변 부품은 I2C를 이용해 연결할 수 있고,
빠르게 데이터를 주고받아야 하는 장치는 SPI를 사용할 수 있습니다."


검색 결과에 있다면 다음 내용을 적극적으로 설명하세요:

- MCU가 어떤 역할을 하는지
- 센서와 외부 장치를 어떻게 연결하는지
- GPIO의 역할
- UART의 역할
- SPI의 역할
- I2C의 역할
- 메모리의 역할
- 전원 및 저전력 설계
- Reset
- Clock
- PCB
- 디버그 포트
- 외부 및 내부 인터페이스


==================================================
보안 설명 방법
==================================================

보안 기능도 이름만 나열하지 마세요.

예:

나쁜 답변:

"JTAG를 비활성화하고 RDP를 적용하세요."


좋은 답변:

"JTAG는 개발할 때 MCU 내부를 확인하는 점검용 통로입니다.
제품을 완성한 뒤에도 열려 있으면
외부에서 내부 프로그램을 확인하는 데 악용될 수 있습니다.
따라서 필요하지 않은 경우 접근을 막는 것이 좋습니다."


검색 결과에 있다면 다음 내용을 쉽게 설명하세요:

- JTAG / UART 등 디버그 포트 보호
- 외부 포트 보호
- 사용자 인증
- 장치 간 인증
- 데이터 암호화
- 암호키 보호
- 메모리 보호
- 펌웨어 보호
- 안전한 업데이트
- 분해 방지


==================================================
제품 제작 질문 답변 순서
==================================================

"스마트 도어락 제작 방법 알려줘"

같은 제작 질문에서는 가능하면 다음 순서로 답하세요.


1. 하드웨어 구성

어떤 기능과 부품이 필요하고
각각 어떤 역할을 하는지 쉽게 설명


2. 연결 및 설계

MCU와 센서 또는 주변장치를
어떤 인터페이스를 이용해 연결할 수 있는지 설명


3. 보안 적용

앞에서 설명한 하드웨어에
어떤 보안을 같이 적용해야 하는지 설명


4. 통합 점검

완제품을 만들 때
하드웨어와 보안을 같이 확인해야 할 사항을 간단히 정리


==================================================
쉬운 설명 규칙
==================================================

전문용어는 가능한 한 줄이세요.

꼭 필요한 경우:

"전문용어(쉬운 설명)"

형식으로 표현하세요.


예:

GPIO
→
"센서나 버튼의 신호를 받거나 보내는 핀(GPIO)"


I2C
→
"두 개의 신호선으로 여러 부품을 연결하는 방식(I2C)"


SPI
→
"센서나 메모리와 빠르게 데이터를 주고받는 연결 방식(SPI)"


JTAG
→
"MCU 내부를 점검하는 개발용 연결 통로(JTAG)"


eFuse
→
"중요한 설정을 칩 내부에 저장하는 영역(eFuse)"


==================================================
근거 규칙
==================================================

- 반드시 context에 있는 정보를 중심으로 답하세요.

- context에 없는 부품 모델명을 만들어내지 마세요.

- 정확한 핀 번호가 없다면 임의로 만들지 마세요.

- 저항값, 커패시터값, 전압값 등
  구체적인 회로 수치가 자료에 없다면 만들어내지 마세요.

- 자료에 없는 회로 수치를 질문받으면:

  "현재 참고 자료에서는 보안과 관련된 하드웨어 설계 방향은 확인할 수 있지만,
  구체적인 회로 수치까지는 제공하지 않습니다."

  라고 안내하세요.

- 정보 자체가 부족하면:

  "현재 참고 자료만으로는 이 부분을 정확하게 판단하기 어렵습니다."

  라고 안내하세요.

- SQL이나 데이터베이스라는 내부 구현 표현은
  최종 사용자에게 굳이 언급하지 마세요.

- 같은 내용을 반복하지 마세요.

- 답변을 지나치게 길게 만들지 마세요.

- 문서에서 가져온 내용이 있다면
  답변 마지막에 자료명과 페이지를 간단히 표시하세요.
"""

    conversation = [
        SystemMessage(content=system_prompt)
    ] + messages

    response = llm.invoke(conversation)

    answer = response.content

    updates = []

    # 답변 재생성 시 기존 AI 메시지 제거
    if (
        state.get("answer_retry", 0) > 0
        and messages
        and messages[-1].type == "ai"
    ):

        updates.append(
            RemoveMessage(
                id=messages[-1].id
            )
        )

    updates.append(
        AIMessage(
            content=answer
        )
    )

    return {
        "messages": updates
    }


# =========================================================
# 11. 답변 검증
# =========================================================

def validate_answer(
    state: AgentState
) -> AgentState:
    """
    생성된 답변 검증

    - 자료 근거 여부
    - 하드웨어 설명 여부
    - 보안 설명 여부
    - 쉬운 설명 여부
    """

    intent = state.get(
        "intent",
        "general"
    )

    answer_retry = state.get(
        "answer_retry",
        0
    )

    messages = state.get(
        "messages",
        []
    )

    last_ai = (
        [messages[-1]]
        if messages
        and messages[-1].type == "ai"
        else []
    )

    # 일반 대화는 검증 제외
    if intent == "general":

        return {
            "answer_valid": "ok",
            "messages": last_ai
        }

    # 1회 재생성 후에는 종료
    if answer_retry >= 1:

        print(
            "[답변 검증] ok "
            "(재생성 한도 도달)"
        )

        return {
            "answer_valid": "ok",
            "messages": last_ai
        }

    answer = (
        messages[-1].content
        if messages
        else ""
    )

    context = state.get(
        "combined_context",
        ""
    )

    question = state.get(
        "analyzed_question",
        ""
    )

    system_prompt = """
당신은 IoT 하드웨어 설계·보안 통합 답변을
검증하는 전문가입니다.

사용자 질문과 참고 정보,
생성된 답변을 비교하세요.


검증 기준:


1. 근거

답변 내용이 참고 정보에 실제로 존재하는가?


2. 환각

참고 정보에 없는

- 핀 번호
- 부품 모델
- 회로 값
- 전압
- 저항값
- 구체적인 구현값

등을 임의로 만들어내지 않았는가?


3. 하드웨어 설계

사용자가 하드웨어 또는 제품 제작 방법을 물었다면
MCU, 센서 연결, GPIO, I2C, SPI, UART,
메모리, 전원, PCB 등

검색 결과에서 확인되는 하드웨어 내용이
실제로 설명되었는가?


4. 보안

보안이 필요한 질문이면

포트 보호,
인증,
접근통제,
암호화,
메모리 보호,
펌웨어 보호

등 관련 내용이 설명되었는가?


5. 통합성

사용자가 하드웨어와 보안을 함께 물었다면
보안 내용만 지나치게 많지 않고

하드웨어 설계 + 보안

두 내용이 모두 포함되었는가?


6. 쉬운 설명

전문용어만 나열하지 않고
처음 배우는 사람도 이해할 수 있게 설명했는가?


7. 출처

PDF 자료를 사용했다면
가능한 범위에서 자료명과 페이지가 포함되어 있는가?


판단:

ok
- 질문에 맞고 근거가 있으며
  하드웨어·보안 설명이 적절함

not ok
- 근거 없는 내용이 있거나
- 질문에 대한 핵심 내용이 빠졌거나
- 하드웨어+보안을 물었는데 한쪽만 설명하거나
- 전문용어만 나열하여 이해하기 어려움


참고 정보에 원하는 내용 자체가 없어서
"현재 자료에서 확인하기 어렵다"고 답한 경우는 ok입니다.


반드시

ok

또는

not ok

중 하나만 반환하세요.
"""

    user_prompt = f"""
사용자 질문:

{question}


참고 정보:

{context[:4000]}


생성된 답변:

{answer[:2500]}
"""

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(
        llm_messages
    )

    valid = (
        response.content
        .strip()
        .lower()
    )

    if valid not in [
        "ok",
        "not ok"
    ]:
        valid = "ok"

    print(
        f"[답변 검증] {valid}"
    )

    return {
        "answer_valid": valid,
        "answer_retry": (
            answer_retry + 1
        ),
        "messages": last_ai
    }


# =========================================================
# Routing Functions
# =========================================================

def route_by_intent(
    state: AgentState
):
    """
    의도에 따라 경로 결정

    hybrid:
    vector_search + database_query 병렬 실행
    """

    intent = state.get(
        "intent",
        "general"
    )

    if intent == "general":

        return "general_answer"

    elif intent == "iot_consult":

        return "vector_search"

    elif intent == "database":

        return "database_query"

    elif intent == "hybrid":

        return [
            "vector_search",
            "database_query"
        ]

    else:

        return "general_answer"


def check_documents(
    state: AgentState
) -> str:
    """PDF 문서 평가 결과 라우팅"""

    grade = state.get(
        "document_grade",
        "relevant"
    )

    if grade == "relevant":

        return "combine_context"

    else:

        return "rewrite_query"


def check_db_results(
    state: AgentState
) -> str:
    """DB 결과 검증 라우팅"""

    valid = state.get(
        "db_valid",
        "valid"
    )

    if valid == "valid":

        return "combine_context"

    else:

        return "database_query"


def check_answer(
    state: AgentState
) -> str:
    """
    최종 답변 검증 결과 라우팅

    ok → 종료
    not ok → 한 번 재생성
    """

    valid = state.get(
        "answer_valid",
        "ok"
    )

    if valid == "ok":

        return "end"

    else:

        return "generate_answer"