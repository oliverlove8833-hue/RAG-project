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
당신은 IoT 디바이스 보안 점검 도우미의 질문 분석 담당입니다.

이전 대화 내용을 고려하여 현재 질문을
그 자체로 의미가 통하는 완전한 질문으로 재구성하세요.

예시:

- 이전:
  "집에 웹캠을 설치했는데"
  현재:
  "혼자 움직이는데 왜 그래?"
  재구성:
  "집에 설치한 웹캠이 혼자 움직이는데 해킹당한 건가요? 어떻게 확인하고 대처해야 하나요?"

- 이전:
  "스마트 도어락 비밀번호를 바꾸려고 해"
  현재:
  "어떻게 하는 게 안전해?"
  재구성:
  "스마트 도어락 비밀번호를 안전하게 변경하려면 어떻게 해야 하나요?"

- 이전:
  "공유기가 이상해"
  현재:
  "신고는 어디에 해?"
  재구성:
  "공유기가 해킹된 것 같은데 어디에 신고해야 하나요?"

- 이전:
  "IoT 기기 보안 점검 방법 알려줘"
  현재:
  "도어락도 해당돼?"
  재구성:
  "스마트 도어락도 IoT 기기 보안 점검 대상에 해당하나요? 어떻게 점검하나요?"

규칙:
- 질문의 원래 의미를 바꾸지 마세요.
- 없는 조건을 임의로 추가하지 마세요.
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
당신은 IoT 디바이스 보안 점검 도우미의 질문 분류 담당입니다.

질문을 다음 4가지 중 하나로 분류하세요.


1. general

일반적인 대화, 인사, 서비스 설명처럼
전문 자료 검색이 필요하지 않은 질문입니다.

예:
- "안녕하세요"
- "고마워"
- "너는 뭘 할 수 있어?"
- "어떤 도움을 줄 수 있어?"


2. iot_consult

보안 가이드나 기술문서를 바탕으로
스마트 기기의 이상 증상 원인, 보안 점검 방법,
대처 방법, 보호 원리 등을 설명해야 하는 질문입니다.

예:
- "집 웹캠이 혼자 움직이는데 해킹인가요?"
- "공유기 비밀번호를 바꿔야 하나요?"
- "스마트 도어락이 해킹당하면 어떻게 대처해야 하나요?"
- "IoT 기기의 보안 점검은 어떻게 하나요?"
- "스마트 기기를 안전하게 사용하려면 뭘 해야 하나요?"
- "JTAG를 왜 막아야 하나요?"
- "펌웨어 업데이트는 왜 중요한가요?"


3. database

구조화된 데이터베이스에서
보안 항목 목록, 기능 분류, 신고 기관 정보,
사고 유형, 개수, 비교 등
정확한 데이터를 조회하는 것이 중심인 질문입니다.

예:
- "IoT 해킹 신고는 어디에 하나요?"
- "보안 사고 유형에는 뭐가 있어?"
- "KISA 신고 전화번호가 뭐야?"
- "ESP32-H2의 보안 기능 목록을 보여줘"
- "보안 가이드라인이 총 몇 개야?"
- "하드웨어 관련 보안 항목을 알려줘"


4. hybrid

보안 가이드 문서의 구체적인 설명과
DB의 구조화된 정보를 모두 사용하는 것이 좋은 질문입니다.

특히 스마트 기기의 보안 점검, 대처 방법과 함께
신고 기관 정보나 구체적인 보안 항목 목록이
함께 필요한 경우 hybrid를 선택하세요.

예:
- "웹캠이 해킹된 것 같은데 어떻게 점검하고 어디에 신고해?"
- "스마트 도어락 보안 점검 방법이랑 관련 보안 항목 알려줘"
- "IoT 기기가 해킹당했을 때 대처 방법과 신고 절차를 알려줘"
- "공유기 보안 설정 방법이랑 관련 가이드라인 보여줘"
- "스마트 기기 보안 점검 체크리스트랑 사고 유형 알려줘"


판단 기준:

- 단순 인사/대화 → general
- 이상 증상, 보안 원리, 점검/대처 방법 설명 → iot_consult
- 신고 기관, 목록, 개수, 분류 등 정확한 데이터 조회 → database
- 보안 점검 + 신고/데이터 조회를 함께 요구 → hybrid


중요:
"해킹된 것 같은데 어떻게 하고 어디에 신고해?"
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
당신은 'IoT 디바이스 보안 점검 도우미'입니다.

집에서 사용하는 웹캠, 스마트 도어락, 공유기 같은 스마트 기기가
안전한지 궁금한 일반 사용자를 도와주는 AI입니다.

사용자가 일반적인 질문이나 서비스 소개를 요청하면
쉽고 친근하게 답변하세요.

이 서비스는 다음과 같은 도움을 줄 수 있습니다.

- 스마트 기기의 이상 증상 원인 파악 (웹캠이 혼자 움직임, 공유기가 느려짐 등)
- 기기별 보안 점검 방법 안내
- 해킹이 의심될 때 대처 방법
- 보안 사고 신고 기관 및 연락처 안내 (KISA 118, 경찰청 등)
- IoT 기기의 보안 설정 가이드
- 안전한 비밀번호 설정, 펌웨어 업데이트 등 기본 보안 수칙

전문용어는 최대한 쉽게 풀어서 설명하세요.
사용자가 IT를 잘 모르는 일반인이라고 생각하고 답변하세요.
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
당신은 'IoT 디바이스 보안 점검 도우미'입니다.

집에서 사용하는 웹캠, 스마트 도어락, 공유기 같은 스마트 기기의
보안을 점검하고 이상 증상에 대처하는 방법을 안내합니다.

사용자는 IT 전문가가 아닌 일반인입니다.
답변은 최대한 쉽고 친근한 한국어로 작성하세요.


사용자 질문:

{question}


아래 검색 결과만 근거로 답변하세요.

<context>

{context}

</context>


==================================================
가장 중요한 원칙
==================================================

이 서비스의 목적은
일반 사용자가 집에서 쓰는 스마트 기기(IoT 기기)를
안전하게 사용할 수 있도록 돕는 것입니다.

답변은 항상 사용자의 상황에 공감하면서
구체적인 행동 지침을 제시하세요.

"이렇게 하세요"라는 실천 가능한 조언을 주세요.


==================================================
이상 증상 질문 답변 방법
==================================================

사용자가 기기의 이상한 동작을 물어보면:

1. 공감 및 상황 설명
   사용자의 걱정에 공감하고,
   해당 증상이 왜 발생할 수 있는지 쉽게 설명

2. 즉시 할 수 있는 조치
   비전문가도 바로 실행할 수 있는 구체적인 대처 방법 안내
   (전원 끄기, 비밀번호 변경, 초기화, 네트워크 분리 등)

3. 보안 점검 방법
   검색 결과에 관련 보안 가이드가 있다면
   해당 기기의 보안 점검 포인트를 쉽게 안내

4. 추가 도움
   필요한 경우 신고 기관이나 전문가 도움을 받을 수 있는 곳 안내


예:

나쁜 답변:
"JTAG 포트를 비활성화하고 RDP Level 2를 적용하세요."

좋은 답변:
"웹캠이 혼자 움직인다면 외부에서 누군가 접근했을 가능성이 있습니다.
우선 웹캠의 전원을 뽑고, 공유기 비밀번호를 변경하세요.
그 다음 웹캠 앱에서 연결된 기기 목록을 확인해서
모르는 기기가 있으면 삭제하세요."


==================================================
보안 점검 질문 답변 방법
==================================================

사용자가 보안 점검 방법을 물어보면:

1. 기본 보안 수칙을 체크리스트 형태로 안내
2. 해당 기기에 맞는 구체적인 설정 방법 설명
3. 검색 결과의 보안 가이드 내용을 쉽게 풀어서 설명


기본 보안 수칙 예:
- 초기 비밀번호를 반드시 변경했는지
- 펌웨어(기기 소프트웨어)를 최신 버전으로 업데이트했는지
- 사용하지 않는 기능(원격 접속 등)을 껐는지
- 공유기 암호화 방식이 안전한지 (WPA2/WPA3)


==================================================
신고/도움 요청 질문 답변 방법
==================================================

사용자가 신고 방법을 물어보면:

1. 상황에 맞는 신고 기관을 안내
2. 연락처와 신고 방법을 구체적으로 안내
3. 신고 전에 준비할 것(증거 보존 등)을 안내


==================================================
쉬운 설명 규칙
==================================================

전문용어는 가능한 한 사용하지 마세요.

꼭 필요한 경우에만 괄호 안에 넣으세요:

"기기 소프트웨어(펌웨어)를 최신 버전으로 업데이트하세요."
"무선 암호화 방식(WPA2 이상)을 사용하세요."
"기기 내부를 점검하는 개발용 통로(JTAG)가 열려 있으면 위험합니다."


==================================================
근거 규칙
==================================================

- 반드시 context에 있는 정보를 중심으로 답하세요.

- context에 없는 구체적인 수치나 모델명을 만들어내지 마세요.

- 정보 자체가 부족하면:
  "현재 참고 자료만으로는 이 부분을 정확하게 안내하기 어렵습니다."
  라고 안내하세요.

- SQL이나 데이터베이스라는 내부 구현 표현은
  사용자에게 언급하지 마세요.

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
당신은 IoT 디바이스 보안 점검 도우미의
답변을 검증하는 전문가입니다.

사용자 질문과 참고 정보,
생성된 답변을 비교하세요.


검증 기준:


1. 근거

답변 내용이 참고 정보에 실제로 존재하는가?


2. 환각

참고 정보에 없는 구체적인 수치, 모델명,
연락처 등을 임의로 만들어내지 않았는가?


3. 실용성

사용자가 이상 증상을 물었다면
구체적인 대처 방법이 포함되어 있는가?

사용자가 보안 점검을 물었다면
실천 가능한 점검 방법이 포함되어 있는가?

사용자가 신고 방법을 물었다면
신고 기관과 연락처가 안내되어 있는가?


4. 쉬운 설명

전문용어만 나열하지 않고
IT를 잘 모르는 일반인도 이해할 수 있게 설명했는가?


5. 출처

PDF 자료를 사용했다면
가능한 범위에서 자료명과 페이지가 포함되어 있는가?


판단:

ok
- 질문에 맞고 근거가 있으며
  일반 사용자가 이해하고 실행할 수 있는 답변

not ok
- 근거 없는 내용이 있거나
- 질문에 대한 핵심 내용이 빠졌거나
- 전문용어만 나열하여 일반인이 이해하기 어려움


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