from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
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


class VectorSearchQuery(BaseModel):
    """벡터 검색을 위한 쿼리 분석 결과"""
    optimized_query: str = Field(
        description="검색에 최적화된 쿼리. 핵심 키워드를 포함하고 명확하게 작성."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="선택된 카테고리 리스트 (1-2개). 명확하게 관련 있는 카테고리만 선택. 애매하거나 불확실한 경우 null 반환. 가능한 값: 근로계약_서류관리, 임금_수당_금품, 근로시간_휴게, 휴일_연차휴가, 임신_출산_육아_모성보호, 취업규칙_사업장규정, 퇴직_퇴직급여, 직장내_권리보호, 최저임금, 노사관계_노사협의회"
    )


def get_cached_retriever():
    """캐시된 retriever 인스턴스 반환 (lazy initialization)"""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def get_cached_text2sql_engine():
    """캐시된 text2sql_engine 인스턴스 반환 (lazy initialization)"""
    global _text2sql_engine
    if _text2sql_engine is None:
        _text2sql_engine = get_text2sql_engine()
    return _text2sql_engine


def analyze_question(state: AgentState) -> AgentState:
    """
    질문을 전처리하고 대화 맥락을 정리하는 노드

    이전 대화 맥락을 반영하여 그 자체로 완전한 질문을 만든다.
    이후 모든 노드는 analyzed_question을 사용한다.

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # messages에서 질문 추출
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided")

    # 마지막 사용자 메시지를 질문으로 사용
    question = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    # 이전 대화가 없으면 그대로 사용
    if len(messages) <= 1:
        return {
            "question": question,
            "analyzed_question": question
        }

    system_prompt = """
당신은 질문 분석 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "주휴수당 조건이 뭐야?" → 현재: "알바도 받아?" → 재구성: "아르바이트생도 주휴수당을 받을 수 있어?"
- 이전: "임금체불 담당 부서 알려줘" → 현재: "전화번호는?" → 재구성: "임금체불 담당 부서의 전화번호는?"
- 이전: "표준근로계약서" → 현재: "몇 쪽이야?" → 재구성: "표준근로계약서 서식은 몇 쪽에 있어?"

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
만약 현재 질문이 이미 완전하다면 그대로 반환하세요.
"""

    conversation = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(conversation)
    analyzed = response.content.strip()

    print(f"[질문 분석] {question} → {analyzed}")

    return {
        "question": question,
        "analyzed_question": analyzed
    }


def classify_intent(state: AgentState) -> AgentState:
    """
    사용자 질문의 의도를 분류하는 노드

    분류 결과:
    - 'general': 일반적인 대화나 인사
    - 'labor_consult': 제도 설명·대응 방법 (PDF 문서 검색)
    - 'database': 담당 기관·서식 위치 (DB 조회)
    - 'hybrid': 설명과 기관 정보가 모두 필요 (PDF + DB 병렬)

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    question = state.get("analyzed_question", "")

    system_prompt = """
당신은 노동자의 근로기준법 상담 질문을 분류하는 전문가입니다.

질문을 다음 4가지 중 하나로 분류하세요:

1. 'general' - 일반적인 대화, 인사, 서비스 안내
   예: "안녕하세요", "고마워", "너는 뭘 할 수 있어?"

2. 'labor_consult' - 근로기준법 제도의 내용, 조건, 대응 방법을 묻는 상담성 질문
   예: "주휴수당을 받을 수 있는 조건이 무엇인가요?"
       "임금을 받지 못했을 때 어떻게 해야 하나요?"
       "휴게시간은 몇 시간 일하면 받을 수 있나요?"

3. 'database' - 담당 부서, 전화번호, 관련 법령, 서식 위치 등 정확한 정보 조회
   예: "임금체불은 어디에 문의해야 하나요?"
       "표준근로계약서 서식은 몇 쪽에 있나요?"
       "근로기준정책과 전화번호는?"

4. 'hybrid' - 제도 설명과 담당 기관 정보가 모두 필요한 질문
   예: "아르바이트비를 못 받았는데 어떻게 해야 하고 어디에 문의하나요?"
       "부당하게 해고당했는데 대응 방법과 상담 기관을 알려주세요"

판단 기준:
- '어떻게/왜/조건'을 물으면 labor_consult
- '어디/누구/번호/몇 쪽'을 물으면 database
- 두 가지를 한 문장에서 함께 물으면 hybrid

반드시 'general', 'labor_consult', 'database', 'hybrid' 중 하나만 답변하세요.
다른 설명 없이 분류 결과만 반환하세요.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    intent = response.content.strip().lower()

    # 유효한 의도인지 확인
    if intent not in ['general', 'labor_consult', 'database', 'hybrid']:
        intent = 'general'

    print(f"[의도 분류] {intent}")

    return {
        "intent": intent
    }


def general_answer(state: AgentState) -> AgentState:
    """
    일반적인 질문에 직접 답변하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 노동자에게 근로기준법을 안내하는 친절한 상담 도우미입니다.
사용자의 질문에 자연스럽고 도움이 되는 답변을 제공하세요.

서비스 소개가 필요하면 이렇게 안내하세요:
- 근로계약, 임금, 근로시간, 휴가, 퇴직금 등 근로기준법 관련 질문에 답할 수 있습니다
- 담당 기관 연락처와 관련 서식 위치도 안내할 수 있습니다
- 법률 자문은 아니며, 실제 분쟁은 고용노동부 상담센터(국번없이 1350) 상담이 필요합니다
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


def vector_search(state: AgentState) -> AgentState:
    """
    Qdrant 벡터 검색을 수행하는 노드 (PDF 문서 검색)

    1. LLM으로 질문 분석 (최적화된 쿼리 + 카테고리 추출)
    2. 병렬 벡터 검색 수행

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 재작성된 쿼리가 있으면 사용, 없으면 분석된 질문 사용
    original_query = state.get("rewritten_query") or state.get("analyzed_question", "")

    # LLM으로 쿼리 분석 및 카테고리 추출 (Structured Output)
    # 시스템 프롬프트: 역할 정의 및 카테고리 설명
    system_prompt = """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 분석하여 벡터 검색에 최적화된 쿼리를 생성하고, 적절한 카테고리를 선택하는 역할을 수행합니다.

사용 가능한 카테고리:
- 근로계약_서류관리: 근로계약서 작성·교부, 근로조건 서면명시, 기간제·단시간 근로자 계약, 근로자명부, 계약서류 보존, 임금대장 관련
- 임금_수당_금품: 임금 지급, 임금명세서, 금품청산, 휴업수당, 연장·야간·휴일근로 수당, 임금체불 관련
- 근로시간_휴게: 법정근로시간, 소정근로시간, 연장근로 한도, 초과근로, 휴게시간, 대기시간 관련
- 휴일_연차휴가: 주휴일, 주휴수당, 공휴일, 대체공휴일, 유급휴일, 연차유급휴가, 연차 미사용수당 관련
- 임신_출산_육아_모성보호: 임산부 보호, 여성·연소자 근로, 출산휴가, 배우자 출산휴가, 육아휴직, 육아기 근로시간 단축 관련
- 취업규칙_사업장규정: 취업규칙 작성·신고·변경, 법령 및 단체협약 준수 관련
- 퇴직_퇴직급여: 퇴직금, 확정급여형·확정기여형 퇴직연금, 중소기업퇴직연금기금제도 관련
- 직장내_권리보호: 직장 내 괴롭힘, 직장 내 성희롱, 고용상 성차별, 비정규직 차별 관련
- 최저임금: 최저임금 적용, 최저임금 계산, 수습·인턴 최저임금, 산입범위, 주지의무 관련
- 노사관계_노사협의회: 노사협의회 설치와 운영, 회의, 고충처리, 근로자위원·사용자위원 관련

카테고리 선택 규칙:
1. 명확하게 관련 있는 카테고리를 1-2개 선택합니다
2. 여러 카테고리와 관련될 수 있으면 최대 2개까지 선택
3. 애매하거나 확신이 없으면 반드시 null을 반환 (잘못된 카테고리보다 null이 나음)
4. 억지로 카테고리를 선택하지 말고, 확실한 경우에만 선택

출력 지침:
1. optimized_query: 문서에 실제로 쓰이는 법률 용어를 포함한 쿼리로 최적화
   (예: "월급 안 줌" → "임금체불 금품청산 지급 의무")
2. categories: 명확하게 관련 있는 카테고리 1-2개를 리스트로 반환. 불확실하면 null"""

    # 유저 프롬프트: 실제 질문
    user_prompt = f"다음 질문을 분석해주세요:\n\n{original_query}"

    # 메시지 객체 생성 (Structured Output용)
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Structured Output으로 LLM 호출
    structured_llm = llm.with_structured_output(VectorSearchQuery)
    query_analysis = structured_llm.invoke(llm_messages)

    optimized_query = query_analysis.optimized_query
    categories = query_analysis.categories

    print(f"[벡터 검색 쿼리 분석]")
    print(f"  원본 쿼리: {original_query}")
    print(f"  최적화된 쿼리: {optimized_query}")
    print(f"  선택된 카테고리: {categories}")

    # 병렬 벡터 검색 수행 (카테고리 필터 적용)
    retriever = get_cached_retriever()
    results = retriever.search(optimized_query, k=3, score_threshold=0.5, categories=categories)

    return {
        "vector_results": results
    }


def grade_documents(state: AgentState) -> AgentState:
    """
    검색된 문서가 질문에 답하기에 충분한지 평가하는 노드

    평가 기준:
    - 질문과의 관련성
    - 답변 가능 여부
    - 출처 신뢰성

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    question = state.get("analyzed_question", "")
    results = state.get("vector_results", [])
    vector_retry = state.get("vector_retry", 0)

    # 검색 결과가 아예 없으면 LLM 호출 없이 바로 부족 판정
    if not results:
        print("[문서 평가] not enough (검색 결과 없음)")
        return {"document_grade": "not enough"}

    # 재시도 한도에 도달했으면 있는 결과로 진행
    if vector_retry >= 2:
        print("[문서 평가] relevant (재시도 한도 도달)")
        return {"document_grade": "relevant"}

    # 검색된 문서 요약
    doc_summary = "\n\n".join([
        f"[문서 {i}] {doc.page_content[:400]}"
        for i, doc in enumerate(results, 1)
    ])

    system_prompt = """
당신은 검색 결과를 평가하는 전문가입니다.

주어진 문서들이 사용자의 질문에 답하기에 충분한지 판단하세요.

평가 기준:
1. 질문과의 관련성 - 문서가 질문의 주제를 다루고 있는가
2. 답변 가능 여부 - 이 문서만으로 질문에 답할 수 있는가
3. 출처 신뢰성 - 근거로 제시할 만한 내용인가

판단 결과:
- 'relevant' - 답변하기에 충분함
- 'not enough' - 관련성이 낮거나 정보가 부족함

반드시 'relevant' 또는 'not enough' 중 하나만 답변하세요.
다른 설명 없이 평가 결과만 반환하세요.
"""

    user_prompt = f"질문:\n{question}\n\n검색된 문서:\n{doc_summary}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    grade = response.content.strip().lower()

    if grade not in ['relevant', 'not enough']:
        grade = 'relevant'

    print(f"[문서 평가] {grade}")

    return {
        "document_grade": grade
    }


def rewrite_query(state: AgentState) -> AgentState:
    """
    검색 결과가 부족할 때 쿼리를 재작성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    question = state.get("analyzed_question", "")
    previous = state.get("rewritten_query")

    system_prompt = """
당신은 검색 쿼리 최적화 전문가입니다.

사용자의 질문이 검색 결과를 얻지 못했습니다. 질문을 다시 작성하세요.

이 문서는 고용노동부가 사업주를 대상으로 작성한 노무관리 가이드북입니다.
일상적인 표현을 문서에 실제로 등장하는 법률 용어로 바꾸는 것이 가장 효과적입니다.

최적화 방법:
- 일상어를 법률 용어로 변환 (예: "월급을 안 줘요" → "임금 체불 금품청산 지급 의무")
- 동의어나 관련 용어 추가
- 질문을 더 구체적이거나 더 일반적으로 변경
- 핵심 키워드 강조

재작성된 쿼리만 반환하세요. 설명은 포함하지 마세요.
"""

    user_prompt = f"원본 질문: {question}"
    if previous:
        user_prompt += f"\n이미 시도한 쿼리(다르게 작성하세요): {previous}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    rewritten = response.content.strip()

    print(f"[쿼리 재작성] {rewritten}")

    return {
        "rewritten_query": rewritten,
        "vector_retry": state.get("vector_retry", 0) + 1
    }


def database_query(state: AgentState) -> AgentState:
    """
    Text2SQL을 수행하여 데이터베이스를 조회하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    question = state.get("analyzed_question", "")
    previous_error = state.get("error")

    # Text2SQL 실행
    text2sql_engine = get_cached_text2sql_engine()
    result = text2sql_engine.query(question, previous_error=previous_error)

    print(f"[SQL 생성] {result['sql_query']}")

    return {
        "sql_query": result["sql_query"],
        "db_results": result["result"],
        "error": result["error"],
        "db_retry": state.get("db_retry", 0) + 1
    }


def validate_db_result(state: AgentState) -> AgentState:
    """
    데이터베이스 조회 결과를 검증하는 노드

    검증 기준:
    - 결과 존재 여부
    - 컬럼/값 타당성
    - 질문 의도 부합 여부

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    error = state.get("error")
    result = state.get("db_results")
    db_retry = state.get("db_retry", 0)

    text2sql_engine = get_cached_text2sql_engine()

    # 오류가 있거나 결과가 비어 있으면 재시도 대상
    if error or not result or text2sql_engine.is_empty_result(result):
        # 재시도 한도에 도달했으면 그대로 진행 (오류 메시지 포함)
        if db_retry >= 2:
            print("[DB 검증] valid (재시도 한도 도달)")
            return {"db_valid": "valid"}

        print(f"[DB 검증] not valid ({'오류' if error else '결과 없음'})")
        return {"db_valid": "not valid"}

    print("[DB 검증] valid")

    return {
        "db_valid": "valid"
    }


def combine_context(state: AgentState) -> AgentState:
    """
    PDF 검색 결과와 DB 조회 결과를 하나의 컨텍스트로 통합하는 노드

    통합 내용:
    - PDF 본문과 출처(페이지)
    - 관련 서식/페이지
    - 관련 법령 및 담당 부서 정보

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    context_parts = []

    # 벡터 검색 결과가 있으면 추가
    if state.get("vector_results"):
        docs = state["vector_results"]
        context_parts.append("관련 문서:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "알 수 없음")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "")

            # 출처 정보 구성
            source_info = f"출처: {source}, 페이지: {page}"
            if category:
                source_info += f", 카테고리: {category}"

            context_parts.append(f"\n[문서 {i}] {source_info}\n{doc.page_content}")

    # DB 검색 결과가 있으면 추가
    if state.get("db_results"):
        context_parts.append(f"\n\n데이터베이스 조회 결과:\n{state['db_results']}")
        if state.get("sql_query"):
            context_parts.append(f"\n실행된 SQL:\n{state['sql_query']}")

    combined = "\n".join(context_parts) if context_parts else "(참고 정보 없음)"

    print(f"[컨텍스트 통합] 문서 {len(state.get('vector_results') or [])}건 / DB {'있음' if state.get('db_results') else '없음'}")

    return {
        "combined_context": combined
    }


def generate_answer(state: AgentState) -> AgentState:
    """
    통합된 컨텍스트를 바탕으로 최종 답변을 생성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])
    context = state.get("combined_context", "(참고 정보 없음)")

    system_prompt = f"""
당신은 노동자에게 근로기준법을 안내하는 상담 전문가입니다.
상담자는 법을 잘 모르는 노동자이므로, 어려운 법률 용어는 쉬운 말로 풀어서 설명하세요.

다음 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요:

<context>
{context}
</context>

답변 시 다음 규칙을 따르세요:
- 주어진 정보를 자연스럽고 간결하게 전달하세요
- 참고 문서는 사업주(사장님)를 대상으로 쓰였습니다. "사용자는 ~해야 한다"는 내용은 "사장님은 ~할 의무가 있으니 요구할 수 있습니다"처럼 노동자 입장으로 바꿔서 설명하세요
- 금액, 시간, 일수 등의 숫자는 주어진 정보에 적힌 그대로 인용하고 직접 계산하지 마세요
- 구체적인 조항, 과태료, 부서명, 전화번호 등의 정보를 명확히 포함하세요
- 문서와 데이터베이스 결과가 모두 있으면 제도 설명을 먼저, 문의할 기관과 연락처를 나중에 안내하세요
- 문서를 인용한 경우 답변 끝에 출처와 페이지 번호를 표시하세요
- 정보가 정말로 없는 경우에만 "제가 가진 자료에는 없는 내용입니다"라고 말하세요
- 상황에 따라 달라질 수 있는 문제라면 고용노동부 상담센터(국번없이 1350) 문의를 안내하세요
- 노동자에게 도움이 되는 친절하고 자연스러운 어조로 답변하세요
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    # 재생성인 경우 이전 AI 답변을 제거하고 새 답변으로 교체
    # (제거하지 않으면 AI 메시지가 두 개 쌓여 화면에서 답변이 바뀌어 보인다)
    updates = []
    if state.get("answer_retry", 0) > 0 and messages and messages[-1].type == "ai":
        updates.append(RemoveMessage(id=messages[-1].id))
    updates.append(AIMessage(content=answer))

    return {
        "messages": updates
    }


def validate_answer(state: AgentState) -> AgentState:
    """
    생성된 답변의 근거와 출처를 검증하는 노드

    검증 기준:
    - 근거 존재 여부
    - 출처(페이지/서식) 명시
    - 답변 신뢰성

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    intent = state.get("intent", "general")
    answer_retry = state.get("answer_retry", 0)
    messages = state.get("messages", [])

    # 마지막 AI 답변을 그대로 다시 내보낸다.
    # 종료 노드가 메시지를 반환하지 않으면 Studio 화면에서
    # 스트리밍된 답변이 사라지므로, 같은 id로 재전송해 유지시킨다.
    last_ai = [messages[-1]] if messages and messages[-1].type == "ai" else []

    # 일반 대화는 근거 검증 대상이 아님
    if intent == "general":
        return {"answer_valid": "ok", "messages": last_ai}

    # 재생성 한도에 도달했으면 통과
    if answer_retry >= 1:
        print("[답변 검증] ok (재생성 한도 도달)")
        return {"answer_valid": "ok", "messages": last_ai}

    answer = messages[-1].content if messages else ""
    context = state.get("combined_context", "")

    system_prompt = """
당신은 답변의 신뢰성을 검증하는 전문가입니다.

주어진 참고 정보와 답변을 비교하여 답변이 적절한지 판단하세요.

검증 기준:
1. 근거 존재 여부 - 답변의 내용이 참고 정보에 실제로 있는가
2. 출처 명시 - 문서를 인용했다면 출처와 페이지가 표시되어 있는가
3. 답변 신뢰성 - 참고 정보에 없는 내용을 지어내지 않았는가

판단 결과:
- 'ok' - 근거가 있고 신뢰할 수 있음
- 'not ok' - 근거가 없거나 출처 표시가 누락됨

참고 정보에 내용이 없어서 "자료에 없다"고 답한 경우는 'ok'입니다.

반드시 'ok' 또는 'not ok' 중 하나만 답변하세요.
다른 설명 없이 검증 결과만 반환하세요.
"""

    user_prompt = f"참고 정보:\n{context[:3000]}\n\n생성된 답변:\n{answer[:2000]}"

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(llm_messages)
    valid = response.content.strip().lower()

    if valid not in ['ok', 'not ok']:
        valid = 'ok'

    print(f"[답변 검증] {valid}")

    return {
        "answer_valid": valid,
        "answer_retry": answer_retry + 1,
        "messages": last_ai
    }


def route_by_intent(state: AgentState):
    """
    의도에 따라 다음 노드를 결정하는 라우팅 함수

    'hybrid'인 경우 리스트를 반환하여 vector_search와 database_query를
    병렬로 실행한다.

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름 (또는 병렬 실행할 노드 이름 리스트)
    """
    intent = state.get("intent", "general")

    if intent == "general":
        return "general_answer"
    elif intent == "labor_consult":
        return "vector_search"
    elif intent == "database":
        return "database_query"
    elif intent == "hybrid":
        return ["vector_search", "database_query"]
    else:
        return "general_answer"


def check_documents(state: AgentState) -> str:
    """
    문서 평가 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    grade = state.get("document_grade", "relevant")

    if grade == "relevant":
        return "combine_context"
    else:
        return "rewrite_query"


def check_db_results(state: AgentState) -> str:
    """
    데이터베이스 검증 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    valid = state.get("db_valid", "valid")

    if valid == "valid":
        return "combine_context"
    else:
        return "database_query"


def check_answer(state: AgentState) -> str:
    """
    답변 검증 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    valid = state.get("answer_valid", "ok")

    # 검증 결과는 상태에 기록하되 답변을 다시 생성하지는 않는다.
    # 재생성하면 이미 출력된 AI 메시지를 교체하게 되어
    # Studio 화면에서 답변이 나타났다가 사라지는 문제가 발생한다.
    # 재생성을 켜려면 아래 return을 주석 처리하고 그 아래 분기를 사용한다.
    return "end"

    if valid == "ok":
        return "end"
    else:
        return "generate_answer"
