from typing import Optional
from langgraph.graph import MessagesState


class InputState(MessagesState):
    """
    사용자 입력 상태

    MessagesState를 상속하여
    대화 메시지를 입력으로 받는다.
    """
    pass


class AgentState(MessagesState):

    # =====================================================
    # 사용자 질문 및 의도
    # =====================================================

    # 사용자가 입력한 원본 질문
    question: Optional[str]

    # 이전 대화 맥락을 반영하여
    # 완전한 질문으로 재구성한 질문
    analyzed_question: Optional[str]

    # 질문 의도 분류 결과
    #
    # 'general'
    #   → 일반 대화 / 에이전트 소개
    #
    # 'iot_consult'
    #   → IoT 보안 가이드 PDF 문서 검색
    #
    # 'database'
    #   → 신고 기관·사고 유형·보안 점검 항목 DB 조회
    #
    # 'hybrid'
    #   → PDF + DB를 함께 사용
    intent: Optional[str]


    # =====================================================
    # 벡터 검색(RAG) 관련
    # =====================================================

    # Qdrant에서 검색된 IoT 보안 가이드 문서
    # Document 리스트
    vector_results: Optional[list]

    # 검색 결과가 부족할 경우
    # 다시 작성한 검색 쿼리
    rewritten_query: Optional[str]

    # 검색된 문서가 질문에 충분한지 평가
    #
    # 'relevant'
    # 'not enough'
    document_grade: Optional[str]


    # =====================================================
    # 데이터베이스(Text2SQL) 관련
    # =====================================================

    # 자연어 질문으로 생성된 PostgreSQL 쿼리
    sql_query: Optional[str]

    # Supabase DB 조회 결과
    #
    # 신고 기관, 사고 유형,
    # 보안 가이드라인,
    # 보안 점검 항목 등
    db_results: Optional[str]

    # DB 조회 결과 검증
    #
    # 'valid'
    # 'not valid'
    db_valid: Optional[str]


    # =====================================================
    # 통합 Context
    # =====================================================

    # PDF 검색 결과와
    # DB 조회 결과를 합친 Context
    #
    # 최종 보안 점검 답변 생성에 사용
    combined_context: Optional[str]


    # =====================================================
    # 답변 검증
    # =====================================================

    # 최종 답변 검증 결과
    #
    # 'ok'
    # 'not ok'
    answer_valid: Optional[str]


    # =====================================================
    # 오류 및 재시도 관리
    # =====================================================

    # PDF 벡터 검색 재시도 횟수
    #
    # hybrid에서는 DB 경로와 병렬로 동작하므로
    # DB retry와 따로 관리
    vector_retry: Optional[int]

    # Text2SQL / DB 조회 재시도 횟수
    db_retry: Optional[int]

    # 최종 답변 재생성 횟수
    answer_retry: Optional[int]

    # SQL 실행 오류 등
    # 경로 수행 중 발생한 오류 메시지
    error: Optional[str]