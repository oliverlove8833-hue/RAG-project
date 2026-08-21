from typing import Optional
from langgraph.graph import MessagesState


class InputState(MessagesState): # messagesState를 상속받아 InputState 정의
    pass


class AgentState(MessagesState):
    # 사용자 질문 및 의도
    question: Optional[str] # 사용자의 원본 질문
    analyzed_question: Optional[str] # 맥락을 반영해 재구성한 완전한 질문
    intent: Optional[str] # 질문 의도 분류 결과 ('general', 'labor_consult', 'database', 'hybrid')

    # 벡터 검색 관련
    vector_results: Optional[list] # Qdrant 벡터 검색 결과 (Document 리스트)
    rewritten_query: Optional[str] # 재작성된 검색 쿼리 (벡터 검색용)
    document_grade: Optional[str] # 검색 결과 평가 ('relevant', 'not enough')

    # 데이터베이스 검색 관련
    sql_query: Optional[str] # 생성된 SQL 쿼리
    db_results: Optional[str] # 데이터베이스 쿼리 실행 결과
    db_valid: Optional[str] # DB 결과 검증 ('valid', 'not valid')

    # 통합 컨텍스트
    combined_context: Optional[str] # PDF + DB 결과를 통합한 컨텍스트

    # 답변 검증
    answer_valid: Optional[str] # 답변 검증 결과 ('ok', 'not ok')

    # 오류 처리 (경로별로 분리: hybrid에서 병렬 실행되므로 카운터를 공유하면 안 됨)
    vector_retry: Optional[int] # 벡터 검색 재시도 횟수
    db_retry: Optional[int] # DB 조회 재시도 횟수
    answer_retry: Optional[int] # 답변 재생성 횟수
    error: Optional[str] # 오류 메시지
