from langgraph.graph import StateGraph, END
from ai.state import AgentState, InputState
from ai.nodes import (
    analyze_question,
    classify_intent,
    general_answer,
    vector_search,
    grade_documents,
    rewrite_query,
    database_query,
    validate_db_result,
    combine_context,
    generate_answer,
    validate_answer,
    route_by_intent,
    check_documents,
    check_db_results,
    check_answer,
)


def create_graph():
    """
    IoT 디바이스 하드웨어·보안 통합 점검 에이전트 그래프 생성

    처리 흐름:
    1. 사용자 질문 분석
    2. 질문 의도 분류
    3. 의도에 따라
       - 일반 답변
       - PDF 문서 검색(RAG)
       - 구조화 데이터 조회(Text2SQL)
       - PDF + DB 통합 검색
    4. 검색 결과 통합
    5. 하드웨어 설계 + 보안 관점의 최종 답변 생성
    6. 답변 검증
    """

    # StateGraph 생성
    graph_builder = StateGraph(
        AgentState,
        input=InputState
    )

    # ========================================
    # 노드 등록
    # ========================================

    # 질문 분석 및 의도 분류
    graph_builder.add_node(
        "analyze_question",
        analyze_question
    )

    graph_builder.add_node(
        "classify_intent",
        classify_intent
    )

    # 일반 질문 답변
    graph_builder.add_node(
        "general_answer",
        general_answer
    )

    # PDF(RAG) 검색 경로
    graph_builder.add_node(
        "vector_search",
        vector_search
    )

    graph_builder.add_node(
        "grade_documents",
        grade_documents
    )

    graph_builder.add_node(
        "rewrite_query",
        rewrite_query
    )

    # DB(Text2SQL) 조회 경로
    graph_builder.add_node(
        "database_query",
        database_query
    )

    graph_builder.add_node(
        "validate_db_result",
        validate_db_result
    )

    # PDF + DB 결과 통합
    # hybrid 경로에서 두 작업이 모두 끝난 뒤 실행
    graph_builder.add_node(
        "combine_context",
        combine_context,
        defer=True
    )

    # 최종 답변 생성 및 검증
    graph_builder.add_node(
        "generate_answer",
        generate_answer
    )

    graph_builder.add_node(
        "validate_answer",
        validate_answer
    )

    # ========================================
    # 시작점
    # ========================================

    graph_builder.set_entry_point(
        "analyze_question"
    )

    # 질문 분석 → 의도 분류
    graph_builder.add_edge(
        "analyze_question",
        "classify_intent"
    )

    # ========================================
    # 의도별 라우팅
    # ========================================
    #
    # general
    # → 일반적인 대화 / 서비스 안내
    #
    # iot_consult
    # → PDF 기반 IoT 하드웨어 설계·보안 지식 검색
    #
    # database
    # → Supabase의 구조화된 MCU/IoT 데이터 조회
    #
    # hybrid
    # → PDF 검색 + DB 조회 병렬 실행
    #

    graph_builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "general_answer": "general_answer",
            "vector_search": "vector_search",
            "database_query": "database_query",
        }
    )

    # ========================================
    # 일반 답변 경로
    # ========================================

    graph_builder.add_edge(
        "general_answer",
        "validate_answer"
    )

    # ========================================
    # PDF(RAG) 경로
    # ========================================
    #
    # vector_search
    #      ↓
    # grade_documents
    #      ├─ relevant → combine_context
    #      └─ not enough → rewrite_query
    #                          ↓
    #                     vector_search
    #

    graph_builder.add_edge(
        "vector_search",
        "grade_documents"
    )

    graph_builder.add_conditional_edges(
        "grade_documents",
        check_documents,
        {
            "rewrite_query": "rewrite_query",
            "combine_context": "combine_context",
        }
    )

    graph_builder.add_edge(
        "rewrite_query",
        "vector_search"
    )

    # ========================================
    # DB(Text2SQL) 경로
    # ========================================
    #
    # database_query
    #      ↓
    # validate_db_result
    #      ├─ valid → combine_context
    #      └─ not valid → database_query
    #

    graph_builder.add_edge(
        "database_query",
        "validate_db_result"
    )

    graph_builder.add_conditional_edges(
        "validate_db_result",
        check_db_results,
        {
            "database_query": "database_query",
            "combine_context": "combine_context",
        }
    )

    # ========================================
    # 통합 답변 생성
    # ========================================

    graph_builder.add_edge(
        "combine_context",
        "generate_answer"
    )

    graph_builder.add_edge(
        "generate_answer",
        "validate_answer"
    )

    # ========================================
    # 최종 답변 검증
    # ========================================

    graph_builder.add_conditional_edges(
        "validate_answer",
        check_answer,
        {
            "generate_answer": "generate_answer",
            "end": END,
        }
    )

    # 그래프 컴파일
    graph = graph_builder.compile()

    return graph


# LangGraph Studio에서 사용하는 그래프 인스턴스
graph = create_graph()