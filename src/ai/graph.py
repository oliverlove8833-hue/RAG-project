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
    # StateGraph 생성 (input state 명시)
    graph_builder = StateGraph(AgentState, input=InputState)

    # 노드 추가
    graph_builder.add_node("analyze_question", analyze_question)
    graph_builder.add_node("classify_intent", classify_intent)
    graph_builder.add_node("general_answer", general_answer)
    graph_builder.add_node("vector_search", vector_search)
    graph_builder.add_node("grade_documents", grade_documents)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("database_query", database_query)
    graph_builder.add_node("validate_db_result", validate_db_result)
    # hybrid에서 두 경로가 합류하므로, 대기 중인 작업이 모두 끝난 뒤 한 번만 실행
    graph_builder.add_node("combine_context", combine_context, defer=True)
    graph_builder.add_node("generate_answer", generate_answer)
    graph_builder.add_node("validate_answer", validate_answer)

    # 시작점 설정
    graph_builder.set_entry_point("analyze_question")

    # 질문 분석 후 의도 분류
    graph_builder.add_edge("analyze_question", "classify_intent")

    # 의도별 조건부 라우팅
    # 'hybrid'인 경우 route_by_intent가 리스트를 반환하여
    # vector_search와 database_query가 병렬로 실행된다
    graph_builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "general_answer": "general_answer",
            "vector_search": "vector_search",
            "database_query": "database_query",
        }
    )

    # 일반 답변은 검증으로 바로 이동
    graph_builder.add_edge("general_answer", "validate_answer")

    # PDF(RAG) 경로: 검색 → 평가 → (재작성 반복 | 통합)
    graph_builder.add_edge("vector_search", "grade_documents")
    graph_builder.add_conditional_edges(
        "grade_documents",
        check_documents,
        {
            "rewrite_query": "rewrite_query",
            "combine_context": "combine_context",
        }
    )
    graph_builder.add_edge("rewrite_query", "vector_search")

    # DB(Text2SQL) 경로: 조회 → 검증 → (재조회 | 통합)
    graph_builder.add_edge("database_query", "validate_db_result")
    graph_builder.add_conditional_edges(
        "validate_db_result",
        check_db_results,
        {
            "database_query": "database_query",
            "combine_context": "combine_context",
        }
    )

    # 통합 후 답변 생성
    graph_builder.add_edge("combine_context", "generate_answer")
    graph_builder.add_edge("generate_answer", "validate_answer")

    # 답변 검증 후 종료 또는 재생성
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


# 그래프 인스턴스 생성 (LangGraph Studio에서 사용)
graph = create_graph()
