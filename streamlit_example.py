import os
import sys
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import streamlit as st
from dotenv import load_dotenv
from ai import create_graph

# 환경 변수 로드
load_dotenv()

graph = create_graph()

# 워크플로 노드 → 사용자에게 보여줄 진행 상황 문구
NODE_LABELS = {
    "analyze_question": "질문 분석",
    "classify_intent": "의도 분류",
    "general_answer": "일반 답변 생성",
    "vector_search": "기술문서 벡터 검색",
    "grade_documents": "검색 문서 평가",
    "rewrite_query": "검색 쿼리 재작성",
    "database_query": "구조화 데이터 조회 (Text2SQL)",
    "validate_db_result": "DB 결과 검증",
    "combine_context": "검색 결과 통합",
    "generate_answer": "최종 답변 생성",
    "validate_answer": "답변 검증",
}

# 의도 분류 결과 → 한글 라벨
INTENT_LABELS = {
    "general": "💬 일반 대화",
    "iot_consult": "📄 기술문서 검색",
    "database": "🗄️ DB 조회",
    "hybrid": "🔀 하이브리드 (문서+DB)",
}


def node_progress_detail(node_name: str, update: dict) -> str:
    """노드 실행 결과에서 진행 상황에 함께 보여줄 요약 정보 추출"""
    if not isinstance(update, dict):
        return ""

    if node_name == "classify_intent":
        intent = update.get("intent")
        return INTENT_LABELS.get(intent, intent or "")

    if node_name == "vector_search":
        return f"{len(update.get('vector_results') or [])}건 검색됨"

    if node_name == "grade_documents":
        grade = update.get("document_grade")
        return "충분함" if grade == "relevant" else "자료 부족 → 재검색"

    if node_name == "rewrite_query":
        return update.get("rewritten_query", "")

    if node_name == "database_query":
        return "SQL 실행 오류" if update.get("error") else "SQL 실행 완료"

    if node_name == "validate_db_result":
        valid = update.get("db_valid")
        return "결과 유효" if valid == "valid" else "결과 없음 → 재조회"

    if node_name == "validate_answer":
        valid = update.get("answer_valid")
        return "통과" if valid == "ok" else "미흡 → 답변 재생성"

    return ""


def run_workflow(input_messages: list, container) -> dict:
    """
    워크플로를 스트리밍 실행하며 진행 상황을 표시하고
    최종 상태를 반환
    """
    final_state = {}
    node_counts = {}

    with container.status("워크플로 실행 중...", expanded=True) as status:
        for mode, chunk in graph.stream(
            {"messages": input_messages},
            stream_mode=["updates", "values"]
        ):
            # 노드 실행이 끝날 때마다 진행 상황 표시
            if mode == "updates":
                for node_name, update in chunk.items():
                    label = NODE_LABELS.get(node_name, node_name)
                    detail = node_progress_detail(node_name, update)

                    node_counts[node_name] = (
                        node_counts.get(node_name, 0) + 1
                    )

                    st.markdown(
                        f"✅ **{label}**"
                        + (f" — {detail}" if detail else "")
                    )

            # 매 스텝의 전체 상태 스냅샷 (마지막 것이 최종 상태)
            else:
                final_state = chunk

        status.update(
            label="워크플로 완료",
            state="complete",
            expanded=False
        )

    # 노드 실행 횟수는 상태에 없는 값이므로 별도로 추가
    final_state["node_counts"] = node_counts

    return final_state


def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_message(role: str, content: str, workflow_info: dict = None):
    """메시지 표시"""
    with st.chat_message(role):
        st.markdown(content)

        # 워크플로 정보가 있으면 표시 (assistant 메시지에만)
        if role == "assistant" and workflow_info:
            display_workflow_info(workflow_info)


def display_workflow_info(result: dict):
    """워크플로 정보 표시"""
    with st.expander("🔍 워크플로 정보"):
        intent = result.get("intent")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("의도", INTENT_LABELS.get(intent, intent or "N/A"))

        with col2:
            st.metric("검색된 문서", len(result.get("vector_results") or []))

        with col3:
            st.metric("DB 조회", "수행됨" if result.get("db_results") else "없음")

        # 재구성된 질문 (이전 대화 맥락 반영 결과)
        question = result.get("question")
        analyzed = result.get("analyzed_question")

        if analyzed and analyzed != question:
            st.info(f"💡 맥락을 반영해 재구성된 질문: {analyzed}")

        # 각 단계 검증 결과
        st.markdown("#### ✅ 단계별 검증")

        grade_cols = st.columns(3)

        with grade_cols[0]:
            grade = result.get("document_grade")
            st.caption(
                "📄 문서 평가: "
                + ("충분함" if grade == "relevant"
                   else "자료 부족" if grade else "미수행")
            )

        with grade_cols[1]:
            db_valid = result.get("db_valid")
            st.caption(
                "🗄️ DB 검증: "
                + ("유효" if db_valid == "valid"
                   else "결과 없음" if db_valid else "미수행")
            )

        with grade_cols[2]:
            answer_valid = result.get("answer_valid")
            st.caption(
                "📝 답변 검증: "
                + ("통과" if answer_valid == "ok"
                   else "미흡" if answer_valid else "미수행")
            )

        # 실행된 노드와 반복 횟수 (스트리밍 중 집계한 값)
        node_counts = result.get("node_counts")

        if node_counts:
            st.markdown("#### 🔁 실행된 노드")
            st.markdown(
                " → ".join(
                    NODE_LABELS.get(name, name)
                    + (f" ×{count}" if count > 1 else "")
                    for name, count in node_counts.items()
                )
            )

        # 벡터 검색 결과 상세 표시
        if result.get("vector_results"):
            st.markdown("#### 📄 검색된 문서")
            for i, doc in enumerate(result["vector_results"], 1):
                with st.expander(f"문서 {i}: {doc.metadata.get('source', '알 수 없음')}"):
                    # 메타데이터 표시
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.caption(f"📖 페이지: {doc.metadata.get('page', 'N/A')}")
                    with meta_cols[1]:
                        if doc.metadata.get('category'):
                            st.caption(f"🏷️ 카테고리: {doc.metadata.get('category')}")
                    with meta_cols[2]:
                        if doc.metadata.get('score'):
                            st.caption(f"⭐ 점수: {doc.metadata.get('score', 0):.3f}")

                    # 문서 내용 표시
                    st.markdown("**내용:**")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        # SQL 쿼리 표시
        if result.get("sql_query"):
            st.code(result["sql_query"], language="sql")

        # 재작성된 쿼리 표시
        if result.get("rewritten_query"):
            st.info(f"재작성된 쿼리: {result['rewritten_query']}")

        # 오류 표시
        if result.get("error"):
            st.error(f"오류: {result['error']}")


def main():
    """메인 함수"""
    st.set_page_config(
        page_title="IoT 하드웨어·보안 AI 에이전트",
        page_icon="🔌",
        layout="wide"
    )

    st.title("🔌 IoT 디바이스 하드웨어·보안 통합 점검 AI 에이전트")
    st.markdown("---")

    # 사이드바 - 환경 변수 확인
    with st.sidebar:
        st.header("⚙️ 설정 확인")

        required_vars = {
            "OPENAI_API_KEY": "OpenAI API",
            "QDRANT_URL": "Qdrant URL",
            "QDRANT_API_KEY": "Qdrant API Key",
            "SUPABASE_DB_URL": "Supabase DB"
        }

        for var, name in required_vars.items():
            if os.getenv(var):
                st.success(f"✓ {name}")
            else:
                st.error(f"✗ {name}")

        st.markdown("---")
        st.header("📖 사용 방법")
        st.markdown("""
        **일반 질문 (general):**
        - "안녕하세요"
        - "너는 뭘 할 수 있어?"

        **기술문서 검색 (iot_consult):**
        - "JTAG를 왜 막아야 하나요?"
        - "MCU 메모리는 어떻게 보호하나요?"
        - "PCB를 설계할 때 보안상 무엇을 고려해야 하나요?"

        **DB 검색 (database):**
        - "ESP32-H2의 하드웨어 기능을 모두 보여줘"
        - "STM32에서 JTAG와 관련된 기능은 뭐가 있어?"
        - "ESP32-H2의 보안 기능은 몇 개야?"

        **하이브리드 검색 (hybrid):**
        - "스마트 도어락 제작 방법 알려줘"
        - "센서를 MCU에 어떻게 연결하고 해킹도 막으려면?"
        """)

        if st.button("대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # 세션 상태 초기화
    init_session_state()

    # 이전 메시지 표시
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("workflow_info")  # 워크플로 정보가 있으면 전달
        )

    # 사용자 입력
    if prompt := st.chat_input("IoT 하드웨어 설계나 보안에 대해 물어보세요..."):
        # 사용자 메시지 표시 및 저장
        display_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 워크플로 실행
        with st.chat_message("assistant"):
            try:
                # 이전 대화 전체를 전달해야
                # analyze_question이 맥락을 반영해 질문을 재구성할 수 있음
                input_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                # 그래프 스트리밍 실행 (진행 상황 표시)
                result = run_workflow(input_messages, st)

                # 답변 표시 (messages의 마지막 AIMessage에서 추출)
                messages = result.get("messages", [])
                if messages:
                    # 마지막 메시지에서 content 추출
                    last_message = messages[-1]
                    answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
                else:
                    answer = "죄송합니다. 답변을 생성할 수 없습니다."

                st.markdown(answer)

                # 워크플로 정보 표시
                display_workflow_info(result)

                # 어시스턴트 메시지와 워크플로 정보 함께 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "workflow_info": result  # 워크플로 정보 저장
                })

            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


if __name__ == "__main__":
    main()
