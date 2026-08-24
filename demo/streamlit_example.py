import os
import sys
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import streamlit as st
from dotenv import load_dotenv

# AI 모듈이 Qdrant 설정을 읽기 전에 .env를 먼저 로드합니다.
load_dotenv()

from ai import create_graph


st.set_page_config(
    page_title="IoT 디바이스 보안 점검 도우미",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_graph():
    """LangGraph 워크플로를 앱 세션 간 재사용합니다."""
    return create_graph()


NODE_LABELS = {
    "analyze_question": "질문 분석",
    "classify_intent": "의도 분류",
    "general_answer": "일반 답변 생성",
    "vector_search": "기술문서 검색",
    "grade_documents": "검색 문서 평가",
    "rewrite_query": "검색어 보완",
    "database_query": "보안 정보 조회",
    "validate_db_result": "조회 결과 검증",
    "combine_context": "검색 결과 통합",
    "generate_answer": "최종 답변 생성",
    "validate_answer": "답변 검증",
}

INTENT_LABELS = {
    "general": "일반 안내",
    "iot_consult": "보안 문서 검색",
    "database": "보안 정보 조회",
    "hybrid": "문서 + 데이터 통합 검색",
}

CASE_QUESTIONS = [
    ("Mirai 봇넷", ":material/hub:", "Mirai 봇넷은 IoT 기기를 어떻게 감염시키고, 일반 사용자는 어떻게 예방할 수 있나요?"),
    ("스마트 CCTV 해킹", ":material/videocam:", "스마트 CCTV나 홈캠이 해킹된 것 같을 때 어떤 징후를 확인하고 어떻게 대처해야 하나요?"),
    ("스마트 도어락 취약점", ":material/lock:", "스마트 도어락의 주요 보안 취약점과 사용자가 확인할 점을 알려주세요."),
    ("의료기기 해킹", ":material/monitor_heart:", "인터넷에 연결된 의료기기에서 중요하게 확인해야 할 보안 위험은 무엇인가요?"),
    ("차량 IoT 해킹", ":material/directions_car:", "인터넷에 연결된 차량에서 발생할 수 있는 해킹 위험과 기본 대처 방법을 알려주세요."),
]

RECOMMENDED_QUESTIONS = [
    ("웹캠이 혼자 움직이는데 해킹인가요?", ":material/videocam:"),
    ("공유기에 모르는 기기가 연결됐어요", ":material/router:"),
    ("스마트 도어락 비밀번호는 어떻게 관리하나요?", ":material/door_front:"),
    ("스마트 기기 해킹이 의심되면 먼저 뭘 해야 하나요?", ":material/security:"),
    ("스마트 기기 해킹은 어디에 신고하나요?", ":material/contact_support:"),
]


def init_session_state():
    """화면 재실행 후에도 대화와 UI 상태가 유지되도록 초기화합니다."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("settings_open", False)
    st.session_state.setdefault("show_workflow", False)


def queue_prompt(question: str):
    """사례나 추천 질문을 클릭하면 다음 실행에서 질문으로 처리합니다."""
    st.session_state.pending_prompt = question


def reset_conversation():
    """현재 탭의 대화 기록만 초기화합니다."""
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def toggle_settings():
    """사이드바의 설정 영역을 열고 닫습니다."""
    st.session_state.settings_open = not st.session_state.settings_open


def inject_styles():
    """참고 시안과 동일한 흰색·회색 기반의 미니멀 UI를 적용합니다."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #17191c;
            --muted: #5e636b;
            --line: #dfe2e6;
        }
        html, body, [class*="css"] {
            font-family: Pretendard, "Noto Sans KR", "Apple SD Gothic Neo", sans-serif;
        }
        *, *::before, *::after { box-sizing: border-box; }
        .stApp { background: #fff; color: var(--ink); }
        #MainMenu, footer,
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"] { display: none !important; }
        [data-testid="stToolbar"] {
            display: flex;
            background: transparent;
        }
        header[data-testid="stHeader"] {
            display: block;
            height: 48px;
            background: transparent;
        }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            z-index: 999999;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1273px;
            padding: 28px 24px 20px;
            position: relative;
            left: 3.5px;
        }
        [data-testid="stSidebar"] {
            width: 340px !important;
            min-width: 340px !important;
            background: #fbfbfb;
            border-right: 1px solid #e2e4e7;
        }
        [data-testid="stSidebarContent"] {
            width: 340px !important;
        }
        [data-testid="stSidebarUserContent"] {
            transform: translateY(-38px);
            padding-inline: 4px !important;
        }
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
            gap: 6px !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p { color: var(--ink); }
        .sidebar-brand {
            font-size: 22px;
            line-height: 1.3;
            font-weight: 700;
            letter-spacing: -.03em;
            margin: 0 0 75px;
        }
        .sidebar-section-title {
            margin: 0 0 24px;
            font-size: 17px;
            line-height: 1.4;
            font-weight: 700;
            letter-spacing: -.025em;
        }
        .sidebar-section-gap {
            margin-top: 46px;
        }
        .st-key-settings_button {
            margin-top: 46px;
        }
        [class*="st-key-case_"] button,
        .st-key-settings_button button {
            min-height: 36px;
            width: 100%;
            padding: 5px 2px;
            justify-content: flex-start !important;
            gap: 14px;
            color: #2e3136;
            background: transparent;
            border: 0;
            border-radius: 7px;
            box-shadow: none;
            font-size: 15px;
            font-weight: 450;
            text-align: left;
        }
        [class*="st-key-case_"] button > div,
        .st-key-settings_button button > div {
            width: auto;
            flex: 1;
            justify-content: flex-start;
        }
        [class*="st-key-case_"] button > div > span,
        .st-key-settings_button button > div > span {
            width: auto;
            justify-content: flex-start;
            gap: 14px;
        }
        [class*="st-key-case_"] button:hover,
        .st-key-settings_button button:hover {
            color: #111315;
            background: #f0f1f2;
            border: 0;
        }
        [class*="st-key-case_"] button:focus,
        .st-key-settings_button button:focus {
            background: transparent;
            box-shadow: none;
        }
        [class*="st-key-case_"] [data-testid="stIconMaterial"] {
            width: 28px;
            min-width: 28px;
            font-size: 25px !important;
            font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 40;
        }
        .st-key-settings_button [data-testid="stIconMaterial"] {
            font-size: 21px !important;
            font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
        }
        .status-list {
            display: grid;
            gap: 12px;
            margin-top: 16px;
            font-size: 14px;
            color: #33363b;
        }
        .status-row {
            display: grid;
            grid-template-columns: 110px 18px 1fr;
            align-items: center;
            min-height: 20px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border: 1px solid #9da2a8;
            border-radius: 50%;
            position: relative;
        }
        .status-dot.connected::after {
            content: "";
            position: absolute;
            inset: 2px;
            border-radius: 50%;
            background: #8e949b;
        }
        .status-dot.disconnected { border-color: #c8cbd0; }
        .status-row.doc-count { grid-template-columns: 128px 1fr; }
        .st-key-reset_button button {
            margin-top: 10px;
            color: #555a61;
            background: transparent;
            border-color: #e1e3e6;
        }
        .hero-card {
            height: 220px;
            min-height: 220px;
            padding: 31px 34px;
            border: 1px solid #dfe2e6;
            border-radius: 9px;
            background: #fff;
            display: flex;
            align-items: center;
            justify-content: space-between;
            overflow: hidden;
        }
        .hero-copy { position: relative; z-index: 2; max-width: 830px; }
        .hero-title {
            margin: 0 0 16px;
            color: #14171a;
            font-size: 34px;
            line-height: 1.25;
            font-weight: 760;
            letter-spacing: -.035em;
            white-space: nowrap;
        }
        .hero-description {
            margin: 0;
            color: #3f4349;
            font-size: 16px;
            line-height: 1.65;
            letter-spacing: -.015em;
        }
        .hero-description + .hero-description {
            margin-top: 8px;
        }
        .hero-art {
            width: 330px;
            min-width: 280px;
            height: 145px;
            margin-right: -2px;
            opacity: .9;
        }
        .recommend-title {
            margin: 36px 0 17px;
            color: #1c1e21;
            font-size: 17px;
            line-height: 1.4;
            font-weight: 700;
            letter-spacing: -.025em;
        }
        [class*="st-key-suggest_"] button {
            width: 100%;
            min-height: 90px;
            padding: 14px 22px;
            justify-content: flex-start;
            gap: 16px;
            color: #30343a;
            background: #fff;
            border: 1px solid #dfe2e6;
            border-radius: 8px;
            box-shadow: none;
            text-align: left;
            font-size: 15px;
            font-weight: 450;
            line-height: 1.45;
        }
        [class*="st-key-suggest_"] button:hover {
            color: #111315;
            background: #fafafa;
            border-color: #bfc4ca;
            transform: translateY(-1px);
        }
        [class*="st-key-suggest_"] button p {
            white-space: normal;
            text-align: left;
        }
        [class*="st-key-suggest_"] button > div {
            width: 100%;
        }
        [class*="st-key-suggest_"] button > div > span {
            width: 100%;
            justify-content: flex-start;
            gap: 14px;
        }
        [class*="st-key-suggest_"] [data-testid="stIconMaterial"] {
            width: 36px;
            min-width: 36px;
            height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 1 !important;
            font-size: 34px !important;
            font-variation-settings: 'FILL' 0, 'wght' 200, 'GRAD' 0, 'opsz' 40;
        }
        .home-illustration {
            position: fixed;
            top: 245px;
            right: 0;
            bottom: 84px;
            left: 340px;
            z-index: 0;
            width: auto;
            height: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            pointer-events: none;
        }
        .home-illustration svg {
            width: min(980px, 82%);
            height: min(560px, 72vh);
            opacity: .72;
            transform: translateX(-12px);
        }
        [data-testid="stChatMessage"] {
            position: relative;
            z-index: 1;
            border: 1px solid #e5e7ea;
            border-radius: 10px;
            background: rgba(255, 255, 255, .82);
            padding: 8px 10px;
        }
        /* 재질문 중 새 답변 영역에 남는 이전 답변의 stale 복제본을 숨깁니다. */
        [data-testid="stChatMessage"]
        [data-testid="stElementContainer"][data-stale="true"] {
            display: none !important;
        }
        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
        ) {
            width: fit-content;
            max-width: 78%;
            margin-left: auto;
            background: rgba(245, 246, 247, .94);
            flex-direction: row-reverse;
        }
        @media (max-width: 768px) {
            .home-illustration {
                top: 220px;
                left: 0;
            }
            .home-illustration svg {
                width: 96%;
                opacity: .58;
            }
            [data-testid="stChatMessage"]:has(
                [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
            ) {
                max-width: 92%;
            }
        }
        [data-testid="stChatInput"] {
            min-height: 67px;
            border: 1px solid #dfe2e6;
            border-radius: 8px;
            background: #fff;
            box-shadow: none;
            width: calc(100% - 72px);
            margin: 0 auto;
        }
        @media (max-width: 768px) {
            [data-testid="stChatInput"] {
                width: calc(100% - 16px);
            }
        }
        [data-testid="stChatInput"] textarea {
            height: 42px !important;
            min-height: 42px !important;
            padding: 10px 18px !important;
            font-size: 14px !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            width: 68px !important;
            height: 42px !important;
            margin-right: 10px;
            color: #24272b !important;
            background: #fff !important;
            border: 1px solid #e1e4e8 !important;
            border-radius: 8px !important;
        }
        [data-testid="stChatInputSubmitButton"] span,
        [data-testid="stChatInputSubmitButton"] svg {
            display: none !important;
        }
        [data-testid="stChatInputSubmitButton"]::after {
            content: "전송";
            color: #24272b;
            font-size: 14px;
            font-weight: 450;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #aeb4bb;
            box-shadow: 0 0 0 1px #aeb4bb;
        }
        [data-testid="stBottom"] {
            transform: translateY(-17px);
        }
        [data-testid="stBottom"] > div { background: rgba(255,255,255,.98); }
        [data-testid="stStatusWidget"] { display: none !important; }
        @media (max-width: 1100px) {
            [data-testid="stMainBlockContainer"] { padding-inline: 18px; }
            .hero-art { width: 250px; }
            .hero-title { font-size: 29px; }
        }
        @media (max-width: 760px) {
            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {
                width: 290px !important;
                min-width: 290px !important;
            }
            [data-testid="stSidebarUserContent"] {
                transform: translateY(-20px);
                padding-inline: 4px !important;
            }
            [data-testid="stMainBlockContainer"] { padding-top: 20px; }
            .hero-card { height: auto; min-height: 210px; padding: 24px; }
            .hero-art { display: none; }
            .hero-title { font-size: 25px; white-space: normal; }
            .home-illustration { height: 260px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def connection_status(label: str, env_name: str) -> str:
    """환경변수 존재 여부를 시안의 연결 상태 행으로 표시합니다."""
    connected = bool(os.getenv(env_name))
    dot_class = "connected" if connected else "disconnected"
    state_text = "연결됨" if connected else "미연결"
    return (
        f'<div class="status-row"><span>{label}</span>'
        f'<span class="status-dot {dot_class}"></span>'
        f'<span>{state_text}</span></div>'
    )


def render_sidebar():
    """해킹 사례, 시스템 상태, 설정을 사이드바에 표시합니다."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">IoT 보안 점검 도우미</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">해킹 사례 바로가기</div>', unsafe_allow_html=True)

        for index, (label, icon, question) in enumerate(CASE_QUESTIONS):
            st.button(
                label,
                icon=icon,
                key=f"case_{index}",
                width="stretch",
                on_click=queue_prompt,
                args=(question,),
            )

        st.markdown(
            '<div class="sidebar-section-title sidebar-section-gap">시스템 상태</div>',
            unsafe_allow_html=True,
        )
        status_html = (
            '<div class="status-list">'
            + connection_status("OpenAI API", "OPENAI_API_KEY")
            + connection_status("Qdrant DB", "QDRANT_URL")
            + connection_status("Supabase DB", "SUPABASE_DB_URL")
            + '<div class="status-row doc-count"><span>문서 코드</span>'
            + '<span>3개</span></div></div>'
        )
        st.markdown(status_html, unsafe_allow_html=True)

        st.button(
            "설정",
            icon=":material/settings:",
            key="settings_button",
            width="stretch",
            on_click=toggle_settings,
        )
        if st.session_state.settings_open:
            st.toggle("분석 과정 표시", key="show_workflow")
            st.button(
                "대화 초기화",
                icon=":material/refresh:",
                key="reset_button",
                width="stretch",
                on_click=reset_conversation,
            )


def render_hero():
    """상단 서비스 소개 카드와 보안 방패 회로 이미지를 표시합니다."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-copy">
                <h1 class="hero-title">IoT 디바이스 보안 점검 도우미</h1>
                <p class="hero-description">
                    웹캠·홈캠, 스마트 도어락, 공유기, 스마트TV에서 이상한 움직임이나 모르는 접속이 발견되었나요?
                </p>
                <p class="hero-description">
                    해킹 의심 증상 확인부터 계정·비밀번호 보호, 즉시 대처 방법까지 쉽게 안내합니다.<br>
                    전문기관 신고·상담 절차도 누구나 이해하고 따라 할 수 있도록 설명해드립니다.
                </p>
            </div>
            <svg class="hero-art" viewBox="0 0 330 150" fill="none" aria-hidden="true">
                <g stroke="#d9dde2" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M170 12c18 15 34 22 58 27v35c0 35-20 52-58 70-38-18-58-35-58-70V39c24-5 40-12 58-27Z"/>
                    <path d="M170 25c15 12 28 17 46 21v28c0 27-15 41-46 57-31-16-46-30-46-57V46c18-4 31-9 46-21Z"/>
                    <circle cx="170" cy="70" r="13"/><path d="M166 82h8l3 23h-14l3-23Z"/>
                    <path d="M112 52H85l-14-14H50"/><circle cx="47" cy="38" r="3"/>
                    <path d="M111 66H72l-10-10H28"/><circle cx="25" cy="56" r="3"/>
                    <path d="M111 80H61l-8 8H18"/><circle cx="15" cy="88" r="3"/>
                    <path d="M111 95H74l-13 13H37"/><circle cx="34" cy="108" r="3"/>
                    <path d="M228 52h27l14-14h20"/><circle cx="292" cy="38" r="3"/>
                    <path d="M229 66h39l10-10h34"/><circle cx="315" cy="56" r="3"/>
                    <path d="M229 80h50l8 8h25"/><circle cx="315" cy="88" r="3"/>
                    <path d="M229 95h37l13 13h24"/><circle cx="306" cy="108" r="3"/>
                </g>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommended_questions():
    """첫 화면에서 클릭 가능한 다섯 개 추천 질문을 표시합니다."""
    st.markdown('<div class="recommend-title">추천 질문</div>', unsafe_allow_html=True)
    columns = st.columns(5, gap="small")
    for index, ((question, icon), column) in enumerate(zip(RECOMMENDED_QUESTIONS, columns)):
        with column:
            st.button(
                question,
                icon=icon,
                key=f"suggest_{index}",
                width="stretch",
                on_click=queue_prompt,
                args=(question,),
            )


def render_home_illustration():
    """대화와 스크롤 뒤에 유지되는 스마트홈 보안 배경을 표시합니다."""
    st.markdown(
        """
        <div class="home-illustration">
            <svg viewBox="0 0 820 320" fill="none" aria-hidden="true">
                <g stroke="#dde1e5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M301 150 410 53l109 97h-26v125H327V150h-26Z"/>
                    <path d="M410 122c21 17 39 23 62 28v44c0 36-21 56-62 75-41-19-62-39-62-75v-44c23-5 41-11 62-28Z"/>
                    <path d="M410 140c16 12 29 17 45 20v34c0 27-15 42-45 57-30-15-45-30-45-57v-34c16-3 29-8 45-20Z"/>
                    <path d="m390 196 14 15 29-35"/>
                    <path d="M136 105h89l26 23h41" stroke-dasharray="6 7"/><circle cx="294" cy="128" r="5"/>
                    <path d="M526 102h48v-18" stroke-dasharray="6 7"/>
                    <path d="M522 220h55l17 17h39" stroke-dasharray="6 7"/><circle cx="519" cy="220" r="5"/>
                    <path d="M299 220h-99l-17 17h-39" stroke-dasharray="6 7"/><circle cx="302" cy="220" r="5"/>
                    <rect x="151" y="61" width="76" height="52" rx="18"/>
                    <circle cx="189" cy="87" r="15"/><circle cx="189" cy="87" r="6"/>
                    <path d="M176 113v10h26v-10M166 123h46"/>
                    <path d="M579 84h73c7 0 13 6 13 13v20c0 7-6 13-13 13h-73c-7 0-13-6-13-13V97c0-7 6-13 13-13Z"/>
                    <path d="M585 84V41M641 84V33"/><circle cx="585" cy="137" r="2"/><circle cx="646" cy="137" r="2"/>
                    <path d="M603 70c8-8 20-8 28 0M609 76c5-5 12-5 17 0"/>
                    <g transform="translate(-44 0)">
                        <rect x="167" y="189" width="65" height="104" rx="13"/>
                        <rect x="176" y="199" width="47" height="57" rx="8"/>
                        <circle cx="199.5" cy="271" r="13"/><circle cx="199.5" cy="271" r="5"/>
                        <circle cx="184" cy="211" r="2"/><circle cx="199" cy="211" r="2"/><circle cx="214" cy="211" r="2"/>
                        <circle cx="184" cy="224" r="2"/><circle cx="199" cy="224" r="2"/><circle cx="214" cy="224" r="2"/>
                        <circle cx="184" cy="237" r="2"/><circle cx="199" cy="237" r="2"/><circle cx="214" cy="237" r="2"/>
                    </g>
                    <rect x="629" y="216" width="91" height="77" rx="10"/>
                    <rect x="640" y="228" width="69" height="48" rx="6"/>
                    <path d="M642 283h65"/><text x="655" y="261" fill="#dde1e5" stroke="none" font-size="28">25°</text>
                    <path d="M311 284h197"/>
                </g>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def node_progress_detail(node_name: str, update: dict) -> str:
    """노드 실행 결과에서 진행 상황에 보여줄 짧은 정보를 추출합니다."""
    if not isinstance(update, dict):
        return ""
    if node_name == "classify_intent":
        intent = update.get("intent")
        return INTENT_LABELS.get(intent, intent or "")
    if node_name == "vector_search":
        return f"{len(update.get('vector_results') or [])}건 검색됨"
    if node_name == "grade_documents":
        return "충분함" if update.get("document_grade") == "relevant" else "자료 보완 중"
    if node_name == "rewrite_query":
        return update.get("rewritten_query", "")
    if node_name == "database_query":
        return "조회 오류" if update.get("error") else "조회 완료"
    if node_name == "validate_db_result":
        return "결과 확인" if update.get("db_valid") == "valid" else "다시 조회 중"
    if node_name == "validate_answer":
        return "검증 완료" if update.get("answer_valid") == "ok" else "답변 보완 중"
    return ""


def run_workflow(input_messages: list, container) -> dict:
    """LangGraph를 스트리밍 실행하고 최종 상태를 반환합니다."""
    final_state = {}
    node_counts = {}
    with container.status("질문을 확인하고 있어요...", expanded=False) as status:
        for mode, chunk in get_graph().stream(
            {"messages": input_messages},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node_name, update in chunk.items():
                    label = NODE_LABELS.get(node_name, node_name)
                    detail = node_progress_detail(node_name, update)
                    node_counts[node_name] = node_counts.get(node_name, 0) + 1
                    status.update(label=f"{label}{f' · {detail}' if detail else ''}")
            else:
                final_state = chunk
        status.update(label="답변 준비 완료", state="complete", expanded=False)
    final_state["node_counts"] = node_counts
    return final_state


def display_workflow_info(result: dict):
    """설정에서 요청한 경우에만 기술적인 처리 정보를 표시합니다."""
    with st.expander("분석 과정", icon=":material/search_insights:"):
        intent = result.get("intent")
        col1, col2, col3 = st.columns(3)
        col1.metric("질문 유형", INTENT_LABELS.get(intent, intent or "확인 중"))
        col2.metric("검색 문서", len(result.get("vector_results") or []))
        col3.metric("DB 조회", "수행" if result.get("db_results") else "미수행")

        analyzed = result.get("analyzed_question")
        if analyzed and analyzed != result.get("question"):
            st.caption(f"대화 맥락을 반영한 질문: {analyzed}")
        if result.get("vector_results"):
            st.markdown("**참고 문서**")
            for index, doc in enumerate(result["vector_results"], 1):
                source = doc.metadata.get("source", "출처 미상")
                page = doc.metadata.get("page", "-")
                st.caption(f"{index}. {source} · {page}쪽")
        if result.get("sql_query"):
            st.code(result["sql_query"], language="sql")
        if result.get("error"):
            st.error(result["error"])


def display_message(role: str, content: str, workflow_info: dict | None = None):
    """사용자와 도우미의 대화 메시지를 표시합니다."""
    avatar = ":material/person:" if role == "user" else ":material/shield:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if role == "assistant" and workflow_info and st.session_state.show_workflow:
            display_workflow_info(workflow_info)


def main():
    init_session_state()
    inject_styles()
    render_sidebar()
    render_hero()

    # 추천 질문을 누른 직후에는 답변 생성 전이라도 추천 영역을 숨깁니다.
    if not st.session_state.messages and not st.session_state.pending_prompt:
        render_recommended_questions()

    # 스마트홈 그림은 대화 상태와 관계없이 항상 유지되는 배경 영역입니다.
    render_home_illustration()

    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("workflow_info"),
        )

    typed_prompt = st.chat_input(
        "웹캠·홈캠, 스마트 도어락, 공유기의 이상 증상이나 보안이 궁금하다면 물어보세요...",
        submit_mode="disable",
    )
    prompt = st.session_state.pop("pending_prompt", None) or typed_prompt
    if not prompt:
        return

    display_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=":material/shield:"):
        try:
            input_messages = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages
            ]
            result = run_workflow(input_messages, st)
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                answer = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
            else:
                answer = "죄송합니다. 현재 답변을 만들지 못했습니다. 잠시 후 다시 질문해주세요."

            st.markdown(answer)
            if st.session_state.show_workflow:
                display_workflow_info(result)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "workflow_info": result,
                }
            )
        except Exception as error:
            error_message = "답변을 불러오는 중 문제가 생겼습니다. 연결 상태를 확인한 뒤 다시 시도해주세요."
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
            if st.session_state.show_workflow:
                st.exception(error)


if __name__ == "__main__":
    main()
