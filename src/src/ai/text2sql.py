import os
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage


class Text2SQLEngine:
    def __init__(self):
        """Text2SQL 엔진 초기화"""
        # Supabase 데이터베이스 연결
        self.db = SQLDatabase.from_uri(os.getenv("SUPABASE_DB_URL"))

        # LLM 초기화
        self.llm = init_chat_model("gpt-5.4-mini")

        # 데이터베이스 스키마 정보 캐싱
        self.schema_info = self.db.get_table_info()

    def generate_sql(self, question: str, feedback: str = None) -> str:
        """
        자연어 질문을 SQL 쿼리로 변환

        Args:
            question: 사용자의 자연어 질문
            feedback: 이전 시도의 오류 피드백 (재시도 시)

        Returns:
            생성된 SQL 쿼리
        """
        system_prompt = f"""
당신은 IoT 디바이스 보안 점검 도우미의 PostgreSQL 전문가입니다.
일반 사용자의 질문을 정확한 SQL 쿼리로 변환하세요.

<database_schema>
{self.schema_info}
</database_schema>

<table_descriptions>

1. "iot_security_agencies_parent" (부모 테이블, 3행)

보안 사고 신고 기관 정보입니다.

주요 컬럼:
- agency_id: 기관 고유 ID (PK)
- agency_name: 기관명 (KISA 118 상담센터, KISA 보호나라·KrCERT/CC, 경찰청 ECRM)
- organization: 소속 기관
- phone: 전화번호 (118, 182)
- contact_method: 연락 방법
- main_role: 주요 역할
- source_url: 홈페이지 URL


2. "iot_security_incidents_child" (자식 테이블, 12행)

보안 사고 유형별 증상과 대처 방법입니다.

주요 컬럼:
- incident_id: 사고 유형 고유 ID (PK)
- agency_id: 신고 기관 ID (FK → iot_security_agencies_parent.agency_id)
- incident_type: 사고 유형 (해킹 의심, 악성코드, 계정 탈취, 개인정보 침해, 랜섬웨어, DDoS 등)
- example_symptom: 증상 예시
- recommended_action: 권장 대처 방법


3. "1_parent_iot_common_guidelines" (부모 테이블, 15행)

IoT 공통보안가이드의 15개 가이드라인 항목입니다.

주요 컬럼:
- guideline_id: 가이드라인 고유 ID (PK)
- principle_name: 상위 원칙 이름
- lifecycle_stage: 수명주기 단계
- guideline_name: 가이드라인 이름
- domain: 분야
- easy_explanation: 쉬운 설명


4. "2_child_home_iot_controls" (자식 테이블, 18행)

홈·가전 IoT 보안가이드의 보안 점검 항목입니다.

주요 컬럼:
- home_control_id: 항목 고유 ID (PK)
- parent_guideline_id: 부모 가이드라인 ID (FK → 1_parent_iot_common_guidelines.guideline_id)
- control_name: 보안 항목 이름
- control_category: 카테고리 (물리적 보안, 인증, 암호화, 데이터 보호, 플랫폼 보안 등)
- easy_explanation: 쉬운 설명
- security_purpose: 보안 목적


5. "3_child_stm32_debug_topics" (자식 테이블, 14행)

STM32 MCU 디버깅 가이드의 주요 항목입니다.
- parent_guideline_id: FK → 1_parent_iot_common_guidelines.guideline_id


6. "4_child_esp32h2_features" (자식 테이블, 26행)

ESP32-H2 기술 참조 매뉴얼의 주요 기능입니다.
- parent_guideline_id: FK → 1_parent_iot_common_guidelines.guideline_id

</table_descriptions>

<relationships>
- iot_security_incidents_child.agency_id = iot_security_agencies_parent.agency_id
- 2_child_home_iot_controls.parent_guideline_id = 1_parent_iot_common_guidelines.guideline_id
- 3_child_stm32_debug_topics.parent_guideline_id = 1_parent_iot_common_guidelines.guideline_id
- 4_child_esp32h2_features.parent_guideline_id = 1_parent_iot_common_guidelines.guideline_id
</relationships>

<priority>
- 신고 기관, 사고 유형, 대처 방법 관련 질문 → iot_security_agencies_parent + iot_security_incidents_child 우선
- 홈가전 보안 점검 관련 질문 → 2_child_home_iot_controls 우선
- 보안 가이드라인 관련 질문 → 1_parent_iot_common_guidelines 우선
- 쉬운 설명이 필요하면 easy_explanation 컬럼 활용
</priority>

<rules>
- PostgreSQL 문법을 사용하세요
- SELECT 쿼리만 생성하세요 (INSERT, UPDATE, DELETE 금지)
- 결과는 최대 10개로 제한하세요 (LIMIT 10)
- SQL 쿼리만 반환하고, 설명은 포함하지 마세요
- 코드 블록(```)이나 'sql' 키워드 없이 순수 쿼리만 반환하세요
- NULL 값을 주의해서 처리하세요
- 존재하지 않는 컬럼을 사용하지 마세요
- 한글 이름으로 검색할 때는 LIKE '%키워드%' 를 사용하세요
- JOIN이 필요한 경우 적절히 사용하세요
- 테이블 이름에 숫자와 언더스코어가 포함되므로 반드시 큰따옴표로 감싸세요
  예: SELECT * FROM "1_parent_iot_common_guidelines"
- 세미콜론(;)으로 쿼리를 종료하세요
</rules>
"""

        if feedback:
            system_prompt += f"\n\n이전 시도의 오류:\n{feedback}\n\n위 오류를 고려하여 쿼리를 수정하세요."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        response = self.llm.invoke(messages)
        sql_query = response.content.strip()

        # 코드 블록 제거
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
            sql_query = sql_query.replace("sql", "").strip()

        return sql_query

    def execute_sql(self, sql_query: str) -> tuple[str, str]:
        """
        SQL 쿼리 실행

        Args:
            sql_query: 실행할 SQL 쿼리

        Returns:
            (결과 문자열, 오류 메시지) 튜플
        """
        try:
            result = self.db.run(sql_query)
            return result, None
        except Exception as e:
            error_msg = str(e)
            return None, error_msg

    def query(self, question: str, previous_error: str = None) -> dict:
        """
        질문에 대한 SQL 생성 및 실행

        Args:
            question: 사용자 질문
            previous_error: 이전 시도의 오류 (재시도 시)

        Returns:
            결과 딕셔너리 (sql_query, result, error)
        """
        # SQL 생성
        sql_query = self.generate_sql(question, feedback=previous_error)

        # SQL 실행
        result, error = self.execute_sql(sql_query)

        return {
            "sql_query": sql_query,
            "result": result,
            "error": error
        }

    def is_empty_result(self, result: str) -> bool:
        """
        결과가 비어있는지 확인

        Args:
            result: SQL 실행 결과

        Returns:
            결과가 비어있으면 True
        """
        if not result:
            return True

        # 빈 결과 패턴 확인
        empty_patterns = ["[]", "()", "no rows", "0 rows"]
        result_lower = result.lower().strip()

        return any(pattern in result_lower for pattern in empty_patterns)


def get_text2sql_engine() -> Text2SQLEngine:
    """Text2SQL 엔진 인스턴스 반환"""
    return Text2SQLEngine()