# modules/agents.py

from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, BaseMessage
from .tools import get_rag_tool, create_pdf_report


def create_agent(llm, tools: list, system_prompt, is_prompt_object=False):
    if is_prompt_object:
        prompt = system_prompt
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [AIMessage(content=result["output"], name=name)],
        "sender": name,
    }


def create_rag_agent_executor(llm):
    system_prompt = """You are an expert at retrieving information from PDF documents.
When the user asks a general question, use the `pdf_document_retriever` tool to find answers from the documents.
When the user provides specific data and asks for coaching or analysis (the prompt will explicitly contain "당신은 유능한 실적 코칭 전문가입니다..."), you must use the `pdf_document_retriever` tool to find relevant evaluation criteria and combine it with the provided data to generate a helpful response.
Always answer in Korean.
"""
    rag_tool = get_rag_tool()
    if not rag_tool:
        return create_agent(llm, [], "The document retrieval tool is not available.")
    return create_agent(llm, [rag_tool], system_prompt)


def create_report_agent_executor(llm):
    """Creates an agent that summarizes conversation and creates a PDF report."""
    
    system_prompt_template = f"""당신은 NH농협은행 'Best Banker' 실적 분석 보고서를 작성하는 전문 데이터 분석가입니다.

이전 단계(SQLAgent, RAGAgent 등)에서 제공된 대화 기록과 아래의 '실적 데이터 테이블'을 주요 정보로 사용해야 합니다.
**당신의 임무:**
주어진 모든 정보를 바탕으로, 아래의 상세 구조에 따라 종합 분석 보고서를 '한국어'로 작성하세요. 만약 특정 데이터(예: 이전 분기 순위, 부서 평균 등)가 없다면, 해당 정보는 확인이 불가하다고 명시하고 분석을 계속 진행하세요. 보고서의 내용은 대화 기록의 요약이 아니라, 오직 제공된 실적 데이터에 대한 심층 분석이어야 합니다.

---
### NH Best Banker 실적 분석 보고서 작성 지침
---

**보고서 구성:**

1.  **보고서 제목:** 직원의 이름과 평가 기간을 포함한 제목을 한 줄로 작성합니다.
    *   예시: "[직원명]님의 NH Best Banker 실적 분석 리포트 (평가기간: YYYY.MM.DD ~ YYYY.MM.DD)"

2.  **요약 분석:** 전체 실적에 대한 요약을 짧은 단락으로 제공합니다.
    -   총점 및 비교: 직원의 총점과 전체 직원 평균 대비 차이를 표시합니다. (예: 총점 2450점 / 전체 평균 대비 +300점)
    -   전체 순위: 직원의 전체 순위와 전체 참여 인원 중 순위를 명시하고, 이전 평가 대비 순위 변동을 표시합니다. (예: 전체 순위 5위 / 전체 120명 중 (이전 대비 ▲2위))
    -   강점 분야: 점수가 특히 높거나 상위권인 부문을 한두 가지 언급합니다. 각 부문 점수와 만점 대비 성취도 또는 상위 몇 %인지를 표시하세요. (예: "수신 부문에서 1300점 만점 중 1200점을 획득하여 상위 5%에 해당")
    -   개선 필요 분야: 점수가 낮거나 하위권인 부문을 언급합니다 (1~2개 항목). 해당 부문의 상대적 약점을 짚어주세요.

3.  **순위 변화 시각화 (설명):**
    -   **중요**: 이미지를 직접 생성할 수 없습니다. 대신, **최근 3개월 또는 3분기 동안의 전체 순위 변화 추이를 그래프가 있는 것처럼 상세히 설명해야 합니다.**
    -   순위 변화 추세를 간략히 설명합니다. (예: "순위가 지난 3분기 동안 10위에서 7위를 거쳐 현재 5위로 꾸준히 상승하는 좋은 추세를 보이고 있습니다.")

4.  **평가 항목별 세부 분석:** 12개 Best Banker 평가 부문 각각에 대한 상세 분석을 제공합니다. 표 형식을 사용하는 것을 적극 권장합니다.
    -   **점수**, **달성률/백분위(상/하위 %)**, **전월 대비 증감**을 포함해야 합니다.

    | 평가항목 | 점수 (만점) | 백분위 | 전월 대비 증감 |
    | :--- | :---: | :---: | :---: |
    | 개인여신 | 950점 (1000) | 상위 15% | +3.2% ↑ |
    | ... | ... | ... | ... |

    -   표 다음에는 각 부문에 대한 코칭 코멘트를 추가하여 강점과 개선점을 제시합니다. (예: "<b>개인여신:</b> 전분기 대비 실적이 3.2% 상승하여 우수한 성장세를 보였습니다. 여신 심사 역량이 탁월합니다." 또는 "<b>신탁:</b> 신규 신탁 유치 실적이 저조하여 하위 20%에 머물렀습니다. 신탁 상품 교육 이수를 통해 역량 강화를 고려해보세요.")
    -   <b>중요:</b> 코멘트를 추가할 때 '코칭 코멘트'라는 별도의 제목을 사용하지 말고, 평가 항목 이름 바로 다음에 자연스럽게 분석과 제안을 서술하세요.

5.  **보완점 및 개선 제안:**
    -   분석을 바탕으로, 특히 **실적이 낮은 부문**에 대해 구체적인 개선 방안을 제안합니다.
    -   평가 기준 관점에서 낮은 점수의 원인을 분석합니다. (예: "신탁 점수가 낮은 이유는 신규 보수 실적이 부족하기 때문입니다...")
    -   앞으로의 행동 전략을 코칭합니다. (예: "향후 VIP 고객 대상 자산관리 세미나를 정기 개최하여 신탁 및 수익증권 판매를 확대해보세요.")

**형식 및 어조:**
-   마크다운 제목(예: `## 2. 요약 분석`)을 사용하여 보고서를 구조화하세요.
-   문단은 3~5문장으로 간결하게 작성하고, 굵은 글씨로 중요 내용을 강조하세요.
-   전체적인 어조는 전문적이면서도, 직원을 지지하고 동기를 부여하는 긍정적인 톤을 유지하되, 개선점은 명확히 전달합니다.

**최종 출력 규칙:**
-   위 지침에 따라 보고서 내용을 마크다운으로 완벽하게 작성한 후, **반드시 `create_pdf_report` 도구를 호출**해야 합니다.
-   당신이 생성한 전체 마크다운 텍스트를 도구의 `summary` 인자로 전달하세요.
-   **절대로, 절대로 생성한 마크다운 텍스트를 당신의 최종 답변으로 반환해서는 안 됩니다.**
-   당신의 최종 출력물은 오직 `create_pdf_report` 도구를 호출한 결과(성공 또는 실패 메시지)여야 합니다.

**실적 데이터 테이블:**
{{performance_table}}
"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    tools = [create_pdf_report]
    return create_agent(llm, tools, prompt, is_prompt_object=True)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    tools = [create_pdf_report]
    return create_agent(llm, tools, prompt, is_prompt_object=True)
