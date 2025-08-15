import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from .pdf_utils import df_to_pdf


def get_current_tool_message(tool_args, tool_call_id):
    """Get the tool message corresponding to the given tool call ID."""
    if tool_call_id:
        for tool_arg in tool_args:
            if tool_arg["tool_call_id"] == tool_call_id:
                return tool_arg
        return None
    else:
        return None


def format_search_result(results):
    """Format search results into a markdown string."""
    results = json.loads(results)
    answer = ""
    for result in results:
        answer += f'**[{result["title"]}]({result["url"]})**\n\n'
        answer += f'{result["content"]}\n\n'
        answer += f'신뢰도: {result["score"]}\n\n'
        answer += "\n-----\n"
    return answer


def stream_handler(streamlit_container, agent_executor, inputs, config):
    """Handle streaming of agent execution results in a Streamlit container."""
    tool_args = []
    agent_answer = ""
    agent_message = None

    container = streamlit_container.container()
    with container:
        for chunk_msg, metadata in agent_executor.stream(
            inputs, config, stream_mode="messages"
        ):
            if hasattr(chunk_msg, "tool_calls") and chunk_msg.tool_calls:
                tool_arg = {
                    "tool_name": "",
                    "tool_result": "",
                    "tool_call_id": chunk_msg.tool_calls[0]["id"],
                }
                tool_arg["tool_name"] = chunk_msg.tool_calls[0]["name"]
                if tool_arg["tool_name"]:
                    tool_args.append(tool_arg)

            if metadata["langgraph_node"] == "tools":
                current_tool_message = get_current_tool_message(
                    tool_args, chunk_msg.tool_call_id
                )
                if current_tool_message:
                    current_tool_message["tool_result"] = chunk_msg.content
                    tool_name = current_tool_message["tool_name"]
                    tool_result = current_tool_message["tool_result"]
                    with st.status(f"✅ {tool_name}"):
                        if tool_name == "web_search":
                            st.markdown(format_search_result(tool_result))
                        elif tool_name == "sql_query":
                            df = pd.read_json(tool_result)
                            st.dataframe(df)
                            chart_fig = None
                            numeric_df = df.select_dtypes(include="number")
                            if not numeric_df.empty:
                                chart_fig, ax = plt.subplots()
                                numeric_df.plot(kind="bar", ax=ax)
                                st.pyplot(chart_fig)
                            pdf_bytes = df_to_pdf(df, chart_fig)
                            st.download_button(
                                label="PDF 다운로드",
                                data=pdf_bytes,
                                file_name="result.pdf",
                                mime="application/pdf",
                            )
                        elif tool_name.startswith("sql_db_"):
                            st.markdown(tool_result)
                        elif tool_name == "bank_manual_rag":
                            st.markdown(tool_result)

            if metadata["langgraph_node"] == "agent":
                if chunk_msg.content:
                    if agent_message is None:
                        agent_message = st.empty()
                    agent_answer += chunk_msg.content
                    agent_message.markdown(agent_answer)

        return container, tool_args, agent_answer
