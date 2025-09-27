# modules/tools.py

import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from .config import DB_PATH
from typing import List
from .utils import load_db_schema_descriptions, get_db_schema_and_samples
import os
import sys
import re
import sqlite3
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

MAX_ROWS_TO_DISPLAY = 20

_db = None


def get_db():
    """Get a cached SQLDatabase instance."""
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri(DB_PATH)
    return _db


@tool
def list_tables() -> List[str]:
    """데이터베이스에 있는 테이블 목록을 반환합니다."""
    db = get_db()
    return db.get_usable_table_names()


@tool
def get_schema(table_names: str, user_id: str = "", is_admin: bool = False) -> str:
    """
    주어진 테이블에 대한 스키마, 샘플 행, 그리고 한국어 설명을 반환합니다.
    사용자 권한에 따라 데이터 조회를 제한합니다. (일반 사용자는 자신의 데이터만 조회)
    """

    db = get_db()
    tables_list = [t.strip() for t in table_names.split(",") if t.strip()]

    # 1. 권한에 따른 스키마 및 샘플 데이터 가져오기
    schema_info = get_db_schema_and_samples(
        db_path=DB_PATH.replace("sqlite:///", ""),
        user_id=user_id,
        is_admin=is_admin, # is_admin 인자를 전달합니다.
        table_names_to_get=tables_list,
        for_prompt=True,  # 이 옵션을 사용하여 프롬프트용 문자열을 받음
    )
    if not schema_info:
        # 실패 시 LangChain의 기본 스키마 정보라도 반환
        schema_info = db.get_table_info(tables_list)

    try:
        descriptions = load_db_schema_descriptions()
        if not descriptions:
            return schema_info

        korean_descriptions = "\n\n---\n\n**테이블 및 컬럼 한국어 설명:**\n"
        has_description = False
        for table in tables_list:
            if table in descriptions:
                has_description = True
                table_desc = descriptions[table]
                korean_descriptions += f"\n- **테이블 `{table}`**: {table_desc.get('description', '설명 없음')}\n"
                col_descs = table_desc.get('columns', {})
                if col_descs:
                    for col, desc in col_descs.items():
                        korean_descriptions += f"  - `{col}`: {desc}\n"
        return schema_info + korean_descriptions if has_description else schema_info
    except Exception:
        return schema_info


@tool
def execute_query(query: str, user_id: str = "", is_admin: bool = False) -> str:
    """
    주어진 SQL 쿼리를 실행하고 결과를 JSON 형식으로 반환합니다.
    결과는 컬럼과 데이터 리스트를 포함하며, 건수가 많을 경우 일부만 반환합니다.
    일반 사용자의 경우, TB_BESTBANKER에 대한 쿼리는 해당 사용자 데이터로 자동 필터링됩니다。
    """
    db_path = DB_PATH.replace("sqlite:///","")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    modified_query = query
    # admin이 아니고, user_id가 있으며, 쿼리에 TB_BESTBANKER가 포함된 경우
    if not is_admin and user_id and re.search(r'\bTB_BESTBANKER\b', query, re.IGNORECASE):
        filtered_table_name = "USER_DATA_ONLY"
        
        # 쿼리에서 TB_BESTBANKER를 CTE 이름으로 교체 (대소문자 무시)
        modified_query = re.sub(r'\bTB_BESTBANKER\b', filtered_table_name, query, flags=re.IGNORECASE)
        
        # 쿼리에 이미 WITH 절이 있는지 확인
        if re.search(r'^\s*WITH', modified_query, re.IGNORECASE):
            # 이미 WITH 절이 있으면, CTE를 추가
            cte_addition = f" {filtered_table_name} AS (SELECT * FROM TB_BESTBANKER WHERE ENO = '{user_id}'),"
            modified_query = re.sub(r'^\s*WITH\s+', f'WITH{cte_addition} ', modified_query, count=1, flags=re.IGNORECASE)
        else:
            # WITH 절이 없으면, 새로 추가
            cte = f"WITH {filtered_table_name} AS (SELECT * FROM TB_BESTBANKER WHERE ENO = '{user_id}') "
            modified_query = cte + modified_query

    try:
        cursor.execute(modified_query)
        rows = cursor.fetchall()
        columns = (
            [description[0] for description in cursor.description]
            if cursor.description
            else []
        )

        truncated = False
        if "limit" not in modified_query.lower() and len(rows) > MAX_ROWS_TO_DISPLAY:
            rows = rows[:MAX_ROWS_TO_DISPLAY]
            truncated = True

        result = {"columns": columns, "data": rows, "truncated": truncated}
    except Exception as e:
        result = {"error": str(e)}
    finally:
        conn.close()

    return json.dumps(result, ensure_ascii=False)


@tool
def create_line_chart(
    data_json: str, title: str, x_axis: str, y_axes: List[str]
) -> str:
    """
    Generates and displays a line chart in the Streamlit app using Matplotlib
    based on the provided data. The y-axis is inverted for ranks.

    Args:
        data_json: A JSON string containing the data for the chart.
        title: The title of the chart.
        x_axis: The name of the column to be used for the x-axis.
        y_axes: A list of column names to be used for the y-axis.

    Returns:
        A message indicating the success or failure of the chart generation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.dates as mdates

    # 한글 폰트 설정 및 캐시 재빌드
    try:
        font_path = "c:\\Users\\whkim\\Desktop\\askally-20250927\\fonts\\NanumGothic.ttf"
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            # 폰트 캐시를 다시 빌드하도록 강제
            fm._rebuild()
            plt.rc("font", family="NanumGothic")
        elif sys.platform == "darwin":
            plt.rc("font", family="AppleGothic")
    except Exception as e:
        # 폰트 설정 실패 시에도 경고만 표시하고 진행
        st.warning(f"폰트 설정 중 오류가 발생했습니다: {e}. 일부 텍스트가 깨질 수 있습니다.")

    plt.rcParams["axes.unicode_minus"] = False

    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data["data"], columns=data["columns"])

        if x_axis not in df.columns or not all(y in df.columns for y in y_axes):
            return f"Error: One or more specified columns ({x_axis}, {y_axes}) not in the data."

        # 날짜 형식 변환
        df[x_axis] = pd.to_datetime(df[x_axis], format='%Y%m%d')
        df = df.sort_values(by=x_axis) # 날짜 기준으로 정렬
        df_chart = df.set_index(x_axis)

        fig, ax = plt.subplots(figsize=(10, 5)) # 크기 약간 조정

        # Plotting
        main_axes = [col for col in y_axes if 'ORD' in col or '순위' in col]
        score_axes = [col for col in y_axes if 'SCR' in col or '점수' in col]
        
        if main_axes:
            df_chart[main_axes].plot(ax=ax, marker='o', linestyle='-')
            ax.set_ylabel("순위", fontsize=10)
            ax.invert_yaxis()
        
        if score_axes:
            ax2 = ax.twinx() if main_axes else ax
            df_chart[score_axes].plot(ax=ax2, marker='s', linestyle='--')
            ax2.set_ylabel("점수", fontsize=10)

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel("기준일자", fontsize=10)
        
        # X축 포맷터 설정
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # 범례 통합
        lines, labels = ax.get_legend_handles_labels()
        if 'ax2' in locals() and ax2 is not ax:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax.legend(lines, labels, loc='best', fontsize=8)

        plt.tight_layout(pad=3.0)
        st.pyplot(fig)
        return f"성공적으로 '{title}' 라인 차트를 표시했습니다."
    except Exception as e:
        import traceback
        return f"라인 차트 생성에 실패했습니다: {e}\n{traceback.format_exc()}"


@tool
def create_pdf_report(summary: str, filename: str = "conversation_report.pdf", header_path: str = "assets/header_image.png", footer_path: str = "assets/footer_image.png") -> str:
    """
    Generates a professional PDF report from a given summary text, which can include markdown.
    The report content is placed within margins to avoid overlapping with the header and footer.

    Args:
        summary: The text content (can include markdown for titles, lists, and tables).
        filename: The name of the PDF file.
        header_path: Path to the header image.
        footer_path: Path to the footer image.

    Returns:
        A message indicating the success or failure of the report generation.
    """
    font_path = "fonts/NanumGothic.ttf"
    font_name = "NanumGothic"
    bold_font_name = "NanumGothicBold"

    if not os.path.exists(font_path):
        # Fallback for other platforms or if font is missing
        return f"Error: Font not found at {font_path}. Please add NanumGothic.ttf to the 'fonts' directory."

    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        if os.path.exists("fonts/NanumGothicBold.ttf"):
            pdfmetrics.registerFont(TTFont(bold_font_name, "fonts/NanumGothicBold.ttf"))
        else:
            bold_font_name = font_name

        # --- Calculate Margins ---
        page_width, page_height = A4
        header_height = 0
        footer_height = 0
        padding = 20  # 20 points padding

        if os.path.exists(header_path):
            img_reader = ImageReader(header_path)
            img_width, img_height = img_reader.getSize()
            aspect = img_height / float(img_width)
            header_height = page_width * aspect
        
        if os.path.exists(footer_path):
            img_reader = ImageReader(footer_path)
            img_width, img_height = img_reader.getSize()
            aspect = img_height / float(img_width)
            footer_height = page_width * aspect

        top_margin = header_height + padding
        bottom_margin = footer_height + padding

        # --- Create Document with correct margins ---
        doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=top_margin, bottomMargin=bottom_margin)

        # --- Define Styles ---
        styles = {
            'Title': ParagraphStyle(name='Title', fontName=bold_font_name, fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20),
            'h1': ParagraphStyle(name='h1', fontName=bold_font_name, fontSize=18, leading=22, spaceAfter=16, spaceBefore=12, textColor=colors.HexColor("#2C3E50")),
            'h2': ParagraphStyle(name='h2', fontName=bold_font_name, fontSize=14, leading=18, spaceAfter=12, spaceBefore=10, textColor=colors.HexColor("#34495E")),
            'BodyText': ParagraphStyle(name='BodyText', fontName=font_name, fontSize=10, leading=14, spaceAfter=8),
            'Bullet': ParagraphStyle(name='Bullet', fontName=font_name, fontSize=10, leading=14, leftIndent=20, spaceAfter=4),
        }
        
        story = []

        # --- Parsing Logic (same as before) ---
        def parse_markdown_table(table_lines):
            # ... (table parsing logic is unchanged)
            header = [h.strip() for h in table_lines[0].strip().strip('|').split('|')]
            data = []
            for row_line in table_lines[2:]:
                data.append([r.strip() for r in row_line.strip().strip('|').split('|')])
            table_data = [header] + data
            t = Table(table_data, repeatRows=1, hAlign='CENTER')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), bold_font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#DCE6F1")),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            return t

        lines = summary.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Convert markdown bold to reportlab bold tags
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)

            if line.startswith('# '):
                story.append(Paragraph(line.replace('# ', ''), styles['h1']))
                from reportlab.platypus import HRFlowable
                story.append(Spacer(1, 12))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#4F81BD")))
                story.append(Spacer(1, 12))
            elif line.startswith('## '):
                story.append(Paragraph(line.replace('## ', ''), styles['h2']))
            elif line.startswith('* ') or line.startswith('- '):
                story.append(Paragraph(f"• {line[2:]}", styles['Bullet']))
            elif '|' in line and i + 1 < len(lines) and '---' in lines[i+1]:
                table_lines = []
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                story.append(Spacer(1, 12))
                story.append(parse_markdown_table(table_lines))
                story.append(Spacer(1, 12))
                continue
            elif line:
                story.append(Paragraph(line, styles['BodyText']))
            i += 1

        # --- Header and Footer Drawing Function ---
        def add_header_and_footer(canvas, doc):
            canvas.saveState()
            # Header
            if os.path.exists(header_path):
                img_reader = ImageReader(header_path)
                img_width, img_height = img_reader.getSize()
                aspect = img_height / float(img_width)
                new_width = doc.pagesize[0]
                new_height = new_width * aspect
                canvas.drawImage(header_path, 0, doc.pagesize[1] - new_height, width=new_width, height=new_height, preserveAspectRatio=True, anchor='n')
            # Footer
            if os.path.exists(footer_path):
                img_reader = ImageReader(footer_path)
                img_width, img_height = img_reader.getSize()
                aspect = img_height / float(img_width)
                new_width = doc.pagesize[0]
                new_height = new_width * aspect
                canvas.drawImage(footer_path, 0, 0, width=new_width, height=new_height, preserveAspectRatio=True, anchor='s')
            canvas.restoreState()

        doc.build(story, onFirstPage=add_header_and_footer, onLaterPages=add_header_and_footer)

        return f"Successfully generated professional PDF report: {filename}"
    except Exception as e:
        import traceback
        return f"Failed to generate PDF report: {e}\n{traceback.format_exc()}"


def get_rag_tool():
    vector_store = st.session_state.get("vector_store")
    if not vector_store:
        return None
    retriever = vector_store.as_retriever()
    return create_retriever_tool(
        retriever,
        "pdf_document_retriever",
        "Searches and returns information from the user's PDF documents.",
    )


def get_sql_tools():
    return [list_tables, get_schema, execute_query]
