# modules/tools.py

import sqlite3
import pandas as pd
import json
from fpdf import FPDF
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_core.tools import tool

from .config import DB_PATH

class PythonREPL:
    def __init__(self):
        self.globals = {}
        self.locals = {}

    def __call__(self, code: str) -> str:
        try:
            # matplotlib, seaborn, pandas를 자동으로 임포트
            exec("import matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd", self.globals)
            
            output_buffer = []
            
            # 코드 실행 및 출력 캡처
            def custom_print(*args, **kwargs):
                output_buffer.append(" ".join(map(str, args)))

            self.locals['print'] = custom_print
            
            # 코드를 실행하여 차트 생성
            exec(code, self.globals, self.locals)
            
            # 생성된 차트 파일 경로 찾기
            image_path = None
            for var in self.locals.values():
                if isinstance(var, str) and var.endswith('.png'):
                    image_path = var
                    break
            
            # 저장된 이미지 파일 경로가 있으면 반환
            if image_path and os.path.exists(image_path):
                 return f"차트가 성공적으로 생성되었습니다. 파일 경로: {image_path}"

            # print()로 출력된 내용이 있으면 반환
            if output_buffer:
                return "\n".join(output_buffer)

            return "코드가 실행되었지만, 생성된 차트 파일 경로나 출력값을 찾을 수 없습니다."

        except Exception as e:
            return f"코드 실행 중 오류 발생: {e}"

# 도구 정의
python_repl = PythonREPL()

@tool
def execute_sql_and_get_results(query: str) -> str:
    """
    SQL 쿼리를 실행하고 결과를 JSON 형식으로 반환합니다.
    [수정] 쿼리에 LIMIT이 없으면 자동으로 'LIMIT 10'을 추가하여 반환 데이터 양을 제한합니다.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # --- FIX ---
    # 쿼리에 LIMIT절이 없는 경우, 최대 10건만 조회하도록 LIMIT 10 추가
    if "limit" not in query.lower():
        # 세미콜론이 있으면 그 앞에, 없으면 맨 뒤에 추가
        if query.strip().endswith(';'):
            query = query.strip()[:-1] + " LIMIT 10;"
        else:
            query += " LIMIT 10"

    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return json.dumps({
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient='split')['data']
        }, ensure_ascii=False, indent=2)
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        conn.close()
        return json.dumps({"error": str(e)})

@tool
def list_tables() -> str:
    """데이터베이스의 모든 테이블 목록을 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ", ".join(tables)

@tool
def get_schema(table_names: str) -> str:
    """주어진 테이블의 스키마 정보를 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    schemas = {}
    for table_name in table_names.split(','):
        table_name = table_name.strip()
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1;", conn)
            schemas[table_name] = df.dtypes.to_string()
        except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
            schemas[table_name] = f"Error: {e}"
    conn.close()
    return json.dumps(schemas, indent=2)

@tool
def create_pdf_report(title: str, content: str) -> str:
    """주어진 제목과 내용으로 PDF 보고서를 생성하고 파일 경로를 반환합니다."""
    # 유니코드(한글) 지원을 위한 폰트 설정
    try:
        # 시스템에 설치된 나눔고딕 폰트 사용 (경로는 시스템에 따라 다를 수 있음)
        font_path = "NanumGothic.ttf" # 로컬에 폰트 파일이 있는 경우
        if not os.path.exists(font_path):
            # MacOS 경로
            font_path_mac = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
            # Windows 경로
            font_path_win = "c:/Windows/Fonts/malgun.ttf"
            if os.path.exists(font_path_mac):
                font_path = font_path_mac
            elif os.path.exists(font_path_win):
                font_path = font_path_win
            else: # 기본 폰트로 대체 (한글 깨질 수 있음)
                 font_path = 'DejaVuSans.ttf' # 로컬에 폰트 파일이 있는 경우
    except Exception:
        font_path = 'DejaVuSans.ttf'
        
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('NanumGothic', '', font_path, uni=True)
    pdf.set_font('NanumGothic', '', 12)
    
    # 제목 추가
    pdf.set_font('NanumGothic', 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(10)
    
    # 내용 추가
    pdf.set_font('NanumGothic', '', 12)
    # 한글 출력을 위해 .encode('latin-1').decode('utf-8') 대신 직접 유니코드 사용
    pdf.multi_cell(0, 10, content)
    
    file_name = f"report_{uuid.uuid4().hex[:8]}.pdf"
    pdf.output(file_name)
    return file_name