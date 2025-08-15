from typing import List, Dict, Any
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.tools import BaseTool


class SQLQueryTool(BaseTool):
    """Tool for querying SQL databases using natural language."""

    name = "sql_query"
    description = "Use this tool to answer questions over the configured database."

    def __init__(self, db: SQLDatabase, llm: ChatOpenAI):
        super().__init__()
        self.db = db
        self.llm = llm
        self.query_chain = create_sql_query_chain(self.llm, self.db)

    def _run(self, query: str) -> str:
        sql = self.query_chain.invoke({"question": query})
        df = pd.read_sql(sql, self.db._engine)
        return df.to_json(orient="records")

    async def _arun(self, query: str) -> str:
        return self._run(query)


def get_db_tools(db_info: Dict[str, Any], model_name: str) -> List[BaseTool]:
    """Create SQL query tools based on DB info."""
    if db_info["type"] == "sqlite":
        uri = f"sqlite:///{db_info.get('path', 'sqlite.db')}"
    elif db_info["type"] == "mysql":
        host = db_info.get("host")
        user = db_info.get("user")
        password = db_info.get("password")
        database = db_info.get("db")
        if not all([host, user, password, database]):
            raise ValueError("MySQL 연결 정보가 부족합니다.")
        uri = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
    else:
        raise ValueError("지원하지 않는 DB 유형입니다.")

    db = SQLDatabase.from_uri(uri)
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    sql_tool = SQLQueryTool(db=db, llm=llm)
    return [sql_tool]
