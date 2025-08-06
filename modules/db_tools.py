from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from typing import List, Dict, Any


def get_db_tools(db_info: Dict[str, Any], model_name: str) -> List[BaseTool]:
    """
    데이터베이스 정보에 따라 SQLDatabaseToolkit을 사용하여 SQL 도구를 생성합니다.
    """
    if db_info["type"] == "sqlite":
        db_path = db_info.get("path", "sqlite.db")
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    elif db_info["type"] == "mysql":
        host = db_info.get("host")
        user = db_info.get("user")
        password = db_info.get("password")
        database = db_info.get("db")
        if not all([host, user, password, database]):
            raise ValueError("MySQL 연결 정보가 부족합니다.")

        # mysql-connector-python을 사용하도록 URI를 구성합니다.
        uri = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
        db = SQLDatabase.from_uri(uri)
    else:
        return []

    llm = ChatOpenAI(model_name=model_name, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    return toolkit.get_tools()
