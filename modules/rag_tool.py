from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.tools import BaseTool


class BankManualTool(BaseTool):
    """RAG tool based on bank product manuals."""

    name = "bank_manual_rag"
    description = "Use this tool to answer questions about bank products based on manuals."

    def __init__(self, pdf_paths: List[str], model_name: str):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.model_name = model_name
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        docs = []
        for path in self.pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)
        return vectordb.as_retriever()

    def _run(self, query: str) -> str:
        llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever)
        return chain.run(query)

    async def _arun(self, query: str) -> str:
        return self._run(query)
