from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever


load_dotenv(override = True)


SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def compression_context_retriever(query: str, vector_store_retriever,k : int):
    llm_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", api_key =os.environ["GOOGLE_API_KEY"])
    compressor = LLMChainExtractor.from_llm(llm_model)
    compressor_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_store_retriever)
    results = compressor_retriever.invoke(query)
    return results[:k]


def ensemble_search(query: str, k: int, retrievers, weights):
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
    results = ensemble_retriever.invoke(query)
    return results[:k]

class HybridRetriever:
    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int = 3):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1- bm25_weight
        self.embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", api_key = os.environ['GOOGLE_API_KEY'])
        self.vector_database = Chroma.from_documents(documents, self.embeddings)
        self.vector_retriever = self.vector_database.as_retriever(search_kwargs={"k": k})
        self.bm25_retriever = BM25Retriever.from_documents(documents,  k=k)
        self.llm_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", api_key =os.environ["GOOGLE_API_KEY"])
    
    def search(self, query: str):
        retriever = compression_context_retriever(query, self.vector_retriever, k = self.k)
        return retriever
    
    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def call_llm(self, question):
        prompt_template = ChatPromptTemplate.from_template("""
            Answer the questions only on the following context and also when you specify make sure to specify current date to the user
            {context}
            Question: {question}

            Answer: 

            Make sure to respond in concise manner and if you don't know the answer, say you don't know.
            """
        )

        rag_chain = {"context":RunnableLambda(self.search) | self.format_docs, "question": RunnablePassthrough()} | prompt_template | self.llm_model | StrOutputParser()

        try:
            result = rag_chain.invoke(question)
            return result
        except Exception as e:
            return f"Error occurred while processing the question: {str(e)}"

if __name__ == "__main__":
    hybrid_retriever = HybridRetriever(SAMPLE_DOCS, bm25_weight=0.5, k=3)
    question = "What is LangChain?"
    answer = hybrid_retriever.call_llm(question)
    print(f"Question: {question}\nAnswer: {answer}")



    



    

