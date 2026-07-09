from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from typing import List
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv(override = True)

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]


def hybrid_retriever(query: str, retriever: list, weight: list, k : int):
    retriever = EnsembleRetriever(retrievers= retriever, weights= weight, k = k)
    results = retriever.invoke(query)
    return results[:k]


class HybridRetrieval:
    def __init__(self, documents: List[Document], bm25_weight: float=0.5, k: int = 4):
        self.bm25_weight = bm25_weight
        self.k = k
        self.vector_weight = 1-bm25_weight
        self.embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001",
                                                 api_key = os.environ['GOOGLE_API_KEY'])
        self.vector_store = Chroma.from_documents(documents, self.embeddings, collection_name = "hybrid_search")
        self.vector_retriever  = self.vector_store.as_retriever(search_kwargs = {'k': k})
        self.bm25_retriever = BM25Retriever.from_documents(documents, k = k)
        self.llm_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", api_key = os.environ['GOOGLE_API_KEY'])

    @traceable(name="Ensemble Search Check", run_type="retriever")
    def search(self, query: str) -> List[Document]:
        return hybrid_retriever(query, 
                                retriever = [self.bm25_retriever, self.vector_retriever], 
                                weight =[self.bm25_weight, self.vector_weight],
                                k = self.k
                            )
    
    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)
        all_docs = self.vector_store.get()
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content= doc) for doc in all_docs['documents']],
            k = self.k
        )

    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])

    @traceable(name="RAG pipeline check", run_type="chain")
    def call_llm(self, question):
        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the questions only on the following context and also when you specify make sure to specify current date to the user
            {context}
            Question: {question}

            Answer: 

            Make sure to respond in concise manner and if you don't know the answer, say you don't know.
            """
        )

        rag_chain = (
            {'context': RunnableLambda(self.search) | self.format_docs, 'question': RunnablePassthrough()}
            | prompt_template
            | self.llm_model
            | StrOutputParser()
        )

        try:
            result = rag_chain.invoke(question)
            return result

        except Exception as e:
            print(f"Unable to invoke RAG chain : {e}")


    

retriever = HybridRetrieval(documents, bm25_weight=0.5, k = 4)
query2 = "How authentication works?"
result = retriever.call_llm(query2)
print(result)
    

