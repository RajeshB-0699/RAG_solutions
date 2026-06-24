from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from typing import List

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

retriever = HybridRetrieval(documents, bm25_weight=0.5, k = 4)
results = retriever.search('SKU-7742X specifications')
for doc in results:
    print(f"The retrived content : {doc.page_content[:100]}")
    

