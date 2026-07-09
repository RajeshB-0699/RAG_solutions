from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv
import os
import logging

load_dotenv(override = True)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

def create_vector_base():
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", api_key = os.environ['GOOGLE_API_KEY'])
    vector_store = Chroma.from_documents(
        documents= TECH_DOCS,
        embedding= embeddings
    )
    return vector_store

def advanced_rag_chain():
    vector_store = create_vector_base()
    vetor_retriever = vector_store.as_retriever(search_kwargs= {'k':3})
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", api_key = os.environ['GOOGLE_API_KEY'])

    multi_retriever = MultiQueryRetriever.from_llm(
        llm = llm,
        retriever= vetor_retriever
    )

    compressor = LLMChainExtractor.from_llm(llm)
    context_compressor = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_retriever
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the following context. Be specific and cite which technologies you're referring to.

Context:
{context}

Question: {question}

Answer:"""
    )


    def fomat_docs(docs):
        return "\n]n".join([doc.page_content for doc in docs])
    
    rag_chain = ({'context': context_compressor | fomat_docs , 'question': RunnablePassthrough()} 
                | prompt
                | llm
                | StrOutputParser()
                )
    
    questions = [
        "What options do I have for building AI agents?",
        "How can I store and search embeddings?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")



if __name__ == "__main__":
    advanced_rag_chain()