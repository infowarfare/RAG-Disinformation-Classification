from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

def get_in_memory_retriever(doc_store, num_k=3):
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store, top_k=num_k)
    return retriever

def get_milvus_retriever(doc_store, num_k=3):
    retriever = MilvusEmbeddingRetriever(document_store=doc_store, top_k=num_k)
    return retriever


