from haystack.document_stores.in_memory import InMemoryDocumentStore
from milvus_haystack import MilvusDocumentStore


def get_in_memory_document_store():
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return document_store

def get_milvus_document_store():
    
    document_store = MilvusDocumentStore(
        connection_args={"uri": "./milvus.db"},
        collection_name="Milvus VectorDB",
        collection_description="VectorDB for testing the RAG Pipeline",
        drop_old=True,
        search_params={"metric_type": "COSINE"}
    )

    return document_store


# Chroma document store