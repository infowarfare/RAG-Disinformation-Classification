from haystack.document_stores.in_memory import InMemoryDocumentStore
from milvus_haystack import MilvusDocumentStore
from haystack.utils import Secret
import os


def get_in_memory_document_store():
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return document_store

def get_milvus_document_store():
    
    document_store = MilvusDocumentStore(
        collection_name="test_db",
        connection_args={
        "uri": "https://in03-aaeed774ea9d1b8.serverless.aws-eu-central-1.cloud.zilliz.com",  # Your Public Endpoint
        "token": os.getenv("ZILLIZ_CLOUD_API_KEY"),  # API key, we recommend using the Secret class to load the token from env variable for security.
        "secure": True
        },
        drop_old=True,
    )
    
    return document_store


# Chroma document store