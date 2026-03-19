from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
import os


def get_in_memory_document_store():
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return document_store
