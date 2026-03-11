from haystack.document_stores.in_memory import InMemoryDocumentStore

def get_in_memory_document_store():
    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return doc_store

# TODO:
# Chroma document store