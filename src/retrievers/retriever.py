from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

def get_in_memory_retriever(doc_store, num_k=3):
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store, top_k=num_k)
    return retriever

# TODO:
# Implement ChromaEmbeddingRetriever



