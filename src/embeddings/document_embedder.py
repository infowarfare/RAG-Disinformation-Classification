from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder

def get_voyage_document_embedder():

    doc_embedder = VoyageDocumentEmbedder(
    model="voyage-3-large",
    input_type="document",
    )

    return doc_embedder


    
