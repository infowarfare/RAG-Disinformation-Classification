from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder

def get_voyage_query_embedding():
    query_embedder = VoyageTextEmbedder(model="voyage-3-large", input_type="query")
    return query_embedder
