import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

import pandas as pd
from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter
from prompts.prompt_builder import get_prompt_builder
from doc_stores.document_store import get_in_memory_document_store
from embeddings.query_embedder import get_voyage_query_embedding
from embeddings.document_embedder import get_voyage_document_embedder
from retrievers.retriever import get_in_memory_retriever
from models.google_genai_generator import get_gemini_generator
from models.openai_generator import get_openai_generator
from metrics.save_metrics import save_metrics_to_file
from metrics.calculate_metrics import evaluate
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

from dotenv import load_dotenv
# Load API Keys
load_dotenv() 


# PARAMS
TOP_K = 4 # 4, 6, 8
TRACE_NAME = f"gemini-3.1-flash-lite-preview_TOP_K={TOP_K}"


# Init
document_store = get_in_memory_document_store()
doc_writer = DocumentWriter(document_store=document_store)
retriever = get_in_memory_retriever(doc_store=document_store, num_k=TOP_K)
query_embedder = get_voyage_query_embedding()  # text embedder
doc_embedder = get_voyage_document_embedder() 
prompt_builder = get_prompt_builder()
chat_generator = get_gemini_generator()


# Load csv data as docs
df = pd.read_csv("src\\german_disinfo_dataset\\processed\\train_clean.csv", encoding="utf-8", sep=";")

# Convert to Haystack Documents
documents = [
    Document(content=row["text"], meta={"label": row["binary_label"]}) 
    for _, row in df.iterrows()
]

# Embed documents
docs_with_embeddings = doc_embedder.run(documents)
document_store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

# RAG Pipeline
pipeline = Pipeline()
# components
pipeline.add_component("tracer", LangfuseConnector(TRACE_NAME))
pipeline.add_component("text_embedder", query_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", chat_generator)

# Connect
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder")
pipeline.connect("prompt_builder.prompt", "llm.messages")

def draw_pipeline() -> None:
    pipeline.draw(path="src\\visualization")


def execute_pipeline(text: str) -> int:
    
    response = pipeline.run({"text_embedder": {"text": text}, "prompt_builder": {"query": text}})
    cls = response["llm"]["replies"][0].text

    # catches invalid input
    if "0" or "1" in cls:
        if "0" in cls:
            cls = 0
        elif "1" in cls:
            cls = 1
    return cls

def execute_rag_classification_pipeline() -> None:

    # load test data
    df = pd.read_csv("src\\german_disinfo_dataset\\processed\\test_clean.csv", encoding="utf-8", sep=";")
    # get labels
    actual_labels = df['binary_label'].tolist()
    # empty list for classification
    predicted_labels = []

    # loop through test dataset
    for count, (_, row) in enumerate(df.iterrows()):
        cls = execute_pipeline(row['text'])
        print(f"No.: {count+1} | Prediction: {cls}")
        predicted_labels.append(cls)

    pred_metrics = evaluate(predictions=predicted_labels, gold_labels=actual_labels)
    print(f"Macro-F1: {pred_metrics["Macro-F1"]}")

    save_metrics_to_file(model_name="gemini-3.1-flash-lite-preview", top_k=TOP_K, pred_metrics=pred_metrics, folder_path="src\\results")

def main():
    execute_rag_classification_pipeline()
    #draw_pipeline()

if __name__ == "__main__":
    main()


    

    





    


    
    