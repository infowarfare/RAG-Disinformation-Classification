import pandas as pd
import json
import uuid
from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter
from prompts.prompt_builder import get_prompt_builder
from doc_stores.document_store import get_in_memory_document_store
from embeddings.query_embedder import get_voyage_query_embedding
from embeddings.document_embedder import get_voyage_document_embedder
from retrievers.retriever import get_in_memory_retriever
from models.google_genai_generator import get_gemini_generator
from eval.eval_metrics import save_metrics_to_file
from datetime import datetime

from dotenv import load_dotenv

# Eval metrics
from sklearn.metrics import f1_score,  precision_score, recall_score, matthews_corrcoef

# Load API Keys
load_dotenv() 

# Init
doc_store = get_in_memory_document_store()
doc_writer = DocumentWriter(document_store=doc_store)
retriever = get_in_memory_retriever(doc_store=doc_store, num_k=15)
query_embedder = get_voyage_query_embedding()  # text embedder
doc_embedder = get_voyage_document_embedder()
prompt_builder = get_prompt_builder()
chat_generator = get_gemini_generator() # Gemini-3-flash-preview

# Load csv data as docs
df = pd.read_csv("src\\data\\propaganda_train.csv", encoding="utf-8")

# Convert to Haystack Documents
documents = [
    Document(content=row["text"], meta={"label": row["label"]}) 
    for _, row in df.iterrows()
]



# Embed documents
docs_with_embeddings = doc_embedder.run(documents)
doc_store.write_documents(docs_with_embeddings["documents"], policy=DuplicatePolicy.OVERWRITE)

# RAG Pipeline
pipeline = Pipeline()
# components
pipeline.add_component("text_embedder", query_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", chat_generator)

# Connect
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder")
pipeline.connect("prompt_builder.prompt", "llm.messages")


def execute_pipeline(text: str) -> int:
    """
    Executes a Haystack pipeline using the provided input text.

    This function wraps a Haystack `Pipeline.run()` call. It passes the input 
    string to the starting component (e.g., a 'text_embedder' or 'prompt_builder') 
    and processes the resulting dictionary to return a specific integer value.

    Args:
        text (str): The input query or document text to be processed by 
            the pipeline.

    Returns:
        int: The result of the pipeline execution
    """
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
    """
    Excute the full RAG-based classification workflow.

    This function initializes a Haystack pipeline, retrieves relevant 
    documents, and passes them to an LLM to classify input data. 
    Results are typically logged or saved to the filesystem.

    Returns:
        None
    """

    # load test data
    df = pd.read_csv("src\\data\\propaganda_test.csv", encoding="utf-8")
    # get labels
    actual_labels = df['label'].tolist()
    # empty list for classification
    predicted_labels = []

    # loop through test dataset
    for count, (_, row) in enumerate(df.iterrows()):
        cls = execute_pipeline(row['text'])
        print(f"No.: {count+1} | Prediction: {cls}")
        predicted_labels.append(cls)

    pred_metrics = {
        "predictions": predicted_labels,
        "Macro-F1": f1_score(actual_labels, predicted_labels, average='macro'),
        "Precision": precision_score(actual_labels, predicted_labels, average='macro'),
        "Recall": recall_score(actual_labels, predicted_labels, average='macro'),
        "MCC": matthews_corrcoef(actual_labels, predicted_labels)
    }

    print(f"Macro-F1: {pred_metrics["Macro-F1"]}")

    save_metrics_to_file("gemini_rag_clf", "gemini-3-flash-preview", pred_metrics, folder_path="src\\eval_results")

def main():
    execute_rag_classification_pipeline()

if __name__ == "__main__":
    main()


    

    





    


    
    