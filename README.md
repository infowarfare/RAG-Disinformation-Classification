# Retrieval Augmented Disinformation Detection in German

Dieses Projekt implementiert ein hybrides System zur Erkennung von Desinformation. Durch Verwendung von **Retrieval-Augmented Generation (RAG)** mit aktuellen Sprachmodellen (LLMs/SLMs) wird eine präzise Klassifizierung in Echtzeit ermöglicht.

## 🛠 Tech Stack

* **Framework:** [Haystack](https://haystack.deepset.ai/) (Orchestrierung der RAG-Pipelines)
* **Vector Database:** InMemoryDB (Migration auf **Milvus** geplant)
* **Embeddings** `voyage-3-large` (MongoDB: https://https://www.voyageai.com/)

## 🧠 Model Zoo

Evaluiert werden aktuelle Sprachmodelle sowie BERT-Modelle + Logistische Regression als Baseline

### 1. Large Language Models (Cloud)
* **Google:** Gemini 3.1
* **OpenAI:** ChatGPT 5.4
* **Anthropic:** Claude 3.5 Sonnet (4.6)

### 2. Small Language Models (SLMs)
Optimiert für lokale Ausführung und spezifische Aufgaben:

| Anbieter | Modell-Bezeichnung | HuggingFace Pfad |
| :--- | :--- | :--- |
| **German (VAGO Solutions)** | SauerkrautLM (8b, 2b, 1.5b) | `VAGOsolutions/Llama-3.1-SauerkrautLM...` |
| **Meta** | Llama 3.2 (1b, 3b) & 3.1 (8b) | `meta-llama/Llama-3.2...` | (= Llama-3.1-SauerkrautLM-8b-Instruct)
| **Google** | Gemma 3 (4b, 1b) & 2 (2b) | `google/gemma-3...` | (= SauerkrautLM-gemma-2-2b-it)
| **Microsoft** | Phi-4 mini (3.8b) | `microsoft/Phi-4-mini-instruct` |
| **Alibaba (Qwen)** | Qwen 2.5 (3b & 1.5b) | `Qwen/Qwen2.5-3B-Instruct` / `-1.5B` |
| **Alibaba (Qwen)** | Qwen 2 (1.5b) | `Qwen/Qwen2-1.5B-Instruct` (≡ SauerkrautLM-1.5b) |

### 3. Deutsche BERT/Encoder Modelle
Für NLU-Tasks und Klassifizierungen:
* `distilbert-base-german-cased`
* `deepset/gbert-large` | `deepset/gbert-base`
* `GottBERT/GottBERT_base_best` (German RoBERTa)
* `LSX-UniWue/ModernGBERT_134M` (German ModernBERT)

### 4. Machine Learning mit Logistischer Regression
Klassifizierung mit LogReg + TF-IDF und SBERT Embeddings
* `T-Systems-onsite/cross-en-de-roberta-sentence-transformer` (SBERT-Embeddings)

## 📋 Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/infowarfare/RAG-Disinformation-Classification.git]
    cd RAG-Disinformation-Classification
    ```

2.  **Virtuelle Umgebung:**
    ```bash
    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Linux/Mac: source .venv/bin/activate
    ```

3.  **Abhängigkeiten:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Konfiguration:**
    Erstelle eine `.env` Datei im Verzeichnis `src/`.
    *Wichtig: Stelle sicher, dass `src/.env` in deiner `.gitignore` steht!*

## 🔬 Methodik
Die Evaluation verfolgt einen mehrstufigen Analyse-Ansatz:
1.  **Baseline:** BERT & Logistische Regression (TF-IDF | SBERT) für schnelle statistische Einordnung.
2.  **Semantik:** Extraktion von Embeddings via `voyage-3-large`.
3.  **Verification:** Haystack RAG-Pipeline zur Klassifizierung von Desinformation mit Hilfe von Vektordatenbanken.
4.  **Evaluation:** Evaluation anhand des Macro-F1 Scores und MCC
5.  **Statistischer Signifikanztest:** Signifikanztest mittels Paired-Bootstrap Test

---
*Dieses Projekt dient der automatisierten Unterstützung bei der Identifikation von Desinformation. Ergebnisse sollten stets kritisch hinterfragt werden.*