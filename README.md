# Local Adaptive-RAG with LLaMA3

This project implements a **Local Retrieval-Augmented Generation (RAG) agent** using the **LLaMA3** model, combining innovative techniques from several key RAG papers. The agent is designed to efficiently route queries, handle irrelevant documents, and self-correct hallucinated answers.

## Key Features

### 1. **Routing (Adaptive RAG)**
   - Adapts to the nature of the question by routing it to the most appropriate retrieval approach. 
   - Utilizes both vector-based retrieval and other techniques based on the question type.

### 2. **Fallback (Corrective RAG)**
   - If the retrieved documents are deemed irrelevant to the query, the agent falls back to web search, ensuring a higher chance of providing useful information.

### 3. **Self-Correction (Self-RAG)**
   - After generating an answer, the agent evaluates whether the response is grounded in the retrieved documents and relevant to the question.
   - If hallucinations are detected or the answer doesn't address the question, the agent corrects the output or provides a fallback mechanism.


## Local Models

The project utilizes local models for both **embedding** generation and **LLM-based answer generation**.


