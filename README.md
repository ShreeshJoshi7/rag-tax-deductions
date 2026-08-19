# [rag-tax-deductions] — Test-Driven RAG for Australian Individual Tax Deductions

## Group ID
Team RAG

## Team Members
| Student ID | Full Name | Role |
|---|---|---|
| s4206111 | Shreesh Anil Joshi | RAG Pipeline — retrieval, embeddings, generation, vector DB integration |
| s4149947 | Athith M S | Evaluation Framework — designing test question sets, measuring faithfulness, effectiveness, and unanswered-question rate |
| s4164152 | Sahana Shetty | Data Engineering — sourcing, cleaning, and chunking ATO documents into the knowledge base |
| s4234052 | Brunda Shankar | Frontend & Demo — building the Streamlit interface and user-facing app |
| s4159521 | Vaishnavi Sreenath | Project Coordination & Reporting — Trello board management, milestone reports, and documentation |

## Project Aim
A Retrieval-Augmented Generation assistant that helps Australian individual
taxpayers understand their eligible income tax deductions by grounding
answers in official ATO guidance, evaluated on answer faithfulness and
unanswered-question rate.

## Reference Architecture
This project reproduces and adapts the RAG pipeline from
[Walert](https://github.com/rmit-ir/walert) (RMIT's FAQ Open Day assistant),
applying it to the Australian tax deductions domain.

## Tech Stack
- Embeddings: sentence-transformers
- Vector store: ChromaDB
- LLM: [OpenAI API / Claude API / Ollama — TBD]
- Interface: Streamlit

## Repository Structure
```
/data          # ATO source documents / processed chunks
/src           # RAG pipeline code (retrieval, generation)
/eval          # Evaluation framework and test question sets
/app           # Streamlit interface
/docs          # Reports, milestone submissions
```

## Status
Successfully reproduced the Walert RAG pipeline (encoding, indexing, BM25 and dense/FAISS retrieval, and NDCG-based evaluation) using their original FAQ dataset. Results confirm RAG-based retrieval (BM25: NDCG@5 = 0.329; Dense/FAISS: NDCG@1 = 0.250) substantially outperforms Walert's original intent-based system (NDCG@5 = 0.039), validating RAG as the stronger approach before adapting the pipeline to the ATO tax deductions domain.
