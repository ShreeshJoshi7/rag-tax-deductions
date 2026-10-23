# TaxWise: A Retrieval-Augmented Assistant for Australian Work-Related Tax Deductions

**Group 73 (Project RAG)** | COSC2669 Work Integrated Learning Project, RMIT University, 2026 | Mentor: Akriti Verma

TaxWise answers questions about Australian employee work-related tax deductions **only from official ATO guidance**. It finds the relevant passages in 51 ATO pages, writes a short plain-English answer, names the pages it used, and says "I don't have enough information" when the question is outside what those pages cover.

It gives general information, not personal tax advice.

## Team

| Student ID | Name | Role |
|---|---|---|
| s4206111 | Shreesh Anil Joshi | RAG Pipeline Lead |
| s4149947 | Athith Mukkalu Suresh Babu | Evaluation Engineer |
| s4164152 | Sahana Nataraja | Data Engineer |
| s4234052 | Brunda Shankar | Frontend Engineer |
| s4159521 | Vaishnavi Sreenath | Environment Lead and Frontend |

Per-person contributions are in the written report (Table 6) and the contribution sheet.

## How it works

```
Prepared once:  1. Collect (51 ATO pages) -> 2. Chunk (366 chunks, 1000 chars, 150 overlap)
                -> 3. Embed (all-MiniLM-L6-v2) -> 4. Store (ChromaDB)
Per question:   5. Retrieve top 3 chunks -> 6. Relevance gate (best cosine distance <= 0.50?)
                   no  -> refuse, language model is not called
                   yes -> 7. Groq gpt-oss-120b writes a cited answer from the 3 chunks
```

Follow-up questions ("What about if I lease it instead?") are rewritten into stand-alone questions using the last four answered turns.

## Tech stack

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (free, runs locally)
- Vector store: ChromaDB (cosine distance)
- Language model: `openai/gpt-oss-120b` on Groq's free API (temperature 0.1, max 700 tokens). We first tried Qwen3 8B on Ollama, but it was too slow on CPU laptops.
- Orchestration: LangChain
- Interface: Streamlit

## Repository structure

The code sits in the repository root so that the scripts' relative paths (`docs/`, `db/chroma_db`) work.

```
README.md
requirements.txt
.env.example               # copy to .env and add your own GROQ_API_KEY
.gitignore

scrape_ato.py              # crawl the ATO "Deductions you can claim" section into docs/
review_docs.py             # list scraped pages into docs/_review.csv for review
trim_docs.py               # move out-of-scope pages to docs_excluded/
ingest.py                  # chunk, embed and store pages in db/chroma_db
test_retrieval.py          # run sample questions through retrieval only
query.py                   # command-line RAG: retrieve, gate, generate, follow-ups
streamlit_app.py           # chat web app
evaluation_questions.csv   # the 30-question test set
run_evaluation.py          # scores answer-versus-refuse behaviour
evaluation_results.csv     # output of the final evaluation run

docs/                      # the 51 cleaned ATO pages (.txt) + _sources.csv (page URLs)
reports/                   # Milestone 1 report, written report, slides
```

Not committed: `.env` (API key), `db/` (rebuilt by `ingest.py`), `conversations.json` (created by the app), `docs_excluded/`.

## Setup and run

Python 3.9 or later.

```bash
python -m venv venv
venv\Scripts\activate            # Windows. On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # then put your free Groq key in .env (console.groq.com/keys)

python ingest.py                 # build the knowledge base from docs/ (one-off)
python query.py "What car expenses can I claim?"
python query.py                  # interactive mode with follow-up memory
streamlit run streamlit_app.py   # web chat app
python run_evaluation.py         # 30-question evaluation, writes evaluation_results.csv
```

To collect the ATO pages again: `python scrape_ato.py`, then `python review_docs.py`, then `python trim_docs.py`. The ATO site can return HTTP 403 to cloud or data-centre IPs, so run the scraper from a normal home or university connection.

## Results

**Walert baseline reproduction** (RMIT's Walert, Cherumanal et al., CHIIR '24; 120 FAQ passages):

| Method | NDCG@1 | NDCG@3 | NDCG@5 |
|---|---|---|---|
| Intent-based (Walert original) | 0.0833 | 0.0391 | 0.0391 |
| RAG with BM25 | 0.1667 | 0.2566 | 0.3291 |
| RAG with dense (FAISS) | 0.2500 | 0.2380 | 0.2380 |

This shows retrieval works better than the intent-based design on Walert's data. It does not prove the same on tax.

**TaxWise evaluation:** 28 of 28 automatically scored questions got the correct answer-or-refuse decision (20 in scope, 1 vague, 7 out of scope). Two further vague questions were reviewed by hand. This measures the answer-or-refuse decision, **not** whether every sentence of an answer is correct.

**Known limitations:** the test set is small and written by the team, the 0.50 threshold was chosen by inspection rather than tuned, some pages have character-encoding damage, the Groq free tier is rate-limited and third-party hosted, and faithfulness and citation correctness were not measured. See Sections 4.3 and 4.4 of the written report.

## Project management

- Trello board: https://trello.com/b/LBDCAcWa/project-rag
- Three sprints between 23 Aug and 22 Oct 2026. The Trello board was rebuilt from completed work on 2 October, so it is a reconstruction. See Section 5 of the written report.

## Status

Complete for submission: pipeline, Streamlit app, evaluation and reports. Next steps are a content-accuracy review of answers, a larger independent test set, hybrid (BM25 + dense) retrieval, retry handling for rate limits, and fixing the UTF-8 text encoding.

## Disclaimer

TaxWise provides general information drawn from public ATO pages. It is not tax, legal or financial advice.
