"""
Quick retrieval sanity check.
--------------------------------
Loads the existing Chroma vector store (built by ingest.py) and runs a
few sample questions through it, printing the top matching chunks so you
can eyeball whether retrieval is actually pulling back relevant content.

This does NOT call any LLM -- it's just testing the retrieval half of
the pipeline (the "R" in RAG) before we add generation on top.

Usage:
    python test_retrieval.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "db/chroma_db"

# Edit this list to try your own questions
TEST_QUESTIONS = [
    "What car expenses can I claim?",
    "Can I claim my mobile phone bill?",
    "What's the cents per kilometre method?",
    "Can I claim gym membership as a work expense?",
]


def main():
    print("Loading vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Loaded. {vectorstore._collection.count()} chunks in the store.\n")

    for question in TEST_QUESTIONS:
        print("=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = vectorstore.similarity_search_with_score(question, k=3)

        for rank, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"\n  [{rank}] score={score:.4f}  source={source}")
            print(f"      {preview}...")

        print()


if __name__ == "__main__":
    main()
