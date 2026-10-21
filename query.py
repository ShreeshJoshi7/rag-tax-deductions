"""
RAG query pipeline: retrieve relevant ATO chunks, then generate a grounded,
cited answer using a local Ollama model. Supports follow-up questions
within a single interactive session (e.g. "what about if I lease it
instead?" after asking about car expenses).

Usage:
    python query.py "What car expenses can I claim?"   # single-shot, no memory

Or run with no argument for an interactive loop with memory:
    python query.py
"""

import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "db/chroma_db"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 3  # how many chunks to retrieve per question (reduced from 4 for speed)

# Cosine DISTANCE threshold (lower = more similar). If the best-matching
# chunk's score is above this, the question is treated as out of scope.
MAX_RELEVANCE_DISTANCE = 0.50

# How many previous turns to keep in memory / feed back into prompts.
MAX_HISTORY_TURNS = 4

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Australian \
individual income tax deductions, using ONLY the context provided below. The context \
comes from official Australian Taxation Office (ATO) guidance.

Rules you must follow:
1. Answer using only the information in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say so \
clearly instead of guessing -- respond with something like "I don't have enough \
information in my knowledge base to answer that confidently."
3. Where relevant, mention specific conditions, thresholds, or exceptions from the \
context rather than giving an overly general answer.
4. Keep the answer clear and in plain English -- the user may not know tax terminology.
5. Do not give personal financial or legal advice framed as certain; present it as \
general information based on the ATO guidance provided.
6. If earlier turns of the conversation are provided, use them only to understand \
what the user is referring to (e.g. "it" or "that"). Still answer only from the \
context given for THIS turn -- do not pull facts from earlier turns' context.
7. Answer directly and concisely. Do not show step-by-step reasoning or a "thinking" \
section -- just give the final answer.
8. Aim for a focused answer of around 120-180 words. Cover the key conditions and any important exceptions, using short paragraphs or a brief bullet list -- not one long multi-section essay, but not a single sentence either.
9. This assistant's scope is specifically work-related employee deductions (car and
transport, tools/equipment, clothing, working from home, education/training, and
similar). If the context mentions a deduction category outside this scope (e.g. gifts
and donations, investment/super/insurance deductions, cost of managing tax affairs),
do not elaborate on it. Briefly note that it is outside what this assistant covers in
depth, and focus on the work-related categories.
/no_think"""

NO_RELEVANT_CONTEXT_MESSAGE = (
    "I don't have enough relevant information in my knowledge base to answer that "
    "confidently. This assistant currently only covers work-related tax deductions "
    "(car expenses, tools/equipment, clothing, working from home, education, and "
    "similar). Your question may be outside that scope, or about a different area "
    "of tax law."
)

REWRITE_PROMPT = """Given the conversation so far and a new follow-up question, \
rewrite the follow-up as a standalone question that makes sense without needing the \
earlier turns. Keep it short and preserve the user's intent. If the follow-up is \
already a standalone question, just return it unchanged.

Conversation so far:
{history}

Follow-up question: {question}

Standalone question:"""


def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def format_history(history):
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def rewrite_followup(llm, question, history):
    """Turn a follow-up like 'what about leasing instead?' into a standalone
    question for retrieval, using recent chat history for context."""
    if not history:
        return question

    prompt = REWRITE_PROMPT.format(history=format_history(history), question=question)
    response = llm.invoke([("human", prompt)])
    rewritten = response.content.strip().strip('"')
    return rewritten if rewritten else question


def retrieve_context(vectorstore, question, k=TOP_K):
    return vectorstore.similarity_search_with_score(question, k=k)


def is_relevant(results):
    if not results:
        return False
    best_score = min(score for _, score in results)
    return best_score <= MAX_RELEVANCE_DISTANCE


def build_prompt(question, results, history):
    context_blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[Source {i}: {source}]\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    history_block = ""
    if history:
        history_block = f"Earlier in this conversation:\n{format_history(history)}\n\n"

    user_prompt = f"""{history_block}Context for this question:
{context_text}

Question: {question}

Answer the question using only the context above. At the end of your answer, list \
which source number(s) you used, like: "Sources used: 1, 3"."""

    return user_prompt


def answer_question(vectorstore, llm, question, history=None):
    """history: list of {"question": str, "answer": str} dicts, oldest first.
    Pass None or [] for a single-shot question with no memory."""
    history = history or []

    # Rewrite follow-ups into standalone questions before retrieval, so
    # retrieval actually finds the right chunks for things like "what about
    # leasing instead?" rather than embedding that fragment on its own.
    retrieval_question = rewrite_followup(llm, question, history) if history else question

    results = retrieve_context(vectorstore, retrieval_question)

    if not is_relevant(results):
        return NO_RELEVANT_CONTEXT_MESSAGE, results, False

    user_prompt = build_prompt(question, results, history)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)

    return response.content, results, True


def print_sources(results, was_answered):
    label = "Retrieved sources" if was_answered else "Retrieved sources (below relevance threshold -- not used)"
    print(f"\n--- {label} ---")
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        flag = "" if score <= MAX_RELEVANCE_DISTANCE else "  <- above threshold"
        print(f"  [{i}] score={score:.4f}  {source}{flag}")


def main():
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Error: set the GROQ_API_KEY environment variable (get a free key at console.groq.com/keys)")

    print("Loading vector store...")
    vectorstore = load_vectorstore()
    print(f"Loaded. {vectorstore._collection.count()} chunks available.\n")

    print(f"Connecting to Groq model: {GROQ_MODEL}...")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, max_tokens=700)
    print("Ready.\n")

    if len(sys.argv) > 1:
        # Single question passed as command-line argument -- no memory, one-shot.
        question = " ".join(sys.argv[1:])
        print(f"QUESTION: {question}\n")
        answer, results, was_answered = answer_question(vectorstore, llm, question)
        print("ANSWER:")
        print(answer)
        print_sources(results, was_answered)
        return

    # Interactive loop with memory across turns in this session.
    print("Interactive mode (with follow-up memory). Type your question, or 'quit' to exit.")
    print("Type 'reset' to clear conversation memory and start fresh.\n")

    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "reset":
            history = []
            print("(conversation memory cleared)\n")
            continue
        if not question:
            continue

        answer, results, was_answered = answer_question(vectorstore, llm, question, history)
        print(f"\nAssistant: {answer}")
        print_sources(results, was_answered)
        print()

        # Only keep answered turns in memory -- a refusal doesn't give the
        # model anything useful to refer back to later.
        if was_answered:
            history.append({"question": question, "answer": answer})
            history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    main()