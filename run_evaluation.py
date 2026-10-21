"""
Run the evaluation question set through the RAG pipeline and score results.

This measures:
  - Unanswered-question rate: for in-scope questions, how often did the
    system correctly answer (vs wrongly refuse)?
  - False-answer rate: for out-of-scope questions, how often did the
    system wrongly answer instead of refusing? (This is the more serious
    failure -- e.g. the superannuation hallucination we found manually.)

This script only checks WHETHER the system answered or refused, matched
against what we expect (expected_answerable column). It does NOT check
whether an answer's *content* is factually correct -- that still needs a
human (or a separate faithfulness check) to read the actual text. Treat
this as the effectiveness / unanswered-rate layer of your evaluation
framework, not the whole thing.

Usage:
    python run_evaluation.py
"""

import os
import csv
import sys
import time
from query import load_vectorstore, answer_question, GROQ_MODEL
from langchain_groq import ChatGroq
from groq import RateLimitError

INPUT_CSV = "evaluation_questions.csv"
OUTPUT_CSV = "evaluation_results.csv"

# Groq's free tier has a tokens-per-minute limit. A short pause between
# questions, plus automatic retry on a 429, keeps a full evaluation run
# from crashing partway through.
DELAY_BETWEEN_QUESTIONS = 3  # seconds
MAX_RETRIES = 5

# If the relevance threshold in query.py lets a weak/borderline match
# through to the LLM, the LLM itself may still correctly refuse in its
# own words (as designed -- see SYSTEM_PROMPT rule 2). This script needs
# to catch that refusal-by-wording case too, not just the was_answered
# flag from the retrieval gate, or it will wrongly score a correct
# refusal as a hallucinated "answer".
REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "don't have enough relevant information",
    "do not have enough relevant information",
    "i'm not able to answer",
    "i am not able to answer",
    "cannot answer that confidently",
    "can't answer that confidently",
    "outside that scope",
    "outside my knowledge base",
]


def is_refusal_text(answer_text):
    # Normalize curly/smart apostrophes to straight ones before matching --
    # LLM output (and copy-pasted text generally) often uses U+2019 (') or
    # U+2018 instead of a plain ASCII apostrophe, which would otherwise
    # silently fail an exact substring match like "don't" vs "don't".
    normalized = answer_text.replace("\u2019", "'").replace("\u2018", "'").lower()
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def answer_with_retry(vectorstore, llm, question):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return answer_question(vectorstore, llm, question)
        except RateLimitError as e:
            wait = 15 * attempt  # back off a bit more each retry
            print(f"    Rate limited -- waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
    raise RuntimeError(f"Gave up after {MAX_RETRIES} retries due to repeated rate limiting.")


def main():
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Error: set the GROQ_API_KEY environment variable (get a free key at console.groq.com/keys)")

    print("Loading vector store and model...")
    vectorstore = load_vectorstore()
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, max_tokens=700)
    print(f"Using Groq model: {GROQ_MODEL}")
    print("Ready.\n")

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    results = []

    for idx, row in enumerate(questions):
        qid = row["question_id"]
        question = row["question"]
        expected = row["expected_answerable"]  # yes / no / partial

        print(f"[{qid}] {question}")
        answer, retrieved, was_answered = answer_with_retry(vectorstore, llm, question)

        # Two layers can cause a "no answer": the retrieval threshold gate
        # (was_answered=False), or the LLM refusing in its own words even
        # though retrieval let it through. Treat both as "no" for scoring.
        actual = "no" if (not was_answered or is_refusal_text(answer)) else "yes"

        # "partial" expected outcomes are judged manually -- don't auto-score them
        if expected == "partial":
            verdict = "MANUAL_REVIEW"
        else:
            verdict = "PASS" if actual == expected else "FAIL"

        best_score = min((s for _, s in retrieved), default=None)

        if not was_answered:
            refusal_layer = "retrieval_threshold"
        elif is_refusal_text(answer):
            refusal_layer = "llm_self_refusal"
        else:
            refusal_layer = "none"

        print(f"    expected={expected}  actual={actual}  -> {verdict}")

        results.append({
            "question_id": qid,
            "question": question,
            "category": row["category"],
            "expected_answerable": expected,
            "actual_answered": actual,
            "verdict": verdict,
            "refusal_layer": refusal_layer,
            "best_retrieval_score": f"{best_score:.4f}" if best_score is not None else "",
            "answer_text": answer.replace("\n", " "),
        })

        # Save progress after every question -- if it crashes partway through
        # (rate limit, network blip, etc.), you don't lose what's already run.
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_id", "question", "category", "expected_answerable",
                "actual_answered", "verdict", "refusal_layer", "best_retrieval_score", "answer_text",
            ])
            writer.writeheader()
            writer.writerows(results)

        if idx < len(questions) - 1:
            time.sleep(DELAY_BETWEEN_QUESTIONS)

    scored = [r for r in results if r["verdict"] in ("PASS", "FAIL")]
    n_scored = len(scored)
    n_correct = sum(1 for r in scored if r["verdict"] == "PASS")

    print("\n" + "=" * 60)
    print(f"Scored questions: {n_scored} (excludes 'partial' -- those need manual review)")
    if n_scored:
        print(f"Correct: {n_correct}/{n_scored}  ({100 * n_correct / n_scored:.1f}%)")
    fails = [r for r in scored if r["verdict"] == "FAIL"]
    if fails:
        print(f"\nFAILED ({len(fails)}):")
        for r in fails:
            print(f"  [{r['question_id']}] {r['question']}  "
                  f"(expected={r['expected_answerable']}, got={r['actual_answered']})")
    manual = [r for r in results if r["verdict"] == "MANUAL_REVIEW"]
    if manual:
        print(f"\nNEEDS MANUAL REVIEW ({len(manual)}):")
        for r in manual:
            print(f"  [{r['question_id']}] {r['question']}")
    print(f"\nFull results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()