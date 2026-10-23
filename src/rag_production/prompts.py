"""Generation prompts by version (`LLM_PROMPT`). v1 is the frozen 2026-10-22 baseline.

New versions are added only while iterating on the dev split (evals/dev_set.jsonl); every version
tried stays here so dev results can be reproduced. Templates may use `{context}` (passages
joined by ---) or `{numbered_context}` ([1] ..., [2] ...) so answers can cite passages.
"""

from collections.abc import Sequence

PROMPTS = {
    "v1": (
        "Answer the question using only the context below. If the context does not contain the "
        "answer, say you don't know. Answer in at most 3 plain sentences, no code.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    "v2": (
        "Answer the question about FastAPI using ONLY the numbered context passages below.\n"
        "Rules:\n"
        "1. Reuse the passages' own wording where possible, and end every sentence with the "
        "number of the passage it comes from, like [2].\n"
        "2. Do not add facts, code, names, values or steps that are not in the passages, even if "
        "you know them.\n"
        "3. If the passages answer only part of the question, answer that part and stop; do not "
        "guess the rest.\n"
        '4. Reply "I don\'t know." only if no passage is relevant to the question.\n'
        "Answer in at most 3 sentences, no code blocks.\n\n"
        "Context:\n{numbered_context}\n\nQuestion: {question}\nAnswer:"
    ),
    "v3": (
        "Answer the question about FastAPI using ONLY the numbered context passages below.\n"
        "Rules:\n"
        "1. Reuse the passages' own wording where possible. Put the number of the passage each "
        "sentence comes from in brackets just BEFORE the sentence's final period, like this [2].\n"
        "2. Do not add facts, code, names, values or steps that are not in the passages, even if "
        "you know them.\n"
        "3. If the passages answer only part of the question, answer that part and stop; do not "
        "guess the rest.\n"
        '4. Reply "I don\'t know." only if no passage is relevant to the question.\n'
        "Answer in at most 3 plain sentences. Never write code or code blocks.\n\n"
        "Context:\n{numbered_context}\n\nQuestion: {question}\nAnswer:"
    ),
}


def render(template: str, question: str, contexts: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    return template.format(
        context="\n\n---\n\n".join(contexts), numbered_context=numbered, question=question
    )
