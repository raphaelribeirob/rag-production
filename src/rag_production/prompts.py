"""Generation prompts by version (`LLM_PROMPT`). v1 is the frozen 2026-10-22 baseline.

New versions are added only while iterating on the dev split (evals/dev_set.jsonl); every version
tried stays here so dev results can be reproduced.
"""

PROMPTS = {
    "v1": (
        "Answer the question using only the context below. If the context does not contain the "
        "answer, say you don't know. Answer in at most 3 plain sentences, no code.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
}
