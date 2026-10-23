"""Failure taxonomy: rules are fixed here (and in evals/README) before they touch test answers."""

from rag_production.taxonomy import (
    agreement,
    classify_answer,
    classify_sentence,
    is_refusal,
    lexical_overlap,
    novel_code_tokens,
    summarize,
)

CTX = [
    "Use `Header` to read headers. Header converts `user_agent` to user-agent automatically.",
    "Decimal values are handled the same as a float in requests and responses.",
]


class FakeNLI:
    """(entailment, contradiction): entail if verbatim in premise; contradict if 'never' in it."""

    def scores(self, pairs):
        out = []
        for p, h in pairs:
            ent = 1.0 if h.lower().rstrip(".") in p.lower() else 0.0
            out.append((ent, 0.9 if "never" in h.lower() else 0.0))
        return out


def test_refusal_patterns():
    assert is_refusal("The context does not provide details on this.")
    assert is_refusal("The specific format isn't detailed in the given context.")
    assert is_refusal("I don't know.")
    assert is_refusal("This is not mentioned in the provided documentation.")
    assert not is_refusal("Use the Header class to read the User-Agent header.")
    assert not is_refusal("It does not return the password.")


def test_novel_code_tokens_must_appear_verbatim_in_context():
    hal = "Use `header.user_agent` in the handler."
    assert novel_code_tokens(hal, CTX) == ["header.user_agent"]
    assert novel_code_tokens("Declare `user_agent` with `Header`.", CTX) == []
    assert novel_code_tokens("Call request_headers.get_value() here.", CTX) == [
        "request_headers.get_value"
    ]


def test_lexical_overlap_is_content_token_recall_against_best_chunk():
    assert lexical_overlap("Decimal values are handled like a float.", CTX) == 1.0
    assert lexical_overlap("Kubernetes pods scale horizontally.", CTX) == 0.0


def test_sentence_rules_in_order():
    s = classify_sentence
    assert s(0.9, 0.0, 0.0, [], False, True) == "supported"
    assert s(0.1, 0.0, 1.0, [], True, True) == "evasive"
    assert s(0.1, 0.0, 1.0, [], True, False) == "justified_refusal"
    assert s(0.1, 0.6, 1.0, [], False, True) == "hallucination"  # contradiction
    assert s(0.1, 0.0, 1.0, ["x.y"], False, True) == "hallucination"  # invented identifier
    assert s(0.1, 0.0, 0.75, [], False, True) == "paraphrase"
    assert s(0.1, 0.0, 0.74, [], False, True) == "hallucination"


def test_answer_label_priority_and_score():
    judge = FakeNLI()
    ok = classify_answer("Decimal values are handled the same as a float.", CTX, True, judge)
    assert ok["label"] == "supported" and ok["score"] == 1.0
    para = classify_answer("Decimal values are handled like a float.", CTX, True, judge)
    assert para["label"] == "paraphrase" and para["score"] == 0.0
    ev = classify_answer(
        "Decimal values are handled the same as a float. The context does not say more.",
        CTX,
        True,
        judge,
    )
    assert ev["label"] == "evasive" and ev["score"] == 0.5
    hal = classify_answer(
        "The context does not say. Use `header.user_agent` in the handler.", CTX, True, judge
    )
    assert hal["label"] == "hallucination"
    assert classify_answer("```py\nx=1\n```", CTX, True, judge)["label"] == "unscored"


def test_summarize_rates_over_scored_answers():
    r = summarize(["supported", "supported", "hallucination", "evasive", "unscored"])
    assert r["n"] == 4 and r["unscored"] == 1
    assert r["rates"] == {
        "supported": 0.5,
        "hallucination": 0.25,
        "evasive": 0.25,
        "paraphrase": 0.0,
        "justified_refusal": 0.0,
    }


def test_agreement_with_hand_labels():
    hand = {"a": "supported", "b": "hallucination", "c": "paraphrase"}
    auto = {"a": "supported", "b": "paraphrase", "c": "paraphrase", "z": "evasive"}
    r = agreement(hand, auto)
    assert r["n"] == 3 and r["agree"] == 2 and r["rate"] == 0.6667
    assert r["confusion"] == {"hallucination->paraphrase": 1}
