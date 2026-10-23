import pytest

from rag_production.faithfulness import faithfulness_score, split_sentences


class FakeNLI:
    """Entailment prob 1.0 if the hypothesis appears verbatim in the premise, else 0.0."""

    def entailment(self, pairs):
        return [1.0 if h.lower().rstrip(".") in p.lower() else 0.0 for p, h in pairs]


def test_split_sentences_skips_code_headings_and_fragments():
    text = (
        "Path Params > Types\n\nUse type hints. They give editor support.\n\n"
        "```python\nx = 1\n```\n\nOk."
    )
    assert split_sentences(text) == ["Use type hints.", "They give editor support."]


def test_score_is_share_of_supported_sentences():
    ctx = ["FastAPI uses type hints. It validates data."]
    answer = "FastAPI uses type hints. It runs on the moon."
    r = faithfulness_score(answer, ctx, FakeNLI(), threshold=0.5)
    assert r == {"sentences": 2, "supported": 1, "score": 0.5}


def test_sentence_supported_by_any_context():
    r = faithfulness_score("It validates data.", ["nope", "It validates data."], FakeNLI())
    assert r["score"] == 1.0


def test_empty_answer_has_no_score():
    assert faithfulness_score("", ["ctx"], FakeNLI())["score"] is None


def test_real_nli_judge_entailment_vs_contradiction():
    pytest.importorskip("sentence_transformers")
    from rag_production.faithfulness import NLIJudge

    judge = NLIJudge("cross-encoder/nli-deberta-v3-small")
    premise = "FastAPI declares path parameters using Python type annotations."
    ent, con = judge.entailment(
        [
            (premise, "Path parameters are declared with type annotations."),
            (premise, "FastAPI does not support type annotations."),
        ]
    )
    assert ent > 0.5 > con


def test_evaluate_faithfulness_mean_and_negative_control():
    from rag_production.faithfulness import evaluate_faithfulness

    items = [
        {"answer": "FastAPI uses type hints.", "contexts": ["FastAPI uses type hints."]},
        {"answer": "It validates request data.", "contexts": ["It validates request data."]},
    ]
    r = evaluate_faithfulness(items, FakeNLI(), threshold=0.5)
    assert r["n"] == 2 and r["faithfulness"] == 1.0
    # negative control: each answer judged against another question's contexts
    assert r["negative_control"] == 0.0


class ExactNLI:
    """Like a weak NLI model: entails only when the premise is exactly the hypothesis."""

    def entailment(self, pairs):
        return [1.0 if p.strip() == h.strip() else 0.0 for p, h in pairs]


def test_premise_is_split_into_sentences_summac_style():
    ctx = ["Chunk heading\n\nFastAPI uses type hints. It validates data. Short."]
    r = faithfulness_score("It validates data.", ctx, ExactNLI())
    assert r["score"] == 1.0


def test_real_nli_judge_scores_entailment_and_contradiction():
    pytest.importorskip("sentence_transformers")
    from rag_production.faithfulness import NLIJudge

    judge = NLIJudge()
    (ent, con), (ent2, con2) = judge.scores(
        [
            ("The parameter is required.", "The parameter is required."),
            ("The parameter is required.", "The parameter is never required."),
        ]
    )
    assert ent > 0.5 > con and con2 > 0.5 > ent2
    assert judge.entailment([("a b c", "a b c")])[0] == judge.scores([("a b c", "a b c")])[0][0]
