import pytest

from llm_judge import answer_relevancy, faithfulness
from tests.conftest import FakeEmbedder, FakeLLM


def test_faithfulness_scores_fraction_of_supported_claims():
    reply = (
        '[{"claim": "Revenue was Rs 100 crore", "supported": true}, '
        '{"claim": "Profit doubled", "supported": false}]'
    )
    llm = FakeLLM(reply=reply)
    answer = "Revenue was Rs 100 crore and profit doubled."
    score = faithfulness("Revenue was Rs 100 crore.", answer, "fake-model", llm_call=llm)
    assert score == 0.5
    assert len(llm.calls) == 1


def test_faithfulness_all_supported_is_one():
    reply = '[{"claim": "A", "supported": true}, {"claim": "B", "supported": true}]'
    score = faithfulness("context", "answer", "fake-model", llm_call=FakeLLM(reply=reply))
    assert score == 1.0


def test_faithfulness_returns_none_when_judge_output_unparseable():
    llm = FakeLLM(reply="not json at all")
    score = faithfulness("context", "answer", "fake-model", llm_call=llm)
    assert score is None


def test_faithfulness_handles_prose_wrapped_around_json():
    reply = 'Sure, here you go:\n[{"claim": "X", "supported": true}]\nHope that helps!'
    score = faithfulness("context", "answer", "fake-model", llm_call=FakeLLM(reply=reply))
    assert score == 1.0


def test_answer_relevancy_returns_none_when_judge_output_unparseable():
    score = answer_relevancy(
        "What was revenue?",
        "Revenue grew.",
        "fake-model",
        FakeEmbedder(),
        llm_call=FakeLLM(reply="nope"),
    )
    assert score is None


def test_answer_relevancy_averages_similarity_over_generated_questions():
    reply = '["What was revenue?", "What was revenue?", "What was revenue?"]'
    llm = FakeLLM(reply=reply)
    score = answer_relevancy(
        "What was revenue?", "Revenue grew.", "fake-model", FakeEmbedder(), n=3, llm_call=llm
    )
    # Same question embedded against itself (FakeEmbedder is deterministic and
    # L2-normalised) -> similarity 1.0
    assert score == pytest.approx(1.0)
    assert len(llm.calls) == 1


def test_answer_relevancy_ignores_non_string_entries():
    reply = '["What was revenue?", 42, null]'
    llm = FakeLLM(reply=reply)
    score = answer_relevancy(
        "What was revenue?", "Revenue grew.", "fake-model", FakeEmbedder(), llm_call=llm
    )
    assert score is not None  # the one valid string entry is still used
