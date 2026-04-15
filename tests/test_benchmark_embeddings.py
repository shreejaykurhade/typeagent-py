# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib.util
from pathlib import Path

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "benchmark_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_embeddings", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
benchmark_embeddings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_embeddings)

AnswerBenchmarkRow = benchmark_embeddings.AnswerBenchmarkRow
AnswerMetrics = benchmark_embeddings.AnswerMetrics
parse_float_list = benchmark_embeddings.parse_float_list
parse_int_list = benchmark_embeddings.parse_int_list
score_answer_pair = benchmark_embeddings.score_answer_pair
select_best_answer_row = benchmark_embeddings.select_best_answer_row


def test_parse_float_list_default_and_custom() -> None:
    assert parse_float_list(None)
    assert parse_float_list("0.25, 0.7") == [0.25, 0.7]


def test_parse_int_list_validates_positive_values() -> None:
    assert parse_int_list("5,10") == [5, 10]


@pytest.mark.asyncio
async def test_score_answer_pair_exact_match() -> None:
    model = create_test_embedding_model()
    score = await score_answer_pair(model, ("Python", True), ("Python", True))
    assert score == 1.0


@pytest.mark.asyncio
async def test_score_answer_pair_expected_answer_missing() -> None:
    model = create_test_embedding_model()
    score = await score_answer_pair(model, ("Python", True), ("No answer", False))
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_answer_pair_expected_no_answer_match() -> None:
    model = create_test_embedding_model()
    score = await score_answer_pair(
        model,
        ("No relevant info", False),
        ("Still none", False),
    )
    assert score == 1.001


def test_select_best_answer_row_prefers_true_eval_metrics() -> None:
    weaker = AnswerBenchmarkRow(
        min_score=0.25,
        max_hits=20,
        metrics=AnswerMetrics(
            mean_score=0.82,
            exact_or_near_rate=60.0,
            zero_score_rate=12.0,
            zero_score_count=6,
        ),
    )
    stronger = AnswerBenchmarkRow(
        min_score=0.7,
        max_hits=10,
        metrics=AnswerMetrics(
            mean_score=0.91,
            exact_or_near_rate=75.0,
            zero_score_rate=4.0,
            zero_score_count=2,
        ),
    )

    best = select_best_answer_row([weaker, stronger])
    assert best is stronger
