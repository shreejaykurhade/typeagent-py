# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib.util
import json
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
load_search_queries = benchmark_embeddings.load_search_queries
parse_float_list = benchmark_embeddings.parse_float_list
parse_int_list = benchmark_embeddings.parse_int_list
score_answer_pair = benchmark_embeddings.score_answer_pair
select_best_answer_row = benchmark_embeddings.select_best_answer_row


def test_parse_float_list_default_and_custom() -> None:
    assert parse_float_list(None)
    assert parse_float_list("0.25, 0.7") == [0.25, 0.7]
    with pytest.raises(ValueError):
        parse_float_list("1.1")


def test_parse_int_list_validates_positive_values() -> None:
    assert parse_int_list("5,10") == [5, 10]


@pytest.mark.asyncio
async def test_score_answer_pair_exact_match() -> None:
    model = create_test_embedding_model()
    score = await score_answer_pair(model, ("Python", True), ("Python", True))
    assert score.answer_type_match is True
    assert score.exact_or_near_answer_match is True
    assert score.semantic_similarity == 1.0


@pytest.mark.asyncio
async def test_score_answer_pair_expected_answer_missing() -> None:
    model = create_test_embedding_model()
    score = await score_answer_pair(model, ("Python", True), ("No answer", False))
    assert score.answer_type_match is False
    assert score.semantic_similarity is None


@pytest.mark.asyncio
async def test_score_answer_pair_expected_no_answer_match() -> None:
    model = create_test_embedding_model()
    score = await score_answer_pair(
        model,
        ("No relevant info", False),
        ("Still none", False),
    )
    assert score.answer_type_match is True
    assert score.no_answer_match is True
    assert score.semantic_similarity is None


def test_load_search_queries_keeps_all_acceptable_matches(tmp_path: Path) -> None:
    data_dir = tmp_path / "tests" / "testdata"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "searchText": "who mentioned books",
            "results": [
                {"messageMatches": [1, 2]},
                {"messageMatches": [4]},
            ],
        }
    ]
    (data_dir / "Episode_53_Search_results.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    cases = load_search_queries(tmp_path)
    assert len(cases) == 1
    assert cases[0].expected_matches == [1, 2, 4]


def test_select_best_answer_row_prefers_true_eval_metrics() -> None:
    weaker = AnswerBenchmarkRow(
        min_score=0.25,
        max_hits=20,
        metrics=AnswerMetrics(
            answer_type_match_rate=82.0,
            exact_or_near_answer_rate=60.0,
            mean_semantic_similarity=0.88,
            no_answer_match_rate=50.0,
            classification_mismatch_rate=18.0,
            classification_mismatch_count=6,
        ),
    )
    stronger = AnswerBenchmarkRow(
        min_score=0.7,
        max_hits=10,
        metrics=AnswerMetrics(
            answer_type_match_rate=91.0,
            exact_or_near_answer_rate=75.0,
            mean_semantic_similarity=0.93,
            no_answer_match_rate=75.0,
            classification_mismatch_rate=9.0,
            classification_mismatch_count=2,
        ),
    )

    best = select_best_answer_row([weaker, stronger])
    assert best is stronger
