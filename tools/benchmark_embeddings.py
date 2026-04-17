# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark embedding settings on retrieval-only or true end-to-end evals.

This script evaluates combinations of `min_score` and `max_hits` for the
Episode 53 dataset in `tests/testdata/`.

Two benchmark modes are supported:
- `answer` (default): run the full slow eval path used by `make eval`
- `retrieval`: run the narrower `messageMatches` retrieval benchmark

The answer mode is the one to use when choosing settings for better final
answers. The retrieval mode is still useful for quick diagnostics, but it does
not prove that a row is best for end-to-end answer quality.

Usage:
    uv run python tools/benchmark_embeddings.py
    uv run python tools/benchmark_embeddings.py --mode retrieval
    uv run python tools/benchmark_embeddings.py --model openai:text-embedding-3-small
"""

import argparse
import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
from statistics import mean
import time
from typing import Literal

from dotenv import load_dotenv

import typechat

from typeagent.aitools import model_adapters, utils
from typeagent.aitools.embeddings import IEmbeddingModel, NormalizedEmbeddings
from typeagent.aitools.model_adapters import create_embedding_model
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase
from typeagent.knowpro import (
    answer_response_schema,
    answers,
    search_query_schema,
    searchlang,
    secindex,
)
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.podcasts.podcast import Podcast
from typeagent.storage.memory.convthreads import ConversationThreads
from typeagent.storage.utils import create_storage_provider

DEFAULT_MIN_SCORES = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]
DEFAULT_MAX_HITS = [5, 10, 15, 20]
DATA_DIR = Path("tests") / "testdata"
INDEX_DATA_PATH = DATA_DIR / "Episode_53_AdrianTchaikovsky_index_data.json"
INDEX_PREFIX_PATH = DATA_DIR / "Episode_53_AdrianTchaikovsky_index"
SEARCH_RESULTS_PATH = DATA_DIR / "Episode_53_Search_results.json"
ANSWER_RESULTS_PATH = DATA_DIR / "Episode_53_Answer_results.json"
DEFAULT_SEARCH_OPTIONS = searchlang.LanguageSearchOptions(
    compile_options=searchlang.LanguageQueryCompileOptions(
        exact_scope=False,
        verb_scope=True,
        term_filter=None,
        apply_scope=True,
    ),
    exact_match=False,
    max_message_matches=25,
)
DEFAULT_ANSWER_OPTIONS = answers.AnswerContextOptions(
    entities_top_k=50,
    topics_top_k=50,
    messages_top_k=None,
    chunking=None,
)
type BenchmarkMode = Literal["answer", "retrieval"]


@dataclass
class SearchQueryCase:
    query: str
    expected_matches: list[int]


@dataclass
class AnswerQueryCase:
    question: str
    expected_answer: str
    expected_success: bool


@dataclass
class SearchMetrics:
    hit_rate: float
    mean_reciprocal_rank: float


@dataclass
class AnswerMetrics:
    answer_type_match_rate: float
    exact_or_near_answer_rate: float
    mean_semantic_similarity: float
    no_answer_match_rate: float
    classification_mismatch_rate: float
    classification_mismatch_count: int


@dataclass
class AnswerScore:
    answer_type_match: bool
    no_answer_match: bool
    exact_or_near_answer_match: bool
    semantic_similarity: float | None


@dataclass
class RetrievalBenchmarkRow:
    min_score: float
    max_hits: int
    metrics: SearchMetrics


@dataclass
class AnswerBenchmarkRow:
    min_score: float
    max_hits: int
    metrics: AnswerMetrics


@dataclass
class TrueEvalContext:
    conversation: Podcast
    embedding_model: IEmbeddingModel
    query_translator: typechat.TypeChatJsonTranslator[search_query_schema.SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[
        answer_response_schema.AnswerResponse
    ]
    settings: ConversationSettings


def parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return DEFAULT_MIN_SCORES
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--min-scores must contain at least one value")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--min-scores values must be between 0.0 and 1.0")
    return values


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None:
        return DEFAULT_MAX_HITS
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--max-hits must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("--max-hits values must be positive integers")
    return values


def load_message_texts(repo_root: Path) -> list[str]:
    index_data = json.loads((repo_root / INDEX_DATA_PATH).read_text(encoding="utf-8"))
    messages = index_data["messages"]
    return [" ".join(message.get("textChunks", [])) for message in messages]


def load_search_queries(repo_root: Path) -> list[SearchQueryCase]:
    search_data = json.loads(
        (repo_root / SEARCH_RESULTS_PATH).read_text(encoding="utf-8")
    )
    cases: list[SearchQueryCase] = []
    for item in search_data:
        search_text = item.get("searchText")
        results = item.get("results", [])
        if not search_text or not results:
            continue
        expected_matches = sorted(
            {
                message_ordinal
                for result in results
                for message_ordinal in result.get("messageMatches", [])
            }
        )
        if not expected_matches:
            continue
        cases.append(SearchQueryCase(search_text, expected_matches))
    return cases


def load_answer_queries(repo_root: Path) -> list[AnswerQueryCase]:
    answer_data = json.loads(
        (repo_root / ANSWER_RESULTS_PATH).read_text(encoding="utf-8")
    )
    cases: list[AnswerQueryCase] = []
    for item in answer_data:
        question = item.get("question")
        answer = item.get("answer")
        has_no_answer = item.get("hasNoAnswer")
        if question is None or answer is None or has_no_answer is None:
            continue
        cases.append(
            AnswerQueryCase(
                question=question,
                expected_answer=answer,
                expected_success=not has_no_answer,
            )
        )
    return cases


async def build_vector_base(
    model_spec: str | None,
    message_texts: list[str],
    batch_size: int,
) -> tuple[IEmbeddingModel, VectorBase]:
    model = create_embedding_model(model_spec)
    settings = TextEmbeddingIndexSettings(
        embedding_model=model,
        min_score=0.0,
        max_matches=None,
        batch_size=batch_size,
    )
    vector_base = VectorBase(settings)

    for start in range(0, len(message_texts), batch_size):
        batch = message_texts[start : start + batch_size]
        await vector_base.add_keys(batch)

    return model, vector_base


def evaluate_search_queries(
    vector_base: VectorBase,
    query_cases: list[SearchQueryCase],
    query_embeddings: NormalizedEmbeddings,
    min_score: float,
    max_hits: int,
) -> SearchMetrics:
    hit_count = 0
    reciprocal_ranks: list[float] = []

    for case, query_embedding in zip(query_cases, query_embeddings):
        scored_results = vector_base.fuzzy_lookup_embedding(
            query_embedding,
            max_hits=max_hits,
            min_score=min_score,
        )
        rank = 0
        for result_index, scored_result in enumerate(scored_results, start=1):
            if scored_result.item in case.expected_matches:
                rank = result_index
                break
        if rank > 0:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return SearchMetrics(
        hit_rate=(hit_count / len(query_cases)) * 100,
        mean_reciprocal_rank=mean(reciprocal_ranks),
    )


async def create_true_eval_context(
    repo_root: Path,
    model_spec: str | None,
) -> TrueEvalContext:
    embedding_model = create_embedding_model(model_spec)
    settings = ConversationSettings(model=embedding_model)
    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        message_type=None,
    )

    raw_data = Podcast._read_conversation_data_from_file(
        str(repo_root / INDEX_PREFIX_PATH)
    )
    raw_data.pop("messageIndexData", None)
    raw_data.pop("relatedTermsIndexData", None)

    conversation = await Podcast.create(settings)
    await conversation.deserialize(raw_data)
    await secindex.build_secondary_indexes(conversation, settings)

    threads = (
        conversation.secondary_indexes.threads
        if conversation.secondary_indexes is not None
        else None
    )
    if isinstance(threads, ConversationThreads) and threads.threads:
        await threads.build_index()

    chat_model = model_adapters.create_chat_model()
    query_translator = utils.create_translator(
        chat_model, search_query_schema.SearchQuery
    )
    answer_translator = utils.create_translator(
        chat_model,
        answer_response_schema.AnswerResponse,
    )

    return TrueEvalContext(
        conversation=conversation,
        embedding_model=embedding_model,
        query_translator=query_translator,
        answer_translator=answer_translator,
        settings=settings,
    )


def answer_response_to_eval_tuple(
    response: answer_response_schema.AnswerResponse,
) -> tuple[str, bool]:
    match response.type:
        case "Answered":
            return response.answer or "", True
        case "NoAnswer":
            return response.why_no_answer or "", False
        case _:
            raise ValueError(f"Unexpected answer type: {response.type}")


async def score_answer_pair(
    embedding_model: IEmbeddingModel,
    expected: tuple[str, bool],
    actual: tuple[str, bool],
) -> AnswerScore:
    expected_text, expected_success = expected
    actual_text, actual_success = actual

    if expected_success != actual_success:
        return AnswerScore(
            answer_type_match=False,
            no_answer_match=False,
            exact_or_near_answer_match=False,
            semantic_similarity=None,
        )
    if not actual_success:
        return AnswerScore(
            answer_type_match=True,
            no_answer_match=True,
            exact_or_near_answer_match=False,
            semantic_similarity=None,
        )
    if expected_text == actual_text:
        return AnswerScore(
            answer_type_match=True,
            no_answer_match=False,
            exact_or_near_answer_match=True,
            semantic_similarity=1.0,
        )
    if expected_text.lower() == actual_text.lower():
        return AnswerScore(
            answer_type_match=True,
            no_answer_match=False,
            exact_or_near_answer_match=True,
            semantic_similarity=0.999,
        )

    embeddings = await embedding_model.get_embeddings([expected_text, actual_text])
    assert embeddings.shape[0] == 2, "Expected two embeddings"
    semantic_similarity = float(embeddings[0] @ embeddings[1])
    return AnswerScore(
        answer_type_match=True,
        no_answer_match=False,
        exact_or_near_answer_match=semantic_similarity >= 0.97,
        semantic_similarity=semantic_similarity,
    )


async def evaluate_answer_queries(
    context: TrueEvalContext,
    query_cases: list[AnswerQueryCase],
    min_score: float,
    max_hits: int,
) -> AnswerMetrics:
    search_options = replace(
        DEFAULT_SEARCH_OPTIONS,
        max_message_matches=max_hits,
        threshold_score=min_score,
    )

    answer_scores: list[AnswerScore] = []
    total = len(query_cases)
    started_at = time.perf_counter()
    for index, case in enumerate(query_cases, start=1):
        if index == 1 or index % 5 == 0 or index == total:
            elapsed = time.perf_counter() - started_at
            print(
                f"    Question {index}/{total} "
                f"(elapsed {elapsed:.1f}s): {case.question}",
                flush=True,
            )
        result = await searchlang.search_conversation_with_language(
            context.conversation,
            context.query_translator,
            case.question,
            search_options,
        )
        if isinstance(result, typechat.Failure):
            actual = (f"Search failed: {result.message}", False)
        else:
            _, combined_answer = await answers.generate_answers(
                context.answer_translator,
                result.value,
                context.conversation,
                case.question,
                options=DEFAULT_ANSWER_OPTIONS,
            )
            actual = answer_response_to_eval_tuple(combined_answer)

        expected = (case.expected_answer, case.expected_success)
        answer_scores.append(
            await score_answer_pair(context.embedding_model, expected, actual)
        )

    answer_type_match_count = sum(
        1 for score in answer_scores if score.answer_type_match
    )
    no_answer_match_count = sum(1 for score in answer_scores if score.no_answer_match)
    exact_or_near_answer_count = sum(
        1 for score in answer_scores if score.exact_or_near_answer_match
    )
    semantic_similarities = [
        score.semantic_similarity
        for score in answer_scores
        if score.semantic_similarity is not None
    ]
    classification_mismatch_count = total - answer_type_match_count

    return AnswerMetrics(
        answer_type_match_rate=(answer_type_match_count / total) * 100,
        exact_or_near_answer_rate=(exact_or_near_answer_count / total) * 100,
        mean_semantic_similarity=(
            mean(semantic_similarities) if semantic_similarities else 0.0
        ),
        no_answer_match_rate=(no_answer_match_count / total) * 100,
        classification_mismatch_rate=(classification_mismatch_count / total) * 100,
        classification_mismatch_count=classification_mismatch_count,
    )


def select_best_retrieval_row(
    rows: list[RetrievalBenchmarkRow],
) -> RetrievalBenchmarkRow:
    # Prefer stricter thresholds and smaller hit limits only after primary
    # retrieval metrics tie.
    return max(
        rows,
        key=lambda row: (
            row.metrics.mean_reciprocal_rank,
            row.metrics.hit_rate,
            -row.min_score,
            -row.max_hits,
        ),
    )


def select_best_answer_row(rows: list[AnswerBenchmarkRow]) -> AnswerBenchmarkRow:
    # Prefer stricter thresholds and smaller hit limits only after the answer
    # quality metrics tie.
    return max(
        rows,
        key=lambda row: (
            row.metrics.answer_type_match_rate,
            row.metrics.exact_or_near_answer_rate,
            row.metrics.mean_semantic_similarity,
            row.metrics.no_answer_match_rate,
            -row.metrics.classification_mismatch_count,
            -row.min_score,
            -row.max_hits,
        ),
    )


def print_retrieval_rows(rows: list[RetrievalBenchmarkRow]) -> None:
    print("=" * 72)
    print("RETRIEVAL BENCHMARK (Episode 53 messageMatches ground truth)")
    print("=" * 72)
    print(f"{'Min Score':<12} | {'Max Hits':<10} | {'Hit Rate (%)':<15} | {'MRR':<10}")
    print("-" * 65)
    for row in rows:
        print(
            f"{row.min_score:<12.2f} | {row.max_hits:<10d} | "
            f"{row.metrics.hit_rate:<15.2f} | "
            f"{row.metrics.mean_reciprocal_rank:<10.4f}"
        )
    print("-" * 65)


def print_answer_rows(rows: list[AnswerBenchmarkRow]) -> None:
    print("=" * 138)
    print("TRUE EVAL BENCHMARK (Episode 53 full answer pipeline)")
    print("=" * 138)
    print(
        f"{'Min Score':<12} | {'Max Hits':<10} | {'Type Match (%)':<15} | "
        f"{'Exact/Near Ans (%)':<19} | {'Mean Similarity':<16} | "
        f"{'No-Answer (%)':<14} | {'Mismatch Count':<14} | {'Mismatch (%)':<12}"
    )
    print("-" * 138)
    for row in rows:
        print(
            f"{row.min_score:<12.2f} | {row.max_hits:<10d} | "
            f"{row.metrics.answer_type_match_rate:<15.2f} | "
            f"{row.metrics.exact_or_near_answer_rate:<19.2f} | "
            f"{row.metrics.mean_semantic_similarity:<16.4f} | "
            f"{row.metrics.no_answer_match_rate:<14.2f} | "
            f"{row.metrics.classification_mismatch_count:<14d} | "
            f"{row.metrics.classification_mismatch_rate:<12.2f}"
        )
    print("-" * 138)


async def run_retrieval_benchmark(
    repo_root: Path,
    model_spec: str | None,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
) -> None:
    message_texts = load_message_texts(repo_root)
    query_cases = load_search_queries(repo_root)
    if not query_cases:
        raise ValueError("No search queries with messageMatches found in the dataset")
    model, vector_base = await build_vector_base(model_spec, message_texts, batch_size)
    query_embeddings = await model.get_embeddings([case.query for case in query_cases])

    rows: list[RetrievalBenchmarkRow] = []
    for min_score in min_scores:
        for max_hits in max_hits_values:
            metrics = evaluate_search_queries(
                vector_base,
                query_cases,
                query_embeddings,
                min_score,
                max_hits,
            )
            rows.append(RetrievalBenchmarkRow(min_score, max_hits, metrics))

    print(f"Mode: retrieval")
    print(f"Model: {model.model_name}")
    print(f"Messages indexed: {len(message_texts)}")
    print(f"Queries evaluated: {len(query_cases)}")
    print()
    print_retrieval_rows(rows)

    best_row = select_best_retrieval_row(rows)
    print()
    print("Best-scoring retrieval row:")
    print(f"  min_score={best_row.min_score:.2f}")
    print(f"  max_hits={best_row.max_hits}")
    print(f"  hit_rate={best_row.metrics.hit_rate:.2f}%")
    print(f"  mrr={best_row.metrics.mean_reciprocal_rank:.4f}")


async def run_answer_benchmark(
    repo_root: Path,
    model_spec: str | None,
    min_scores: list[float],
    max_hits_values: list[int],
    limit: int,
) -> None:
    query_cases = load_answer_queries(repo_root)
    if not query_cases:
        raise ValueError("No answer eval cases found in the dataset")
    if limit > 0:
        query_cases = query_cases[:limit]

    context = await create_true_eval_context(repo_root, model_spec)

    rows: list[AnswerBenchmarkRow] = []
    for min_score in min_scores:
        for max_hits in max_hits_values:
            row_started_at = time.perf_counter()
            print(
                f"Evaluating min_score={min_score:.2f}, max_hits={max_hits}...",
                flush=True,
            )
            metrics = await evaluate_answer_queries(
                context,
                query_cases,
                min_score,
                max_hits,
            )
            rows.append(AnswerBenchmarkRow(min_score, max_hits, metrics))
            row_elapsed = time.perf_counter() - row_started_at
            print(
                "  Completed row: "
                f"type_match_rate={metrics.answer_type_match_rate:.2f}%, "
                f"exact_or_near_answer_rate={metrics.exact_or_near_answer_rate:.2f}%, "
                f"mean_similarity={metrics.mean_semantic_similarity:.4f} "
                f"in {row_elapsed:.1f}s",
                flush=True,
            )

    print()
    print(f"Mode: answer")
    print(f"Model: {context.embedding_model.model_name}")
    print(f"Queries evaluated: {len(query_cases)}")
    print()
    print_answer_rows(rows)

    best_row = select_best_answer_row(rows)
    print()
    print("Best-scoring true-eval row:")
    print(f"  min_score={best_row.min_score:.2f}")
    print(f"  max_hits={best_row.max_hits}")
    print(f"  answer_type_match_rate={best_row.metrics.answer_type_match_rate:.2f}%")
    print(
        "  exact_or_near_answer_rate="
        f"{best_row.metrics.exact_or_near_answer_rate:.2f}%"
    )
    print(
        "  mean_semantic_similarity=" f"{best_row.metrics.mean_semantic_similarity:.4f}"
    )
    print(f"  no_answer_match_rate={best_row.metrics.no_answer_match_rate:.2f}%")
    print(
        "  classification_mismatch_count="
        f"{best_row.metrics.classification_mismatch_count}"
    )


async def run_benchmark(
    mode: BenchmarkMode,
    model_spec: str | None,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
    limit: int,
) -> None:
    load_dotenv()
    repo_root = Path(__file__).resolve().parent.parent

    if mode == "retrieval":
        await run_retrieval_benchmark(
            repo_root,
            model_spec,
            min_scores,
            max_hits_values,
            batch_size,
        )
    else:
        await run_answer_benchmark(
            repo_root,
            model_spec,
            min_scores,
            max_hits_values,
            limit,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval settings for an embedding model."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["answer", "retrieval"],
        default="answer",
        help="Use 'answer' for the slow true eval path or 'retrieval' for the narrow messageMatches benchmark.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Provider and model name, e.g. 'openai:text-embedding-3-small'",
    )
    parser.add_argument(
        "--min-scores",
        type=str,
        default=None,
        help="Comma-separated min_score values to test.",
    )
    parser.add_argument(
        "--max-hits",
        type=str,
        default=None,
        help="Comma-separated max_hits values to test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used when building the retrieval-only benchmark index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of true-eval questions to run (default: all). Ignored in retrieval mode.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            mode=args.mode,
            model_spec=args.model,
            min_scores=parse_float_list(args.min_scores),
            max_hits_values=parse_int_list(args.max_hits),
            batch_size=args.batch_size,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
