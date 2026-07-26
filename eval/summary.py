from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import EvalResult


def result_to_json(result: EvalResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "row_index": result.row_index,
        "id": result.id,
        "dataset": result.dataset,
        "split": result.split,
        "model": result.model,
        "engine": result.engine,
        "answer": result.answer,
        "prediction": result.prediction,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "strict_prediction": result.strict_prediction,
        "strict_correct": result.strict_correct,
        "response": result.response,
        "category": result.category,
        "question_type": result.question_type,
        "generation_elapsed_sec": result.generation_elapsed_sec,
        "batch_generation_elapsed_sec": result.batch_generation_elapsed_sec,
        "judge_enabled": result.judge_enabled,
        "judge_provider": result.judge_provider,
        "judge_task": result.judge_task,
        "judge_model": result.judge_model,
        "judge_prediction": result.judge_prediction,
        "judge_correct": result.judge_correct,
        "judge_used_for_score": result.judge_used_for_score,
        "judge_response": result.judge_response,
        "judge_error": result.judge_error,
        "judge_elapsed_sec": result.judge_elapsed_sec,
        "sample_eval_elapsed_sec": result.sample_eval_elapsed_sec,
        "total_eval_elapsed_sec": result.total_eval_elapsed_sec,
        "error": result.error,
        "metadata": result.metadata,
    }


def write_result(handle, result: EvalResult) -> None:
    handle.write(json.dumps(result_to_json(result), ensure_ascii=False) + "\n")
    handle.flush()


def completed_indices(output_path: Path) -> set[int]:
    if not output_path.is_file():
        return set()
    done: set[int] = set()
    for record in _latest_records(_read_records(output_path)):
        if record.get("ok") and isinstance(record.get("row_index"), int):
            done.add(int(record["row_index"]))
    return done


def summarize_results(output_path: Path, total_eval_elapsed_sec: float | None = None) -> dict[str, Any]:
    raw_records = _read_records(output_path)
    records = _latest_records(raw_records)
    summary = _summarize_subset(records)
    summary["raw_records"] = len(raw_records)
    summary["total_eval_elapsed_sec"] = total_eval_elapsed_sec

    by_dataset: dict[str, Any] = {}
    by_split: dict[str, Any] = {}
    by_category: dict[str, Any] = {}
    by_cycle_category: dict[str, Any] = {}
    by_dataset_category: dict[str, dict[str, Any]] = {}
    for key, group in _group_records(records, "dataset").items():
        by_dataset[key] = _summarize_subset(group)
        dataset_categories = {
            category: _summarize_subset(category_group)
            for category, category_group in _group_records(group, "category").items()
        }
        if dataset_categories:
            by_dataset_category[key] = dataset_categories
    for key, group in _group_records(records, "split").items():
        by_split[key] = _summarize_subset(group)
    for key, group in _group_records(records, "category").items():
        by_category[key] = _summarize_subset(group)
    for key, group in _group_records_by_metadata(records, "cycle_category").items():
        by_cycle_category[key] = _summarize_subset(group)
    summary["by_dataset"] = by_dataset
    summary["by_split"] = by_split
    summary["by_category"] = by_category
    summary["by_dataset_category"] = by_dataset_category
    summary["by_cycle_category"] = by_cycle_category
    dataset_metrics = _dataset_specific_metrics(records)
    if dataset_metrics:
        summary["dataset_metrics"] = dataset_metrics
    return summary


def _read_records(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        latest[_record_key(record)] = record
    return list(latest.values())


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    row_index = record.get("row_index")
    return (
        str(record.get("dataset") or ""),
        str(record.get("split") or ""),
        "" if row_index is None else str(row_index),
        str(record.get("id") or ""),
    )


def _group_records(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is None or value == "":
            continue
        groups[str(value)].append(record)
    return dict(groups)


def _group_records_by_metadata(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        value = metadata.get(field)
        if value is None or value == "":
            continue
        groups[str(value)].append(record)
    return dict(groups)


def _summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    ok_records = [record for record in records if record.get("ok")]
    ok = len(ok_records)
    correct = sum(1 for record in ok_records if record.get("correct") is True)
    failed = total - ok
    unparsable = sum(1 for record in ok_records if record.get("prediction") is None)
    strict_correct = sum(1 for record in ok_records if record.get("strict_correct") is True)
    strict_unparsable = sum(
        1
        for record in ok_records
        if "strict_prediction" in record and record.get("strict_prediction") is None
    )
    judge_records = [record for record in ok_records if record.get("judge_enabled")]
    judge_failed = sum(1 for record in judge_records if record.get("judge_error"))
    judge_correct = sum(1 for record in judge_records if record.get("judge_correct") is True)
    judge_unparsable = sum(1 for record in judge_records if record.get("judge_correct") is None)
    judge_times = [
        float(record["judge_elapsed_sec"])
        for record in judge_records
        if record.get("judge_elapsed_sec") is not None
    ]
    generation_times = [
        float(record["generation_elapsed_sec"])
        for record in ok_records
        if record.get("generation_elapsed_sec") is not None
    ]
    sample_times = [
        float(record["sample_eval_elapsed_sec"])
        for record in ok_records
        if record.get("sample_eval_elapsed_sec") is not None
    ]
    return {
        "total_records": total,
        "ok": ok,
        "failed": failed,
        "correct": correct,
        "unparsable": unparsable,
        "accuracy": correct / total if total else None,
        "accuracy_on_ok": correct / ok if ok else None,
        "strict_correct": strict_correct,
        "strict_unparsable": strict_unparsable,
        "strict_accuracy_on_ok": strict_correct / ok if ok else None,
        "judge_records": len(judge_records),
        "judge_failed": judge_failed,
        "judge_correct": judge_correct,
        "judge_unparsable": judge_unparsable,
        "judge_accuracy_on_judged": judge_correct / len(judge_records) if judge_records else None,
        "judge_elapsed_sec": _mean(judge_times),
        "mean_judge_elapsed_sec": _mean(judge_times),
        "generation_elapsed_sec": _mean(generation_times),
        "sample_eval_elapsed_sec": _mean(sample_times),
        "mean_generation_elapsed_sec": _mean(generation_times),
        "mean_sample_eval_elapsed_sec": _mean(sample_times),
    }


def _dataset_specific_metrics(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute opt-in official metrics without changing generic summaries."""
    metrics: dict[str, dict[str, Any]] = {}
    pope_records = [record for record in records if record.get("dataset") == "pope"]
    if pope_records:
        by_split = {
            split: _pope_metric_block(group)
            for split, group in sorted(_group_records(pope_records, "split").items())
        }
        metrics["pope"] = {
            "primary_metric": "f1",
            "positive_label": "yes",
            "overall": _pope_metric_block(pope_records),
            "by_split": by_split,
        }

    mmvp_records = [record for record in records if record.get("dataset") == "mmvp"]
    if mmvp_records:
        metrics["mmvp"] = _mmvp_metrics(mmvp_records)
    return metrics


def _pope_metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("ok")]
    evaluated = [
        record
        for record in ok_records
        if _pope_label(record.get("answer")) is not None
        and _pope_label(record.get("prediction")) is not None
    ]
    true_positive = sum(
        1
        for record in evaluated
        if _pope_label(record.get("answer")) == "yes"
        and _pope_label(record.get("prediction")) == "yes"
    )
    false_positive = sum(
        1
        for record in evaluated
        if _pope_label(record.get("answer")) == "no"
        and _pope_label(record.get("prediction")) == "yes"
    )
    true_negative = sum(
        1
        for record in evaluated
        if _pope_label(record.get("answer")) == "no"
        and _pope_label(record.get("prediction")) == "no"
    )
    false_negative = sum(
        1
        for record in evaluated
        if _pope_label(record.get("answer")) == "yes"
        and _pope_label(record.get("prediction")) == "no"
    )
    precision = _zero_safe_ratio(true_positive, true_positive + false_positive, evaluated)
    recall = _zero_safe_ratio(true_positive, true_positive + false_negative, evaluated)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if evaluated else None
    )
    return {
        "total_records": len(records),
        "ok": len(ok_records),
        "failed": len(records) - len(ok_records),
        "evaluated_records": len(evaluated),
        "unparsable": len(ok_records) - len(evaluated),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "accuracy": _ratio(true_positive + true_negative, len(evaluated)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": _ratio(true_positive + false_positive, len(evaluated)),
    }


def _pope_label(value: Any) -> str | None:
    label = str(value).strip().lower() if value is not None else ""
    return label if label in {"yes", "no"} else None


def _mmvp_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = _mmvp_pair_id(record)
        if pair_id is not None:
            pairs[pair_id].append(record)

    complete_pairs = [group for group in pairs.values() if len(group) == 2]
    pair_correct = sum(
        1
        for group in complete_pairs
        if all(record.get("ok") and record.get("correct") is True for record in group)
    )
    strict_pair_correct = sum(
        1
        for group in complete_pairs
        if all(record.get("ok") and record.get("strict_correct") is True for record in group)
    )
    question_correct = sum(
        1 for record in records if record.get("ok") and record.get("correct") is True
    )
    strict_question_correct = sum(
        1
        for record in records
        if record.get("ok") and record.get("strict_correct") is True
    )
    question_accuracy = _ratio(question_correct, len(records))
    pair_accuracy = _ratio(pair_correct, len(complete_pairs))
    strict_question_accuracy = _ratio(strict_question_correct, len(records))
    strict_pair_accuracy = _ratio(strict_pair_correct, len(complete_pairs))
    return {
        "primary_metric": "pair_accuracy",
        "total_records": len(records),
        "ok": sum(1 for record in records if record.get("ok")),
        "failed": sum(1 for record in records if not record.get("ok")),
        "question_correct": question_correct,
        "question_accuracy": question_accuracy,
        "pair_total": len(pairs),
        "complete_pairs": len(complete_pairs),
        "incomplete_pairs": len(pairs) - len(complete_pairs),
        "pair_correct": pair_correct,
        "pair_accuracy": pair_accuracy,
        "average": question_accuracy,
        "overall": pair_accuracy,
        "strict_question_correct": strict_question_correct,
        "strict_question_accuracy": strict_question_accuracy,
        "strict_pair_correct": strict_pair_correct,
        "strict_pair_accuracy": strict_pair_accuracy,
    }


def _mmvp_pair_id(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("pair_id") is not None:
        return str(metadata["pair_id"])
    try:
        index = int(str(record.get("id")))
    except (TypeError, ValueError):
        return None
    return str((index - 1) // 2 + 1)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _zero_safe_ratio(
    numerator: int,
    denominator: int,
    evaluated: list[dict[str, Any]],
) -> float | None:
    if denominator:
        return numerator / denominator
    return 0.0 if evaluated else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
