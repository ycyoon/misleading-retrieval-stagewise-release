"""Unified row schema for TRACE experiments.

Defines the Row dataclass (one row per (dataset, qid, condition)), JSONL
read/write helpers, and structural validation. gold_answers and
metadata.a_false are EVAL-ONLY; model-input code must use views.InputView.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

# Core condition strings (exact). C_insuf_b is an optional extra emitted only
# when clasheval adapters are run with include_insuf_b=True.
CORE_CONDITIONS = ("C_suf_clean", "C_poison", "C_mixed", "C_insuf")
ALLOWED_CONDITIONS = CORE_CONDITIONS + ("C_insuf_b",)

DEFAULT_READER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_RETRIEVER_NAME = "bge-base-en-v1.5+IVF18336"
DEFAULT_CORPUS_SNAPSHOT = "wikipedia-2018-12-20"


def empty_generation_scores() -> Dict[str, Any]:
    """A generation-scores dict with all fields unset (answered=False)."""
    return {"answered": False, "normalized": None, "logprobs": None,
            "entropy": None, "consistency": None}


@dataclass
class Row:
    """One unified experiment row: a (dataset, qid, condition) triple."""

    dataset: str
    split: str
    qid: str
    group_id: str
    condition: str
    question: str
    gold_answers: List[str]
    supplied_context: List[Dict[str, Any]]  # [{pid, text, source}]
    reference_passages: Optional[List[Dict[str, Any]]] = None  # [{pid, text}]
    candidates: Optional[List[Dict[str, Any]]] = None
    context_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    parametric_answer: Optional[str] = None
    context_generation_scores: Optional[Dict[str, Any]] = None
    reference_generation_scores: Optional[Dict[str, Any]] = None
    parametric_generation_scores: Optional[Dict[str, Any]] = None
    retrieval_scores: Optional[List[float]] = None
    source_document_ids: List[str] = field(default_factory=list)
    reader_model: str = DEFAULT_READER_MODEL
    retriever_name: str = DEFAULT_RETRIEVER_NAME
    corpus_snapshot: str = DEFAULT_CORPUS_SNAPSHOT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Row":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown row fields: {sorted(unknown)}")
        return cls(**d)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"invalid row: {msg}")


def validate_row(row: Row) -> None:
    """Raise ValueError if the row violates the unified schema."""
    for name in ("dataset", "split", "qid", "group_id", "condition", "question"):
        _check(isinstance(getattr(row, name), str) and getattr(row, name) != "",
               f"{name} must be a non-empty str")
    _check(row.condition in ALLOWED_CONDITIONS,
           f"condition {row.condition!r} not in {ALLOWED_CONDITIONS}")
    _check(isinstance(row.gold_answers, list) and
           all(isinstance(a, str) for a in row.gold_answers),
           "gold_answers must be a list of str")
    _check(isinstance(row.supplied_context, list), "supplied_context must be a list")
    for p in row.supplied_context:
        _check(isinstance(p, dict) and {"pid", "text", "source"} <= set(p),
               "supplied_context items need pid/text/source")
    if row.reference_passages is not None:
        for p in row.reference_passages:
            _check(isinstance(p, dict) and {"pid", "text"} <= set(p),
                   "reference_passages items need pid/text")
    for name in ("context_generation_scores", "reference_generation_scores",
                 "parametric_generation_scores"):
        scores = getattr(row, name)
        if scores is not None:
            _check(isinstance(scores, dict),
                   f"{name} must be a dict when present")
            if "answered" in scores:
                _check(isinstance(scores["answered"], bool),
                       f"{name}.answered must be bool")
    if row.retrieval_scores is not None:
        _check(all(isinstance(s, (int, float)) for s in row.retrieval_scores),
               "retrieval_scores must be floats")
    _check(isinstance(row.source_document_ids, list) and
           all(isinstance(s, str) for s in row.source_document_ids),
           "source_document_ids must be a list of str")
    _check(isinstance(row.metadata, dict), "metadata must be a dict")


def read_jsonl(path: str | Path, validate: bool = True) -> List[Row]:
    """Load rows from a unified-schema JSONL file."""
    return list(iter_jsonl(path, validate=validate))


def iter_jsonl(path: str | Path, validate: bool = True) -> Iterator[Row]:
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = Row.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{i + 1}: {e}") from e
            if validate:
                validate_row(row)
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Row], validate: bool = True) -> int:
    """Write rows to JSONL; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            if validate:
                validate_row(row)
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
