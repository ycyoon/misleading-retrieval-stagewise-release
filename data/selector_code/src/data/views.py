"""InputView: the only lens through which model-input/feature code may see rows.

Structurally omits eval-only fields — gold_answers, metadata.a_false,
condition, and dataset. Accessing any of those on an InputView raises
AttributeError. Feature code must accept InputView, never Row.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .schema import Row

# Metadata keys that may leak the label or condition identity.
_HIDDEN_METADATA_KEYS = ("a_false",)


class InputView:
    """Read-only projection of a Row safe for model-input / feature code."""

    __slots__ = (
        "row",
        "qid", "group_id", "split", "question",
        "supplied_context", "reference_passages", "candidates",
        "context_answer", "reference_answer", "parametric_answer",
        "context_generation_scores", "reference_generation_scores",
        "parametric_generation_scores",
        "retrieval_scores", "source_document_ids",
        "reader_model", "retriever_name", "corpus_snapshot", "metadata",
    )

    def __init__(self, row: Row) -> None:
        self.row = row
        self.qid: str = row.qid
        self.group_id: str = row.group_id
        self.split: str = row.split
        self.question: str = row.question
        self.supplied_context: List[Dict[str, Any]] = copy.deepcopy(row.supplied_context)
        self.reference_passages: Optional[List[Dict[str, Any]]] = (
            copy.deepcopy(row.reference_passages))
        self.candidates: Optional[List[Dict[str, Any]]] = copy.deepcopy(row.candidates)
        self.context_answer = row.context_answer
        self.reference_answer = row.reference_answer
        self.parametric_answer = row.parametric_answer
        self.context_generation_scores = copy.deepcopy(row.context_generation_scores)
        self.reference_generation_scores = copy.deepcopy(row.reference_generation_scores)
        self.parametric_generation_scores = copy.deepcopy(row.parametric_generation_scores)
        self.retrieval_scores = list(row.retrieval_scores) if row.retrieval_scores is not None else None
        self.source_document_ids = list(row.source_document_ids)
        self.reader_model = row.reader_model
        self.retriever_name = row.retriever_name
        self.corpus_snapshot = row.corpus_snapshot
        self.metadata: Dict[str, Any] = {
            k: copy.deepcopy(v) for k, v in row.metadata.items()
            if k not in _HIDDEN_METADATA_KEYS
        }

    def __repr__(self) -> str:  # no gold/condition/dataset in repr either
        return f"InputView(qid={self.qid!r}, n_ctx={len(self.supplied_context)})"


def input_view(row: Row) -> InputView:
    """Factory: project a Row down to its model-input-safe view."""
    return InputView(row)
