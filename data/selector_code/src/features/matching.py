"""Answer-matching primitives (frozen semantics) for M0/M1 methods.

normalize = strip/lower/collapse-whitespace; agreement = bidirectional normalized
substring; contains = one-directional (poison-copy check). Embedding/NLI matchers
are optional and degrade gracefully to None when unavailable.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

_WS = re.compile(r"\s+")

# Lazily loaded sentence-transformers model (None until first successful load;
# False once a load attempt has failed, so we do not retry every call).
_EMBED_MODEL = None
_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_NLI_MODEL = None
_NLI_MODEL_NAME = None


def normalize(s: str) -> str:
    """Frozen normalization: strip, lowercase, collapse internal whitespace."""
    return _WS.sub(" ", s.strip().lower())


def normalized_exact(a: str, b: str) -> bool:
    """True iff a and b are identical after normalization."""
    return normalize(a) == normalize(b)


def agreement(a: str, b: str) -> bool:
    """Frozen agreement semantics: bidirectional normalized substring.

    True iff normalize(a) is a substring of normalize(b) or vice versa.
    Empty strings never agree (an empty normalized answer matches nothing).
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def alias_match(answer: str, aliases: List[str]) -> bool:
    """Bidirectional normalized substring match of answer against any alias."""
    return any(agreement(answer, alias) for alias in aliases)


def contains(target: str, answer: str) -> bool:
    """One-directional containment: normalized target occurs in normalized answer.

    Used for poison-copy detection (did the answer copy the target string).
    """
    nt = normalize(target)
    if not nt:
        return False
    return nt in normalize(answer)


def token_f1(a: str, b: str) -> float:
    """Token-level F1 over whitespace tokens of the normalized strings.

    Both empty -> 1.0; exactly one empty -> 0.0.
    """
    ta, tb = normalize(a).split(), normalize(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    overlap = sum((Counter(ta) & Counter(tb)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(a: str, b: str, no_embed: bool = False) -> Optional[float]:
    """Cosine similarity under bge-base sentence embeddings; None if unavailable.

    The model is loaded lazily on first use. Returns None when no_embed is set,
    when sentence_transformers is not installed, or when the model cannot load.
    """
    global _EMBED_MODEL
    if no_embed or _EMBED_MODEL is False:
        return None
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
        except Exception:
            _EMBED_MODEL = False
            return None
    try:
        emb = _EMBED_MODEL.encode([a, b], normalize_embeddings=True)
        return float((emb[0] * emb[1]).sum())
    except Exception:
        return None


def nli_equivalence(a: str, b: str) -> Optional[bool]:
    """Return bidirectional NLI equivalence using ``TRACE_NLI_MODEL``.

    Equivalence requires entailment in both directions and neither direction
    may have contradiction as its highest class.  With no configured local
    checkpoint the function remains explicitly unavailable (``None``).
    """
    import os
    global _NLI_MODEL, _NLI_MODEL_NAME
    model_name = os.environ.get("TRACE_NLI_MODEL")
    if not model_name or not normalize(a) or not normalize(b):
        return None
    if _NLI_MODEL is False or (_NLI_MODEL is not None and _NLI_MODEL_NAME != model_name):
        return None
    if _NLI_MODEL is None:
        try:
            from transformers import pipeline  # type: ignore
            _NLI_MODEL = pipeline("text-classification", model=model_name, tokenizer=model_name,
                                  truncation=True, device=0)
            _NLI_MODEL_NAME = model_name
        except Exception:
            _NLI_MODEL = False
            return None
    try:
        out = _NLI_MODEL([(a, b), (b, a)], top_k=None)
        def entailment_score(item):
            scores = {str(x["label"]).lower(): float(x["score"]) for x in item}
            return scores.get("entailment", 0.0), scores.get("contradiction", 0.0)
        e1, c1 = entailment_score(out[0])
        e2, c2 = entailment_score(out[1])
        return bool(e1 >= 0.5 and e2 >= 0.5 and c1 < e1 and c2 < e2)
    except Exception:
        return None


def dedupe_candidates(candidates: Dict[str, Optional[str]]) -> List[dict]:
    """Merge candidate answers that are equal under agreement().

    candidates maps a source label (e.g. 'C', 'R', 'P') to an answer string or
    None. Returns a list of {answer, sources} groups in first-seen order; the
    surface form kept is the first-seen answer. None/empty answers are skipped.
    A new answer joins the first existing group it agrees with.
    """
    groups: List[dict] = []
    for source, answer in candidates.items():
        if answer is None or not normalize(answer):
            continue
        for g in groups:
            if agreement(answer, g["answer"]):
                g["sources"].append(source)
                break
        else:
            groups.append({"answer": answer, "sources": [source]})
    return groups
