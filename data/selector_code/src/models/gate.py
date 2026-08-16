"""Learned gate M1: PRIMARY (answer_context vs abstain) and RECOVERY (4-way)
heads behind one interface, over sklearn LogisticRegression /
GradientBoostingClassifier / small MLPClassifier. Features come from InputView
only; the abstain threshold is chosen on dev by config utility, then FROZEN.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.src.data.views import InputView
from experiments.src.features.gate_features import FEATURE_NAMES, build_features
from experiments.src.features.matching import normalize

HEADS = ("primary", "recovery")
MODEL_TYPES = ("logreg", "gbt", "mlp")
ACTIONS = ("answer_context", "answer_reference", "answer_parametric", "abstain")

# action -> InputView answer attribute / candidate_matches key
_ANSWER_ATTR = {
    "answer_context": "context_answer",
    "answer_reference": "reference_answer",
    "answer_parametric": "parametric_answer",
}
_MATCH_KEY = {
    "answer_context": "context",
    "answer_reference": "reference",
    "answer_parametric": "parametric",
}

DEFAULT_THRESHOLD = 0.5


def _make_estimator(model_type: str, seed: int, params: Optional[dict]) -> Any:
    """Instantiate the sklearn estimator for one head."""
    params = dict(params or {})
    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        params.setdefault("max_iter", 1000)
        return LogisticRegression(random_state=seed, **params)
    if model_type == "gbt":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=seed, **params)
    if model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        params.setdefault("hidden_layer_sizes", (32, 16))
        params.setdefault("max_iter", 500)
        return MLPClassifier(random_state=seed, **params)
    raise ValueError(f"unknown model_type {model_type!r}; want one of {MODEL_TYPES}")


def featurize(
    views: Sequence[InputView],
    embedding_sims: Optional[Sequence[Optional[dict]]] = None,
) -> np.ndarray:
    """Feature matrix in fixed FEATURE_NAMES column order."""
    rows = []
    for i, view in enumerate(views):
        sims = embedding_sims[i] if embedding_sims is not None else None
        feats = build_features(view, embedding_sims=sims)
        rows.append([feats[name] for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float).reshape(len(views), len(FEATURE_NAMES))


class Gate:
    """One trained head. fit -> select_threshold (frozen) -> predict."""

    def __init__(self, head: str = "primary", model_type: str = "logreg",
                 seed: int = 0, model_params: Optional[dict] = None) -> None:
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        self.head = head
        self.model_type = model_type
        self.seed = seed
        self.model_params = dict(model_params or {})
        self.estimator = _make_estimator(model_type, seed, model_params)
        self.threshold: Optional[float] = None
        self._const: Optional[Any] = None  # degenerate single-class fallback

    # -- training ----------------------------------------------------------
    def fit(self, views: Sequence[InputView], labels: Sequence[Any],
            embedding_sims: Optional[Sequence[Optional[dict]]] = None) -> "Gate":
        """labels: ints {0,1} for primary; action strings for recovery."""
        y = self._check_labels(labels)
        X = featurize(views, embedding_sims)
        if len(set(y)) < 2:  # degenerate training set: constant prediction
            self._const = y[0]
            return self
        self.estimator.fit(X, np.asarray(y, dtype=object if self.head == "recovery" else int))
        return self

    def _check_labels(self, labels: Sequence[Any]) -> list:
        y = list(labels)
        if not y:
            raise ValueError("empty label list")
        if self.head == "primary":
            assert all(l in (0, 1) for l in y), "primary labels must be 0/1"
        else:
            assert all(l in ACTIONS for l in y), f"recovery labels must be in {ACTIONS}"
        return y

    # -- scoring -----------------------------------------------------------
    def _scores(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Per-row (p_answer, proposed answer action). p_answer is P(correct
        context) for primary, the best answer-class probability for recovery."""
        n = X.shape[0]
        if self._const is not None:
            if self.head == "primary":
                return (np.full(n, 1.0 if self._const == 1 else 0.0),
                        ["answer_context"] * n)
            if self._const == "abstain":
                return np.zeros(n), ["abstain"] * n
            return np.ones(n), [str(self._const)] * n
        proba = self.estimator.predict_proba(X)
        classes = list(self.estimator.classes_)
        if self.head == "primary":
            return proba[:, classes.index(1)], ["answer_context"] * n
        answer_idx = [i for i, c in enumerate(classes) if c != "abstain"]
        if not answer_idx:
            return np.zeros(n), ["abstain"] * n
        sub = proba[:, answer_idx]
        best = sub.argmax(axis=1)
        p = sub[np.arange(n), best]
        proposed = [classes[answer_idx[j]] for j in best]
        return p, proposed

    # -- threshold selection (dev; then FROZEN) ----------------------------
    def select_threshold(
        self,
        views: Sequence[InputView],
        candidate_matches: Sequence[Dict[str, bool]],
        c: float = 2.0,
        embedding_sims: Optional[Sequence[Optional[dict]]] = None,
    ) -> float:
        """Choose the abstain threshold on dev by maximizing sum utility:
        +1 per answered-and-correct row, -c per answered-and-wrong, 0 abstain.
        Ties prefer the higher (more abstaining) threshold. Freezes it."""
        X = featurize(views, embedding_sims)
        p, proposed = self._scores(X)
        correct = np.array([
            bool(m.get(_MATCH_KEY[a], False)) if a != "abstain" else False
            for m, a in zip(candidate_matches, proposed)
        ])
        candidates = np.unique(np.concatenate([p, [0.0, 1.0 + 1e-9]]))
        best_u, best_t = -np.inf, float(candidates[-1])
        for t in np.sort(candidates):  # ascending; >= keeps highest tied t
            answered = p >= t
            n_ans = int(answered.sum())
            n_correct = int(correct[answered].sum())
            u = n_correct - c * (n_ans - n_correct)
            if u >= best_u:
                best_u, best_t = u, float(t)
        self.threshold = best_t
        return best_t

    # -- prediction --------------------------------------------------------
    def predict(
        self,
        views: Sequence[InputView],
        embedding_sims: Optional[Sequence[Optional[dict]]] = None,
    ) -> List[Dict[str, Any]]:
        """Per-row {action, answer, p_answer} (+proposed_action). Abstains when
        p_answer < threshold or the proposed candidate answer is missing."""
        X = featurize(views, embedding_sims)
        p, proposed = self._scores(X)
        thr = self.threshold if self.threshold is not None else DEFAULT_THRESHOLD
        out: List[Dict[str, Any]] = []
        for view, pi, act in zip(views, p, proposed):
            action, answer = "abstain", None
            if act != "abstain" and pi >= thr:
                candidate = getattr(view, _ANSWER_ATTR[act])
                if isinstance(candidate, str) and normalize(candidate):
                    action, answer = act, candidate
            out.append({"action": action, "answer": answer,
                        "p_answer": float(pi), "proposed_action": act})
        return out

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "head": self.head,
            "model_type": self.model_type,
            "seed": self.seed,
            "model_params": self.model_params,
            "threshold": self.threshold,
            "feature_names": list(FEATURE_NAMES),
            "estimator": self.estimator,
            "const": self._const,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path) -> "Gate":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload["feature_names"] != list(FEATURE_NAMES):
            raise ValueError(
                f"{path}: feature set drift — model was trained with "
                f"{payload['feature_names']}, code has {list(FEATURE_NAMES)}")
        gate = cls(head=payload["head"], model_type=payload["model_type"],
                   seed=payload["seed"], model_params=payload["model_params"])
        gate.estimator = payload["estimator"]
        gate.threshold = payload["threshold"]
        gate._const = payload["const"]
        return gate
