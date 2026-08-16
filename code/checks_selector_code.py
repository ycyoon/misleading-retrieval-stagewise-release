"""selector_code: the shipped selector source, checked against the manuscript.

Every other family in this release recomputes a number from released rows.
This one closes a different gap. Until now the release carried only analysis
scripts, so the manuscript's description of the selector ARCHITECTURE -- its
loss weights, its hinge margin, its observable count, its heads -- could not be
checked at all. The authors' own model and feature source now ships under
`data/selector_code/src/`, and this module PARSES that shipped source and
asserts the values the manuscript prints.

Nothing here is hard-coded and compared to itself. Every `got` is read out of
the shipped file text with `ast` (or, for the YAML method spec, a small
indentation parser); every `want` is a number the manuscript prints, with the
manuscript line quoted beside it. Change a constant in
`data/selector_code/src/models/cgrv.py` and the matching check fails.

What is covered (manuscript paragraph "Competence-gated relational verifier",
beginning line 423, and "Verification-only fixed comparator", line 456):

  * the objective  L = L_action + 0.3 L_relation + 0.5 L_pair  (lines 439-445),
    that the shipped default objective has no further weighted term, that
    L_action is binary cross-entropy for verification and cross-entropy for
    answer selection (lines 446-447), and that L_pair is a margin-0.2 hinge
    ranking the correct-context variant above the incorrect one (lines 449-451);
  * the 18 source-level observables, the groups the sentence names
    (lines 427-430), and the log transform of counts and lengths (line 431);
  * the frozen BAAI/bge-base-en-v1.5 encoder and the question-plus-answer text
    it embeds (lines 424-427, 451);
  * the shared relation head over four classes (lines 433-434);
  * the decision layer: one competence gate per candidate source, one head per
    presence state, a verification head returning a_C or abstain and a
    selection head over any available candidate or abstain (lines 434-437);
  * the three development-split uses -- early stopping, calibration,
    cost-specific threshold selection (lines 451-453) -- and the rule that gold
    answers provide training targets but are never selector features
    (lines 453-454), read structurally off the shipped Row/InputView pair: no
    feature-name table carries a gold-derived name, the view projects none of
    the Row's eval-only fields, no module on the selector's input path reads
    `gold_answers`, and the label builder does;
  * the four-rule verification comparator registry and the availability,
    normalization and agreement semantics stated for it (lines 457-473).

The shipped source is reference source: it imports the original package
(`experiments.src...`), so it is parsed here, never imported. That is also what
makes these checks honest -- they read exactly the file a reviewer reads.
"""

from __future__ import annotations

import ast
import re

# ---------------------------------------------------------------------------
# Shipped files
# ---------------------------------------------------------------------------

CGRV = "src/models/cgrv.py"
RELATIONAL = "src/models/relational.py"
GATE = "src/models/gate.py"
RULES = "src/models/rules.py"
GATE_FEATURES = "src/features/gate_features.py"
NODE_FEATURES = "src/features/node_features.py"
MATCHING = "src/features/matching.py"
SCHEMA = "src/data/schema.py"
VIEWS = "src/data/views.py"
LABELS = "src/evaluation/labels.py"
CONFIG = "configs/protocol_full.yaml"

# The modules on the selector's INPUT path. None of them may read gold answers
# (manuscript line 453: "never selector features"); `labels.py` may, and must,
# because it builds the training targets.
INPUT_PATH = (CGRV, RELATIONAL, GATE, RULES, GATE_FEATURES, NODE_FEATURES,
              MATCHING, VIEWS)

# The manuscript's own arithmetic anchors, quoted rather than assumed:
#   "three frozen candidates generated from the supplied context, an
#    independently retrieved reference corpus, and parametric memory"
#    (abstract, line 50; "three-candidate frozen-decision comparison", line 492)
N_CANDIDATES = 3
N_PAIRS = N_CANDIDATES * (N_CANDIDATES - 1) // 2      # unordered pairs
N_PRESENCE_STATES = 2 ** N_CANDIDATES                 # one per presence pattern
N_ACTIONS = N_CANDIDATES + 1                          # "any candidate or abstain"
# "equivalent, compatible, contradictory, or unknown" (line 433)
REL_CLASSES_NAMED = ("equivalent", "compatible", "contradictory", "unknown")
# "I don't know" / "I dont know" (lines 469-470)
REFUSALS_PRINTED = ("i don't know", "i dont know")
# "BAAI/bge-base-en-v1.5" (line 425)
ENCODER_NAME = "BAAI/bge-base-en-v1.5"

_UNRESOLVED = object()


# ---------------------------------------------------------------------------
# Readers: python AST, and the two-space-indent YAML subset the config uses
# ---------------------------------------------------------------------------

def _value(node, env):
    """Best-effort constant folding for module-level assignments."""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return env.get(node.id, _UNRESOLVED)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _value(node.left, env), _value(node.right, env)
        if isinstance(left, tuple) and isinstance(right, tuple):
            return left + right
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "len" and len(node.args) == 1):
        inner = _value(node.args[0], env)
        if inner is not _UNRESOLVED and hasattr(inner, "__len__"):
            return len(inner)
    return _UNRESOLVED


def _consts(tree):
    """Constants assigned at module level (including module-level `if` bodies).

    Function-local and class-body assignments are deliberately not collected:
    a local variable must not be able to shadow the constant a check reads.
    """
    env = {}

    def walk(body):
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                val = _value(node.value, env) if node.value is not None \
                    else _UNRESOLVED
                if val is _UNRESOLVED:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = val
            elif isinstance(node, ast.If):
                walk(node.body)
                walk(node.orelse)

    walk(tree.body)
    return env


def _find_def(tree, name, cls=None):
    """The FunctionDef `name`, optionally restricted to ClassDef `cls`."""
    scope = tree
    if cls is not None:
        scope = next((n for n in ast.walk(tree)
                      if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if scope is None:
            raise SystemExit(
                f"selector_code: class {cls} not found in the shipped source")
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(
        f"selector_code: function {name} not found in the shipped source")


def _assign_value(tree, name):
    """Value node of the module-level `name = ...` assignment."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise SystemExit(f"selector_code: no module-level assignment to {name} in "
                     f"the shipped source; it has been restructured")


def _class_const(tree, cls, name):
    """Literal value of the class-body assignment `cls.name = ...`."""
    node = next((c for c in ast.walk(tree)
                 if isinstance(c, ast.ClassDef) and c.name == cls), None)
    if node is None:
        raise SystemExit(f"selector_code: class {cls} not found in the shipped "
                         f"source")
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            return ast.literal_eval(stmt.value)
    raise SystemExit(f"selector_code: {cls}.{name} not found in the shipped "
                     f"source")


def _names_used(tree, name):
    """How often `name` appears as a string constant or an attribute."""
    return sum(1 for n in ast.walk(tree)
               if (isinstance(n, ast.Constant) and n.value == name)
               or (isinstance(n, ast.Attribute) and n.attr == name))


def _key(node):
    """'k' for a `something["k"]` expression, else None."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _terms(node):
    """Flatten a chain of `+` into its summands."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _terms(node.left) + _terms(node.right)
    return [node]


def _calls(node):
    """Every called function/method name appearing inside an expression."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Attribute):
                out.append(sub.func.attr)
            elif isinstance(sub.func, ast.Name):
                out.append(sub.func.id)
    return out


def _assigned(func, key):
    """Value node of the (unique) `<name>["key"] = ...` inside `func`."""
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and _key(node.targets[0]) == key:
            return node.value
    raise SystemExit(f"selector_code: no assignment to [{key!r}] in "
                     f"{func.name}; the shipped source has been restructured")


def _scalar(text):
    text = text.strip()
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text.strip("'\"")


def _yaml(text):
    """Nested dicts for the two-space-indented `key: value` subset that
    protocol_full.yaml uses. Comments, blank lines and `- item` lists are
    dropped; none of the selector hyperparameters is a list. Written out here
    because the release's import set is numpy/pandas/pyarrow plus the standard
    library -- no pyyaml."""
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        if line.startswith("- ") or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key, rest = key.strip(), rest.split(" #")[0].strip()
        parent = stack[-1][1]
        if rest:
            parent[key] = _scalar(rest)
        else:
            child = {}
            parent[key] = child
            stack.append((indent, child))
    return root


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def run(D, check):
    P = D / "selector_code"

    def text(rel):
        path = P / rel
        if not path.is_file():
            raise SystemExit(f"selector_code: shipped source missing: {rel}")
        return path.read_text()

    cgrv_t = ast.parse(text(CGRV))
    rel_t = ast.parse(text(RELATIONAL))
    rules_t = ast.parse(text(RULES))
    match_t = ast.parse(text(MATCHING))
    node_t = ast.parse(text(NODE_FEATURES))
    gfeat_t = ast.parse(text(GATE_FEATURES))
    schema_t = ast.parse(text(SCHEMA))
    views_t = ast.parse(text(VIEWS))
    labels_t = ast.parse(text(LABELS))
    ast.parse(text(GATE))       # the generic supervised selector, also shipped
    cfg = _yaml(text(CONFIG))

    C, R = _consts(cgrv_t), _consts(rel_t)
    RU, NF, GF, MA = (_consts(rules_t), _consts(node_t),
                      _consts(gfeat_t), _consts(match_t))
    methods = cfg.get("methods", {})
    spec = methods.get("cgrv")
    if not isinstance(spec, dict) or not isinstance(spec.get("loss_weights"), dict):
        raise SystemExit("selector_code: protocol_full.yaml has no methods.cgrv "
                         "block with a loss_weights map; the method-spec checks "
                         "below would be vacuous")

    # ---- the training objective ------------------------------------------
    # Manuscript, lines 439-445:
    #   "Training minimizes   L = L_action + 0.3 L_relation + 0.5 L_pair ."
    print("== selector_code: the training objective ==")
    w = C["DEFAULT_LOSS_WEIGHTS"]
    check("relation-loss weight, cgrv.py (line 443)", float(w["relation"]), 0.3)
    check("pair-loss weight, cgrv.py (line 444)",
          float(w["counterfactual"]), 0.5)
    # L_action carries no printed coefficient (line 442): weight 1 on both
    # action heads, verification and answer selection.
    check("verification action-loss weight (line 442)", float(w["primary"]), 1.0)
    check("selection action-loss weight (line 442)", float(w["recovery"]), 1.0)
    # The printed objective has exactly three terms. The shipped module keeps
    # two further switches (source_conf, gate_l2); both default to OFF, so the
    # default objective IS the printed equation. Defaulting either one on would
    # make the paper's equation wrong, and fails here.
    printed = ("primary", "recovery", "relation", "counterfactual")
    check("weight on terms outside the printed equation",
          float(sum(float(v) for k, v in w.items() if k not in printed)),
          0.0, tol=0)

    # A weight table is only a claim if the weights multiply the losses they
    # name, so the assembled objective is parsed rather than trusted.
    losses = _find_def(cgrv_t, "_batch_losses")
    named, mismatched = set(), 0
    for term in _terms(_assigned(losses, "total")):
        if _key(term) is not None:                     # unweighted summand
            named.add(_key(term))
            continue
        if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
            loss_key, weight_key = _key(term.right), None
            for sub in ast.walk(term.left):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    weight_key = sub.value
            if loss_key is not None:
                named.add(loss_key)
                mismatched += int(weight_key != loss_key)
    check("loss terms scaled by a differently-named weight",
          float(mismatched), 0.0, tol=0)
    check("printed terms present in the assembled total loss",
          float(len({"action", "relation", "counterfactual"} & named)), 3, tol=0)
    action_keys = {_key(t.right) for t in _terms(_assigned(losses, "action"))
                   if isinstance(t, ast.BinOp) and isinstance(t.op, ast.Mult)}
    check("action heads summed into L_action (line 446)",
          float(len(action_keys & {"primary", "recovery"})), 2, tol=0)

    # Manuscript, lines 446-447: "L_action is binary cross-entropy for
    # verification and cross-entropy for answer selection."
    by_loss = {}
    for node in ast.walk(losses):
        if isinstance(node, ast.Assign) and _key(node.targets[0]):
            by_loss.setdefault(_key(node.targets[0]), []).extend(
                _calls(node.value))
    check("verification loss is binary cross-entropy (line 446)",
          float("binary_cross_entropy_with_logits" in by_loss.get("primary", [])),
          1.0, tol=0)
    check("selection loss is cross-entropy (line 447)",
          float("cross_entropy" in by_loss.get("recovery", [])), 1.0, tol=0)
    # Line 448 says L_relation "trains the relation head using frozen
    # Qwen2.5-72B pair labels"; it does NOT print a loss name, so what is
    # asserted here is only that the shipped term is a classification loss over
    # those categorical pair labels -- not a number the manuscript quotes.
    check("relation head trained by a classification loss (line 448)",
          float("cross_entropy" in by_loss.get("relation", [])), 1.0, tol=0)

    # Manuscript, lines 449-451: "L_pair is a margin-0.2 hinge loss that ranks
    # a question's correct context variant above its incorrect variant."
    check("hinge margin, relational.py (line 450)",
          float(R["DEFAULT_CF_MARGIN"]), 0.2)
    cf = _find_def(rel_t, "counterfactual_loss")
    hinges = [n for n in ast.walk(cf)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "relu"
              and any(isinstance(s, ast.Name) and s.id == "margin"
                      for s in ast.walk(n))]
    check("hinge terms over that margin in L_pair (line 450)",
          float(len(hinges)), 1.0, tol=0)
    # Orientation: the ranked-above member of every pair must be the variant
    # whose context supports the accepted answer, never the poisoned or the
    # answer-free one.
    conds, pairs = R["CF_CONDITIONS"], R["CF_PAIRS"]
    corrupted = {"C_poison", "C_insuf"}
    if not pairs or not (corrupted <= set(conds)):
        raise SystemExit("selector_code: the counterfactual condition table no "
                         "longer names the corrupted-context conditions; the "
                         "orientation check would be vacuous")
    check("pairs ranking an incorrect context above a correct one (line 451)",
          float(sum(1 for hi, lo in pairs
                    if conds[hi] in corrupted or conds[lo] not in corrupted)),
          0.0, tol=0)

    # The same three weights and the same margin in the method spec the
    # released panel was produced from.
    print("== selector_code: the same objective in the shipped method spec ==")
    lw = spec["loss_weights"]
    check("relation-loss weight, methods.cgrv (line 443)",
          float(lw["relation"]), 0.3)
    check("pair-loss weight, methods.cgrv (line 444)",
          float(lw["counterfactual"]), 0.5)
    check("action-loss weights, methods.cgrv (line 442)",
          float(lw["primary"]) + float(lw["recovery"]), 2.0)
    check("hinge margin, methods.cgrv (line 450)",
          float(spec["counterfactual"]["margin"]), 0.2)
    check("methods.cgrv terms beyond the printed three",
          float(len(set(lw) - set(printed))), 0.0, tol=0)

    # ---- the 18 source-level observables ---------------------------------
    # Manuscript, lines 427-431:
    #   "The selector also receives 18 source-level observables covering
    #    context length and answer presence, reference retrieval-score and
    #    passage-count statistics, and parametric answer presence, length,
    #    abstention wording, and familiarity. Counts and lengths are
    #    log-transformed."
    print("== selector_code: the source-level observables ==")
    obs = C["OBS_FEATURES"]
    check("source-level observables the selector receives (line 427)",
          float(len(obs)), 18, tol=0)
    check("duplicate observable names", float(len(obs) - len(set(obs))),
          0.0, tol=0)
    # The per-source blocks are READ OUT of the shipped `OBS_FEATURES = OBS_C +
    # OBS_R + OBS_P` assignment rather than named here, so merging two blocks in
    # the shipped source drops this count instead of leaving it at a constant 3.
    summands = _terms(_assign_value(cgrv_t, "OBS_FEATURES"))
    block_names = [t.id for t in summands if isinstance(t, ast.Name)]
    if len(block_names) != len(summands) or not block_names:
        raise SystemExit("selector_code: OBS_FEATURES is no longer a sum of "
                         "named per-source blocks; the block checks would be "
                         "vacuous")
    blocks = {n: C[n] for n in block_names}
    check("observable blocks, one per candidate source",
          float(len(blocks)), N_CANDIDATES, tol=0)
    check("observables belonging to exactly one source block",
          float(sum(1 for n in obs
                    if sum(n in b for b in blocks.values()) == 1)), 18, tol=0)
    # The sentence enumerates eight observable groups; each must be populated
    # in the shipped tuple. Rename any one of these names and this fails.
    GROUPS = (
        ("context length", ("c_n_passages", "c_total_chars", "c_mean_chars",
                            "c_len")),
        ("context answer presence", ("c_commit",)),
        ("reference retrieval score", ("r_mean", "r_max", "r_std", "r_margin",
                                       "r_count", "r_missing")),
        ("reference passage count", ("r_n_passages",)),
        ("parametric answer presence", ("p_present",)),
        ("parametric answer length", ("p_len",)),
        ("parametric abstention wording", ("p_abstain_text",)),
        ("parametric familiarity", ("p_familiarity", "p_familiarity_missing")),
    )
    check("observable groups the sentence names (lines 428-430)",
          float(sum(1 for _, names in GROUPS if all(n in obs for n in names))),
          len(GROUPS), tol=0)
    spare = [n for n in obs
             if n not in {x for _, names in GROUPS for x in names}]
    # Reported, not asserted: the sentence does not name a reference ANSWER
    # length, and the shipped tuple carries one.
    print(f"  note  {len(spare)} shipped observable(s) outside the groups the "
          f"sentence names: {', '.join(spare) or 'none'}")

    # "Counts and lengths are log-transformed" (line 431).
    countlen = re.compile(r"(_n_passages|_count|_chars|_len)$")
    n_countlen = sum(1 for n in obs if countlen.search(n))
    if n_countlen == 0:
        raise SystemExit("selector_code: no count/length observable names "
                         "matched; the log-transform check would be vacuous")
    builder = _find_def(cgrv_t, "observable_features")
    unlogged = [k for node in ast.walk(builder)
                if isinstance(node, ast.Assign)
                for k in [_key(node.targets[0])]
                if k and countlen.search(k)
                and "log1p" not in _calls(node.value)]
    check(f"count/length observables built without log1p (of {n_countlen})",
          float(len(unlogged)), 0.0, tol=0)

    # ---- the frozen encoder ----------------------------------------------
    # Manuscript, lines 424-427: "a frozen BAAI/bge-base-en-v1.5 encoder
    # jointly embeds the question and candidate answer"; line 451: "The encoder
    # remains frozen".
    print("== selector_code: the frozen encoder ==")
    enc_init = _find_def(rel_t, "__init__", cls="BgeEncoder")
    make_enc = _find_def(rel_t, "make_encoder")
    sites = (int(any(isinstance(d, ast.Constant) and d.value == ENCODER_NAME
                     for d in enc_init.args.defaults))
             + int(any(isinstance(n, ast.Constant) and n.value == ENCODER_NAME
                       for n in ast.walk(make_enc)))
             + int(MA.get("_EMBED_MODEL_NAME") == ENCODER_NAME)
             + int(spec["encoder"]["model_name"] == ENCODER_NAME))
    check("shipped defaults naming the paper's encoder (line 425)",
          float(sites), 4.0, tol=0)
    # Frozen: the encoder is switched to eval mode when loaded, and the trained
    # module holds no encoder submodule, so no encoder weight can take a
    # gradient from the objective checked above.
    check("eval() calls freezing the shipped encoder (line 451)",
          float(_calls(_find_def(rel_t, "_load", cls="BgeEncoder")).count("eval")),
          1.0, tol=0)
    cgrv_init = _find_def(cgrv_t, "__init__", cls="CGRV")
    attrs = [t.attr for n in ast.walk(cgrv_init) if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Attribute)]
    check("encoder submodules inside the trained selector (line 451)",
          float(sum(1 for a in attrs if "encod" in a.lower())), 0.0, tol=0)
    # "jointly embeds the question and candidate answer": the embedded string
    # is question + separator + answer and nothing else -- no evidence text.
    joined = _terms(next(n for n in ast.walk(_find_def(cgrv_t, "answer_texts"))
                         if isinstance(n, ast.IfExp)).orelse)
    check("text fields joined for the encoder (line 426)",
          float(sum(1 for p in joined
                    if not (isinstance(p, ast.Name) and p.id == "SEP"))),
          2.0, tol=0)
    check("evidence fields in the embedded text (line 426)",
          float(sum(1 for p in joined if isinstance(p, ast.Attribute)
                    and p.attr in ("supplied_context", "reference_passages"))),
          0.0, tol=0)

    # ---- the shared relation head ----------------------------------------
    # Manuscript, lines 433-434: "A shared relation head predicts whether pairs
    # of candidate answers are equivalent, compatible, contradictory, or
    # unknown."
    print("== selector_code: the relation head ==")
    classes = C["REL_CLASSES"]
    check("relation classes the sentence names, found in the head (line 433)",
          float(sum(1 for c in REL_CLASSES_NAMED if c in classes)),
          len(REL_CLASSES_NAMED), tol=0)
    check("relation classes in the shipped head (line 433)",
          float(len(classes)), len(REL_CLASSES_NAMED), tol=0)
    check("output width of the typed relation head",
          float(C["N_TYPED"]), len(REL_CLASSES_NAMED), tol=0)
    check("relation-head modules (line 433: one shared head)",
          float(sum(1 for a in attrs if a == "rel_mlp")), 1.0, tol=0)
    check("candidate pairs that one head scores",
          float(len(C["PAIR_TAGS"])), N_PAIRS, tol=0)
    check("candidate pairs in the node-feature pair table",
          float(len(NF["PAIRS"])), N_PAIRS, tol=0)
    # Manuscript, line 448: L_relation uses "frozen Qwen2.5-72B pair labels".
    # The typed head is the one that consumes them; the method spec the
    # released panel was produced from turns it on and names the label source.
    check("typed relation head enabled in methods.cgrv (line 433)",
          float(bool(spec["model"]["typed_relations"])), 1.0, tol=0)
    check("pair-label source configured for L_relation (line 448)",
          float(bool(spec.get("relation_labels"))), 1.0, tol=0)

    # ---- the decision layer ----------------------------------------------
    # Manuscript, lines 434-437: "Decision heads then combine source
    # competence, candidate presence, and these learned relations.
    # Verification returns only a_C or abstains, whereas answer selection may
    # return any available candidate or abstain."
    print("== selector_code: the decision layer ==")
    check("competence gates named by the selector",
          float(len(C["GATE_KEYS"])), N_CANDIDATES, tol=0)
    check("competence-gate modules built in the selector",
          float(sum(1 for a in attrs if re.fullmatch(r"gate_[crp]", a))),
          N_CANDIDATES, tol=0)
    z = C["Z_FEATURES"]
    check("competence gates entering the decision vector (line 435)",
          float(sum(1 for n in z if re.fullmatch(r"g_[crp]", n, re.I))),
          N_CANDIDATES, tol=0)
    check("candidate-presence indicators in the decision vector (line 435)",
          float(sum(1 for n in z if re.fullmatch(r"c_[crp]", n, re.I))),
          N_CANDIDATES, tol=0)
    check("per-pair relation probabilities in the decision vector (line 435)",
          float(sum(1 for n in z
                    if re.fullmatch(r"(eq|comp|contra)_(cr|cp|rp)", n))),
          N_PAIRS * (len(REL_CLASSES_NAMED) - 1), tol=0)
    check("presence states for the decision heads",
          float(C["N_STATES"]), N_PRESENCE_STATES, tol=0)
    # The presence pattern must be encoded injectively, or "per presence state"
    # would collapse states onto one head.
    coeffs = []
    for term in _terms(next(n.value for n in ast.walk(_find_def(cgrv_t, "state_index"))
                            if isinstance(n, ast.Return)
                            and isinstance(n.value, ast.BinOp))):
        coeffs.append(int(term.right.value)
                      if isinstance(term, ast.BinOp)
                      and isinstance(term.op, ast.Mult)
                      and isinstance(term.right, ast.Constant) else 1)
    check("distinct presence patterns the state index separates",
          float(len({sum(c * ((bits >> i) & 1) for i, c in enumerate(coeffs))
                     for bits in range(2 ** len(coeffs))})),
          N_PRESENCE_STATES, tol=0)
    heads = {}
    for node in ast.walk(cgrv_init):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == "StateHead":
            for kw in node.value.keywords:
                if kw.arg == "out_dim":
                    heads[node.targets[0].attr] = _value(
                        kw.value, {**C, "ACTIONS": NF["ACTIONS"]})
    check("verification head width (line 436: a_C or abstain)",
          float(heads.get("primary_head", -1)), 1.0, tol=0)
    check("selection head width (line 437: any candidate or abstain)",
          float(heads.get("recovery_head", -1)), N_ACTIONS, tol=0)
    check("actions the selection head ranks (line 437)",
          float(len(NF["ACTIONS"])), N_ACTIONS, tol=0)
    check("abstain among those actions (line 437)",
          float("abstain" in NF["ACTIONS"]), 1.0, tol=0)

    # ---- the development split and the leakage rule ----------------------
    # Manuscript, lines 451-454: "The encoder remains frozen; development data
    # are used for early stopping, calibration, and cost-specific threshold
    # selection. Gold answers provide training targets but are never selector
    # features."
    print("== selector_code: development-split use and the leakage rule ==")
    fit = _find_def(cgrv_t, "fit", cls="CGRVGate")
    early = int("dev" in [a.arg for a in fit.args.args]
                and any(isinstance(n, ast.Assign)
                        and getattr(n.targets[0], "id", None) == "monitor"
                        and any(isinstance(s, ast.Name) and s.id == "dev"
                                for s in ast.walk(n.value))
                        for n in ast.walk(fit))
                and any(isinstance(n, ast.Constant) and n.value == "patience"
                        for n in ast.walk(fit)))
    calib = int("dev" in [a.arg for a in
                          _find_def(cgrv_t, "calibrate", cls="CGRVGate").args.args])
    sel = _find_def(cgrv_t, "select_threshold", cls="CGRVGate")
    costed = int("c" in [a.arg for a in sel.args.args]
                 and "sweep_threshold" in _calls(sel))
    check("development-split uses the sentence names (lines 452-453)",
          float(early + calib + costed), 3.0, tol=0)
    sweep = _find_def(rel_t, "sweep_threshold")
    check("wrong-answer cost entering the threshold sweep (line 453)",
          float(sum(1 for n in ast.walk(sweep)
                    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
                    and isinstance(n.left, ast.Name) and n.left.id == "c")),
          1.0, tol=0)
    # Gold answers are never selector features: every shipped feature-name
    # table is screened against the same forbidden-substring list.
    forbidden = C["_FORBIDDEN_NAME_SUBSTRINGS"]
    if not forbidden:
        raise SystemExit("selector_code: the forbidden-name list is empty; the "
                         "leakage screen below would be vacuous")
    tables = (obs, GF["FEATURE_NAMES"], z, NF["ACTIONS"])
    check("gold/label-derived names in the feature tables (line 453)",
          float(sum(1 for names in tables for n in names
                    for bad in forbidden if bad in n)), 0.0, tol=0)
    check("feature builders that reject raw rows (InputView guard)",
          float(sum(1 for tree, fn in ((cgrv_t, "observable_features"),
                                       (cgrv_t, "answer_texts"),
                                       (gfeat_t, "build_features"),
                                       (node_t, "_require_view"))
                    if any(isinstance(n, ast.Raise)
                           for n in ast.walk(_find_def(tree, fn))))),
          4.0, tol=0)
    # The guard above is a runtime type test. The claim behind it is STRUCTURAL,
    # and the shipped view settles it: the Row carries the eval-only fields, and
    # the InputView the builders accept does not project them, so no gold-derived
    # quantity is reachable from a feature builder at all.
    row_fields = [n.target.id for n in
                  next(c for c in ast.walk(schema_t)
                       if isinstance(c, ast.ClassDef) and c.name == "Row").body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    slots = _class_const(views_t, "InputView", "__slots__")
    eval_only = [f for f in row_fields if any(bad in f for bad in forbidden)]
    if not eval_only or not slots:
        raise SystemExit("selector_code: the Row carries no eval-only field "
                         "matching the forbidden list, or InputView declares no "
                         "__slots__; the projection check would be vacuous")
    check("eval-only Row fields the InputView projects (line 454)",
          float(sum(1 for f in eval_only if f in slots)), 0.0, tol=0)
    # "Gold answers provide training targets but are never selector features"
    # (lines 453-454), read as two halves over the shipped modules.
    check("selector-input modules reading gold answers (line 454)",
          float(sum(_names_used(ast.parse(text(m)), "gold_answers")
                    for m in INPUT_PATH)), 0.0, tol=0)
    check("gold answers reaching the training targets (line 453)",
          float(_names_used(_find_def(labels_t, "candidate_matches",
                                      cls="LabelBuilder"), "gold_answers")),
          1.0, tol=0)
    # Reported, not asserted: the generic supervised selector (manuscript lines
    # 412-421) has its own, larger feature table and the manuscript prints no
    # count for it, so the number is reported rather than checked. The 18
    # observables checked above are the CGrV ones in cgrv.py, not these.
    print(f"  note  gate_features.FEATURE_NAMES, the generic supervised "
          f"selector's table, has {len(GF['FEATURE_NAMES'])} features; the "
          f"manuscript prints no count for it")

    # ---- the verification-only fixed comparator --------------------------
    # Manuscript, lines 456-464: "a separate four-rule registry: always
    # abstain; return a_C whenever it is available; and two corroboration
    # rules. ... When neither corroborating candidate is available, one rule
    # returns a_C without corroboration, while the stricter rule abstains."
    print("== selector_code: the verification-only fixed comparator ==")
    registry = next(n.value for n in ast.walk(rules_t)
                    if isinstance(n, (ast.Assign, ast.AnnAssign))
                    and getattr(getattr(n, "target", None), "id", None) == "RULES")
    shipped_rules = [k.value for k in registry.keys]
    FOUR = ("always_abstain", "context_only", "current_paper_rule",
            "strict_corroboration")
    check("registry rules the sentence names, found in rules.py (line 457)",
          float(sum(1 for n in FOUR if n in shipped_rules)), 4, tol=0)
    check("uncorroborated rule that still returns a_C (line 463)",
          float("_answer" in _calls(
              _find_def(rules_t, "current_paper_rule").body[-1])), 1.0, tol=0)
    check("stricter rule that abstains uncorroborated (line 464)",
          float("_abstain" in _calls(
              _find_def(rules_t, "strict_corroboration").body[-1])), 1.0, tol=0)
    print(f"  note  rules.py ships {len(shipped_rules)} deterministic rules in "
          f"total; the four above are the verification-only registry")

    # Manuscript, lines 466-473: availability, normalization, agreement.
    abstain = RU["ABSTAIN_SUBSTRINGS"]
    check("refusal wordings the sentence prints (lines 469-470)",
          float(sum(1 for s in REFUSALS_PRINTED if s in abstain)),
          len(REFUSALS_PRINTED), tol=0)
    check("refusal wordings in the availability rule (line 469)",
          float(len(abstain)), len(REFUSALS_PRINTED), tol=0)
    check("shipped modules sharing one refusal list",
          float(sum(1 for env in (RU, NF, GF)
                    if tuple(env.get("ABSTAIN_SUBSTRINGS", ())) == tuple(abstain))),
          3.0, tol=0)
    check("unavailability conditions in candidate() (lines 468-470)",
          float(sum(1 for n in ast.walk(_find_def(rules_t, "candidate"))
                    if isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Constant)
                    and n.value.value is None)), 3.0, tol=0)
    steps = _calls(_find_def(match_t, "normalize"))
    check("normalization steps: strip, lowercase, collapse (line 470)",
          float(sum(1 for s in ("strip", "lower", "sub") if s in steps)),
          3.0, tol=0)
    check("whitespace pattern the normalizer collapses (line 471)",
          float(sum(1 for n in ast.walk(match_t) if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "_WS"
                    and any(isinstance(s, ast.Constant) and s.value == r"\s+"
                            for s in ast.walk(n.value)))), 1.0, tol=0)
    agree_ret = next(n.value for n in ast.walk(_find_def(match_t, "agreement"))
                     if isinstance(n, ast.Return)
                     and isinstance(n.value, ast.BoolOp))
    check("containment directions tested by agreement() (lines 471-473)",
          float(sum(1 for n in ast.walk(agree_ret) if isinstance(n, ast.Compare)
                    and any(isinstance(o, ast.In) for o in n.ops))), 2.0, tol=0)

    # ---- what the shipped config settles, and what it does not -----------
    # Reported, not asserted: protocol_full.yaml carries many development
    # variants of the selector. `methods.cgrv`, checked above, is the block the
    # released panel's decision rows were produced from; the others are
    # ablations and tuning arms and are not the paper's P_3.
    variants = [k for k, v in methods.items()
                if isinstance(v, dict) and v.get("kind") == "cgrv"]
    differing = [k for k in variants
                 if isinstance(methods[k].get("loss_weights"), dict)
                 and (methods[k]["loss_weights"].get("relation") != 0.3
                      or methods[k]["loss_weights"].get("counterfactual") != 0.5
                      or set(methods[k]["loss_weights"]) - set(printed))]
    print(f"  note  protocol_full.yaml carries {len(variants)} cgrv method "
          f"blocks; {len(differing)} of them differ from the printed objective "
          f"and are development variants, not the paper's structured selector")
    # Named explicitly, because a reader who found this block and assumed the
    # tag meant "the model in the paper" would conclude the printed equation and
    # the four-class relation head are contradicted by the authors' own config.
    tagged = [k for k in variants if methods[k].get("final_model") is True]
    if tagged:
        print(f"  note  {', '.join(tagged)} carries `final_model: true` but is "
              f"NOT the arm behind the released rows: it belongs to the earlier "
              f"gate-fix line, turns the typed relation head off and adds "
              f"weighted terms the printed equation does not have. The two "
              f"producers of the panel decision rows read "
              f"`methods['confirmatory_selector']` if present and otherwise "
              f"`methods['cgrv']`; `confirmatory_selector` is not a key in this "
              f"file, so `methods.cgrv` -- checked above -- is what ran")
