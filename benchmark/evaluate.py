"""
Evaluate benchmark predictions and generate model comparison reports.

Loads all prediction files from benchmark/results/, scores each model
against gold answers, and prints a side-by-side comparison table.

Usage:
    python -m benchmark.evaluate
    python -m benchmark.evaluate --results-dir benchmark/results --save-report
"""

import json
import math
import os
import argparse
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_TEXT_TO_INT = {
    "not_eligible": 0,
    "partially_eligible": 1,
    "eligible": 2,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["questions"]}


def load_all_predictions(results_dir):
    """Return {model_name: {qid: prediction_dict}}."""
    models = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        name = data["metadata"]["model"]
        preds = {p["question_id"]: p for p in data["predictions"]}
        models[name] = preds
    return models


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _dcg(scores, k):
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(predicted_scores, k):
    ideal = sorted(predicted_scores, reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(predicted_scores, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Eligibility scoring
# ---------------------------------------------------------------------------

def score_eligibility(questions, preds):
    correct = total = 0
    per_label = defaultdict(lambda: {"tp": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for qid, pred in preds.items():
        q = questions.get(qid)
        if not q or q["type"] != "eligibility" or not pred.get("parse_success"):
            continue

        gold = q["gold_label"]
        decision = pred["parsed_response"].get("decision", "").strip().lower()
        pred_label = LABEL_TEXT_TO_INT.get(decision, -1)
        if pred_label == -1:
            continue

        total += 1
        per_label[gold]["total"] += 1
        confusion[gold][pred_label] += 1
        if pred_label == gold:
            correct += 1
            per_label[gold]["tp"] += 1

    acc = correct / total if total else 0

    # Weighted accuracy (partially_eligible is harder, gets 1.5x weight)
    weights = {0: 1.0, 1: 1.5, 2: 1.0}
    w_num = sum(per_label[l]["tp"] * weights.get(l, 1) for l in per_label)
    w_den = sum(per_label[l]["total"] * weights.get(l, 1) for l in per_label)
    wacc = w_num / w_den if w_den else 0

    # Binary: eligible(2) vs not-eligible(0+1)
    b_ok = b_tot = 0
    for qid, pred in preds.items():
        q = questions.get(qid)
        if not q or q["type"] != "eligibility" or not pred.get("parse_success"):
            continue
        gold = q["gold_label"]
        decision = pred["parsed_response"].get("decision", "").strip().lower()
        pl = LABEL_TEXT_TO_INT.get(decision, -1)
        if pl == -1:
            continue
        b_tot += 1
        if (gold == 2) == (pl == 2):
            b_ok += 1
    bin_acc = b_ok / b_tot if b_tot else 0

    return {
        "accuracy": _pct(acc),
        "weighted_accuracy": _pct(wacc),
        "binary_eligible_accuracy": _pct(bin_acc),
        "evaluated": total,
        "per_label": {str(k): v for k, v in per_label.items()},
        "confusion": {str(k): dict(v) for k, v in confusion.items()},
    }


# ---------------------------------------------------------------------------
# Ranking scoring
# ---------------------------------------------------------------------------

def score_ranking(questions, preds):
    ndcg_vals = []
    concordance_vals = []

    for qid, pred in preds.items():
        q = questions.get(qid)
        if not q or q["type"] != "ranking" or not pred.get("parse_success"):
            continue

        gold_scores = q["gold_scores"]
        trial_map = q["trial_map"]
        ranking = pred["parsed_response"].get("ranking", [])
        if not ranking:
            continue

        rel_seq = []
        valid = True
        for letter in ranking:
            letter = letter.strip().upper()
            nctid = trial_map.get(letter)
            if nctid is None:
                valid = False
                break
            rel_seq.append(gold_scores.get(nctid, 0))
        if not valid or not rel_seq:
            continue

        k = min(5, len(rel_seq))
        ndcg_vals.append(ndcg_at_k(rel_seq, k))

        conc = disc = 0
        for i in range(len(rel_seq)):
            for j in range(i + 1, len(rel_seq)):
                if rel_seq[i] > rel_seq[j]:
                    conc += 1
                elif rel_seq[i] < rel_seq[j]:
                    disc += 1
        pairs = conc + disc
        if pairs:
            concordance_vals.append(conc / pairs)

    return {
        "ndcg_at_5": _pct(_mean(ndcg_vals)),
        "concordance": _pct(_mean(concordance_vals)),
        "evaluated": len(ndcg_vals),
    }


# ---------------------------------------------------------------------------
# Criterion analysis scoring
# ---------------------------------------------------------------------------

def score_criterion(questions, preds):
    expected = responded = reasoned = evidenced = 0

    for qid, pred in preds.items():
        q = questions.get(qid)
        if not q or q["type"] != "criterion_analysis" or not pred.get("parse_success"):
            continue

        assessments = pred["parsed_response"].get("assessments", {})
        expected += len(q.get("criteria", []))
        responded += len(assessments)

        for _, a in assessments.items():
            if isinstance(a, dict):
                if a.get("reasoning"):
                    reasoned += 1
                ev = a.get("evidence", "")
                if ev and ev.lower() not in ("not mentioned", "n/a", "none", ""):
                    evidenced += 1

    return {
        "coverage": _pct(responded / expected if expected else 0),
        "reasoning_rate": _pct(reasoned / responded if responded else 0),
        "evidence_rate": _pct(evidenced / responded if responded else 0),
        "criteria_expected": expected,
        "criteria_assessed": responded,
    }


# ---------------------------------------------------------------------------
# Missing-info scoring
# ---------------------------------------------------------------------------

def score_missing_info(questions, preds):
    total = 0
    item_counts = []
    has_rec = has_comp = 0
    importance = defaultdict(int)

    for qid, pred in preds.items():
        q = questions.get(qid)
        if not q or q["type"] != "missing_info" or not pred.get("parse_success"):
            continue

        total += 1
        parsed = pred["parsed_response"]
        items = parsed.get("missing_items", [])
        item_counts.append(len(items))

        for it in items:
            if isinstance(it, dict):
                importance[it.get("importance", "unknown")] += 1

        if parsed.get("recommendation"):
            has_rec += 1
        if parsed.get("overall_completeness") is not None:
            has_comp += 1

    return {
        "avg_items_found": round(_mean(item_counts), 1),
        "recommendation_rate": _pct(has_rec / total if total else 0),
        "completeness_rate": _pct(has_comp / total if total else 0),
        "importance_dist": dict(importance),
        "evaluated": total,
    }


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------

def overall_score(elig, rank, crit, miss):
    """Weighted combination — adjust weights to taste."""
    components = []
    if elig["evaluated"]:
        components.append((elig["weighted_accuracy"], 0.40))
    if rank["evaluated"]:
        components.append((rank["ndcg_at_5"], 0.25))
    if crit["criteria_expected"]:
        components.append((crit["coverage"], 0.20))
    if miss["evaluated"]:
        components.append((miss["recommendation_rate"], 0.15))

    if not components:
        return 0.0
    tw = sum(w for _, w in components)
    return round(sum(s * w for s, w in components) / tw, 1)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results):
    models = sorted(results.keys())
    col_w = max(20, max(len(m) for m in models) + 2)

    hdr = f"{'Metric':<40}" + "".join(f"{m:>{col_w}}" for m in models)
    sep = "=" * len(hdr)

    print(f"\n{sep}")
    print("  CLINICAL TRIAL RECRUITMENT BENCHMARK")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    rows = [
        ("OVERALL SCORE", lambda r: r["overall"]),
        None,
        ("Eligibility — Accuracy (%)", lambda r: r["eligibility"]["accuracy"]),
        ("Eligibility — Weighted Acc (%)", lambda r: r["eligibility"]["weighted_accuracy"]),
        ("Eligibility — Binary Acc (%)", lambda r: r["eligibility"]["binary_eligible_accuracy"]),
        ("  questions evaluated", lambda r: r["eligibility"]["evaluated"]),
        None,
        ("Ranking — NDCG@5 (%)", lambda r: r["ranking"]["ndcg_at_5"]),
        ("Ranking — Concordance (%)", lambda r: r["ranking"]["concordance"]),
        ("  questions evaluated", lambda r: r["ranking"]["evaluated"]),
        None,
        ("Criteria — Coverage (%)", lambda r: r["criterion"]["coverage"]),
        ("Criteria — Reasoning Rate (%)", lambda r: r["criterion"]["reasoning_rate"]),
        ("Criteria — Evidence Rate (%)", lambda r: r["criterion"]["evidence_rate"]),
        None,
        ("Missing Info — Avg Items Found", lambda r: r["missing_info"]["avg_items_found"]),
        ("Missing Info — Rec. Rate (%)", lambda r: r["missing_info"]["recommendation_rate"]),
        None,
        ("JSON Parse Success (%)", lambda r: r["parse_rate"]),
    ]

    for row in rows:
        if row is None:
            print()
            continue
        label, getter = row
        line = f"  {label:<38}"
        for m in models:
            val = getter(results[m])
            line += f"{val:>{col_w}}"
        print(line)

    print(sep)
    best = max(models, key=lambda m: results[m]["overall"])
    print(f"\n  Best model: {best}  (score {results[best]['overall']})")
    print()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pct(v):
    return round(v * 100, 1) if v <= 1 else round(v, 1)

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--questions",
                        default=os.path.join(BASE_DIR, "benchmark", "data",
                                             "questions.json"))
    parser.add_argument("--results-dir",
                        default=os.path.join(BASE_DIR, "benchmark", "results"))
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    print("Loading questions ...")
    questions = load_questions(args.questions)
    print(f"  {len(questions)} questions")

    print("Loading predictions ...")
    all_models = load_all_predictions(args.results_dir)
    if not all_models:
        print("No prediction files found in", args.results_dir)
        return
    print(f"  models: {', '.join(all_models.keys())}")

    results = {}
    for model_name, preds in all_models.items():
        print(f"Scoring {model_name} ...")
        elig = score_eligibility(questions, preds)
        rank = score_ranking(questions, preds)
        crit = score_criterion(questions, preds)
        miss = score_missing_info(questions, preds)
        ov = overall_score(elig, rank, crit, miss)
        parse_ok = sum(1 for p in preds.values() if p.get("parse_success"))
        parse_rate = _pct(parse_ok / len(preds)) if preds else 0

        results[model_name] = {
            "overall": ov,
            "eligibility": elig,
            "ranking": rank,
            "criterion": crit,
            "missing_info": miss,
            "parse_rate": parse_rate,
        }

    print_report(results)

    if args.save_report:
        report_dir = os.path.join(BASE_DIR, "benchmark", "reports")
        os.makedirs(report_dir, exist_ok=True)
        out = os.path.join(report_dir, "evaluation_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed report saved -> {out}")


if __name__ == "__main__":
    main()
