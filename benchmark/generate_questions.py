"""
Generate benchmark questions from TrialGPT datasets.

Reads retrieved_trials.json (patient-trial pairs with gold relevance labels)
and produces a standardized question bank with 4 question types:

  1. eligibility        — Is this patient eligible for this trial? (3-class)
  2. ranking            — Rank these trials for this patient (ordering)
  3. criterion_analysis — Assess each criterion individually (per-criterion)
  4. missing_info       — What info is missing to determine eligibility?

Usage:
    python -m benchmark.generate_questions
    python -m benchmark.generate_questions --dataset trec_2021 --seed 42
"""

import json
import os
import random
import argparse
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

LABEL_MAP = {0: "not_eligible", 1: "partially_eligible", 2: "eligible"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset="trec_2022"):
    dataset_path = os.path.join(DATASET_DIR, dataset)

    with open(os.path.join(dataset_path, "retrieved_trials.json"), encoding="utf-8") as f:
        retrieved = json.load(f)

    qrels = {}
    with open(os.path.join(dataset_path, "qrels", "test.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, nctid, score_str = parts[0], parts[1], parts[2]
            try:
                score = int(score_str)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[nctid] = score

    return retrieved, qrels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_trial(trial):
    """Full trial description for prompts."""
    t = f"Title: {trial['brief_title']}\n"
    t += f"Phase: {trial.get('phase') or 'Not specified'}\n"
    t += f"Target Conditions: {', '.join(trial.get('diseases_list', []))}\n"
    t += f"Interventions: {', '.join(trial.get('drugs_list', []))}\n"
    t += f"Summary: {trial['brief_summary']}\n\n"
    t += f"Inclusion Criteria:\n{trial.get('inclusion_criteria', 'N/A')}\n\n"
    t += f"Exclusion Criteria:\n{trial.get('exclusion_criteria', 'N/A')}"
    return t


def parse_criteria_list(criteria_text):
    """Split a criteria block into individual criterion strings."""
    if not criteria_text:
        return []
    out = []
    for p in criteria_text.split("\n\n"):
        p = p.strip()
        if len(p) < 5:
            continue
        if "inclusion criteria" in p.lower() or "exclusion criteria" in p.lower():
            continue
        out.append(p)
    return out


def count_criteria(text):
    return len(parse_criteria_list(text))


# ---------------------------------------------------------------------------
# Question type 1 — Eligibility determination (3-class)
# ---------------------------------------------------------------------------

def generate_eligibility_questions(retrieved, per_patient=12, seed=42):
    random.seed(seed)
    questions = []

    for patient in retrieved:
        pid = patient["patient_id"]
        note = patient["patient"]

        for label_str in ["0", "1", "2"]:
            trials = patient.get(label_str, [])
            if not trials:
                continue
            label = int(label_str)
            n = min(len(trials), max(1, per_patient // 3))
            sampled = random.sample(trials, n)

            for trial in sampled:
                trial_text = format_trial(trial)

                prompt = (
                    "You are a clinical trial recruitment specialist. "
                    "Review the patient record and clinical trial below, "
                    "then determine the patient's eligibility.\n\n"
                    f"PATIENT RECORD:\n{note}\n\n"
                    f"CLINICAL TRIAL:\n{trial_text}\n\n"
                    "Assess ALL inclusion and exclusion criteria, then classify:\n"
                    '- "eligible": Patient meets key inclusion criteria and does not trigger exclusion criteria\n'
                    '- "partially_eligible": Some criteria cannot be confirmed, or patient is borderline\n'
                    '- "not_eligible": Patient clearly fails critical criteria\n\n'
                    "Respond with ONLY a JSON object:\n"
                    '{"reasoning": "your clinical reasoning (2-4 sentences)", '
                    '"decision": "eligible" | "partially_eligible" | "not_eligible"}'
                )

                questions.append({
                    "id": f"elig_{pid}_{trial['NCTID']}",
                    "type": "eligibility",
                    "patient_id": pid,
                    "trial_id": trial["NCTID"],
                    "prompt": prompt,
                    "gold_label": label,
                    "gold_label_text": LABEL_MAP[label],
                    "metadata": {
                        "n_inclusion": count_criteria(trial.get("inclusion_criteria", "")),
                        "n_exclusion": count_criteria(trial.get("exclusion_criteria", "")),
                        "note_length": len(note),
                    },
                })

    return questions


# ---------------------------------------------------------------------------
# Question type 2 — Ranking (order N trials for a patient)
# ---------------------------------------------------------------------------

def generate_ranking_questions(retrieved, n_trials=5, seed=42):
    random.seed(seed)
    questions = []

    for patient in retrieved:
        pid = patient["patient_id"]
        note = patient["patient"]

        by_label = defaultdict(list)
        for label_str in ["0", "1", "2"]:
            for trial in patient.get(label_str, []):
                by_label[int(label_str)].append(trial)

        if sum(len(v) for v in by_label.values()) < 3:
            continue

        # Pick a diverse set: at least 1 from each available label
        selected = []
        gold_scores = {}
        for label in sorted(by_label.keys()):
            pick = random.sample(by_label[label], min(2, len(by_label[label])))
            for t in pick:
                selected.append(t)
                gold_scores[t["NCTID"]] = label

        if len(selected) > n_trials:
            random.shuffle(selected)
            selected = selected[:n_trials]
            gold_scores = {t["NCTID"]: gold_scores[t["NCTID"]] for t in selected}
        elif len(selected) < n_trials:
            remaining = []
            for label_str in ["0", "1", "2"]:
                for t in patient.get(label_str, []):
                    if t["NCTID"] not in gold_scores:
                        remaining.append((t, int(label_str)))
            random.shuffle(remaining)
            for t, lbl in remaining[: n_trials - len(selected)]:
                selected.append(t)
                gold_scores[t["NCTID"]] = lbl

        random.shuffle(selected)

        trial_descriptions = ""
        trial_map = {}
        for i, trial in enumerate(selected):
            letter = chr(65 + i)
            trial_map[letter] = trial["NCTID"]
            trial_descriptions += (
                f"\nTRIAL {letter}:\n"
                f"  Title: {trial['brief_title']}\n"
                f"  Conditions: {', '.join(trial.get('diseases_list', []))}\n"
                f"  Interventions: {', '.join(trial.get('drugs_list', []))}\n"
                f"  Phase: {trial.get('phase') or 'Not specified'}\n"
                f"  Summary: {trial['brief_summary']}\n"
                f"  Inclusion Criteria: {trial.get('inclusion_criteria') or 'N/A'}\n"
                f"  Exclusion Criteria: {trial.get('exclusion_criteria') or 'N/A'}\n"
            )

        prompt = (
            "You are a clinical trial recruitment specialist. "
            "Rank the following trials from MOST to LEAST suitable "
            "for this patient. Consider disease relevance, eligibility "
            "likelihood, and overall clinical fit.\n\n"
            f"PATIENT RECORD:\n{note}\n\n"
            f"CLINICAL TRIALS:{trial_descriptions}\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"reasoning": "brief ranking rationale", '
            '"ranking": ["X", "Y", ...]}'
            " where letters are ordered most → least suitable."
        )

        gold_ranking = sorted(
            trial_map.keys(), key=lambda l: -gold_scores[trial_map[l]]
        )

        questions.append({
            "id": f"rank_{pid}",
            "type": "ranking",
            "patient_id": pid,
            "prompt": prompt,
            "trial_map": trial_map,
            "gold_scores": gold_scores,
            "gold_ranking": gold_ranking,
            "metadata": {
                "n_trials": len(selected),
                "label_dist": {
                    str(l): sum(1 for s in gold_scores.values() if s == l)
                    for l in [0, 1, 2]
                },
            },
        })

    return questions


# ---------------------------------------------------------------------------
# Question type 3 — Criterion-level analysis
# ---------------------------------------------------------------------------

def generate_criterion_questions(retrieved, per_patient=3, seed=42):
    random.seed(seed)
    questions = []

    for patient in retrieved:
        pid = patient["patient_id"]
        note = patient["patient"]

        # Prefer partially-eligible trials — most interesting for criterion analysis
        candidates = []
        for label_str in ["1", "2", "0"]:
            for t in patient.get(label_str, []):
                candidates.append((t, int(label_str)))

        if not candidates:
            continue

        sampled = random.sample(candidates, min(per_patient, len(candidates)))

        for trial, label in sampled:
            inc = parse_criteria_list(trial.get("inclusion_criteria", ""))
            exc = parse_criteria_list(trial.get("exclusion_criteria", ""))
            if not inc and not exc:
                continue

            criteria_block = ""
            criteria_list = []
            idx = 1
            for c in inc:
                criteria_block += f"{idx}. [INCLUSION] {c}\n"
                criteria_list.append({"index": idx, "type": "inclusion", "text": c})
                idx += 1
            for c in exc:
                criteria_block += f"{idx}. [EXCLUSION] {c}\n"
                criteria_list.append({"index": idx, "type": "exclusion", "text": c})
                idx += 1

            prompt = (
                "You are a clinical trial recruitment specialist. "
                "For each criterion below, assess whether the patient "
                "meets it based on the available information.\n\n"
                f"PATIENT RECORD:\n{note}\n\n"
                f"TRIAL: {trial['brief_title']}\n"
                f"Conditions: {', '.join(trial.get('diseases_list', []))}\n\n"
                f"CRITERIA TO ASSESS:\n{criteria_block}\n"
                "For each criterion choose one status:\n"
                '  "met" — patient clearly satisfies this criterion\n'
                '  "not_met" — patient clearly does NOT satisfy it\n'
                '  "insufficient_info" — not enough info in the patient record\n'
                '  "not_applicable" — criterion does not apply to this patient\n\n'
                "Respond with ONLY a JSON object:\n"
                '{"assessments": {"1": {"status": "...", "reasoning": "...", '
                '"evidence": "relevant quote or \'not mentioned\'"}, "2": {...}, ...}}'
            )

            questions.append({
                "id": f"crit_{pid}_{trial['NCTID']}",
                "type": "criterion_analysis",
                "patient_id": pid,
                "trial_id": trial["NCTID"],
                "prompt": prompt,
                "criteria": criteria_list,
                "gold_eligibility": label,
                "metadata": {
                    "n_criteria": len(criteria_list),
                    "n_inclusion": len(inc),
                    "n_exclusion": len(exc),
                },
            })

    return questions


# ---------------------------------------------------------------------------
# Question type 4 — Missing information identification
# ---------------------------------------------------------------------------

def generate_missing_info_questions(retrieved, per_patient=2, seed=42):
    random.seed(seed)
    questions = []

    for patient in retrieved:
        pid = patient["patient_id"]
        note = patient["patient"]

        # Use partially-eligible trials — most likely to have genuine info gaps
        targets = patient.get("1", []) or patient.get("2", [])
        if not targets:
            continue

        sampled = random.sample(targets, min(per_patient, len(targets)))

        for trial in sampled:
            trial_text = format_trial(trial)

            prompt = (
                "You are a clinical trial recruitment specialist doing a "
                "pre-screening chart review. Identify what critical "
                "information is MISSING from this patient's record that "
                "you would need to make a definitive eligibility determination.\n\n"
                f"PATIENT RECORD:\n{note}\n\n"
                f"CLINICAL TRIAL:\n{trial_text}\n\n"
                "List the missing items, which criterion needs each, "
                "and how critical each gap is.\n\n"
                "Respond with ONLY a JSON object:\n"
                '{"missing_items": [{"criterion": "...", "what_is_needed": "...", '
                '"importance": "critical | important | minor"}], '
                '"overall_completeness": <0-100>, '
                '"recommendation": "proceed_with_screening | need_more_info | likely_ineligible"}'
            )

            questions.append({
                "id": f"miss_{pid}_{trial['NCTID']}",
                "type": "missing_info",
                "patient_id": pid,
                "trial_id": trial["NCTID"],
                "prompt": prompt,
                "gold_eligibility": 1,
                "metadata": {
                    "note_length": len(note),
                    "n_criteria": count_criteria(trial.get("inclusion_criteria", ""))
                    + count_criteria(trial.get("exclusion_criteria", "")),
                },
            })

    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the benchmark question bank"
    )
    parser.add_argument("--dataset", default="trec_2021")
    parser.add_argument("--output", default=None)
    parser.add_argument("--elig-per-patient", type=int, default=12)
    parser.add_argument("--crit-per-patient", type=int, default=3)
    parser.add_argument("--miss-per-patient", type=int, default=2)
    parser.add_argument("--rank-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        out_dir = os.path.join(BASE_DIR, "benchmark", "data")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, "questions.json")

    print(f"Loading {args.dataset} ...")
    retrieved, qrels = load_data(args.dataset)
    print(f"  {len(retrieved)} patients, "
          f"{sum(len(v) for v in qrels.values())} qrel judgments")


    generators = [
        ("eligibility", generate_eligibility_questions,
         {"per_patient": args.elig_per_patient, "seed": args.seed}),
        ("ranking", generate_ranking_questions,
         {"n_trials": args.rank_trials, "seed": args.seed}),
        ("criterion_analysis", generate_criterion_questions,
         {"per_patient": args.crit_per_patient, "seed": args.seed}),
        ("missing_info", generate_missing_info_questions,
         {"per_patient": args.miss_per_patient, "seed": args.seed}),
    ]

    all_questions = []
    by_type = {}

    for name, fn, kwargs in generators:
        print(f"Generating {name} questions ...")
        qs = fn(retrieved, **kwargs)
        all_questions.extend(qs)
        by_type[name] = len(qs)
        print(f"  -> {len(qs)}")

    output = {
        "metadata": {
            "dataset": args.dataset,
            "seed": args.seed,
            "total_questions": len(all_questions),
            "by_type": by_type,
        },
        "questions": all_questions,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_questions)} questions -> {args.output}")
    for name, count in by_type.items():
        print(f"    {name}: {count}")


if __name__ == "__main__":
    main()
