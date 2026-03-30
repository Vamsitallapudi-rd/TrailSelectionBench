"""
Run benchmark questions against an LLM and collect predictions.

Supports Azure OpenAI, OpenAI, Anthropic, and Google Gemini providers.
Runs questions in parallel using a thread pool (default 40 workers).
Saves results incrementally so interrupted runs can be resumed.

Usage:
    python -m benchmark.run_benchmark --model gpt-4o --provider azure_openai
    python -m benchmark.run_benchmark --model claude-3-5-sonnet-20241022 --provider anthropic
    python -m benchmark.run_benchmark --model gemini-2.5-pro-preview-05-06 --provider gemini
    python -m benchmark.run_benchmark --model gpt-4o --provider openai --no-resume
    python -m benchmark.run_benchmark --model gpt-4o --provider azure_openai --workers 20
"""

import json
import os
import time
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM_MSG = (
    "You are a clinical trial recruitment specialist. "
    "Always respond with valid JSON exactly as instructed."
)

MAX_TOKENS = 5000


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

def create_client(provider, **kwargs):
    if provider == "azure_openai":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_version=kwargs.get("api_version", "2024-02-15-preview"),
            azure_endpoint=os.environ["OPENAI_ENDPOINT"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_model(client, provider, model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            if provider in ("azure_openai", "openai"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                )
                return resp.choices[0].message.content.strip()

            elif provider == "anthropic":
                resp = client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=0,
                    system=SYSTEM_MSG,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()

            elif provider == "gemini":
                from google.genai import types
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_MSG,
                        temperature=0,
                        max_output_tokens=MAX_TOKENS,
                    ),
                )
                return resp.text.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Warning: {e}  -- retrying in {wait}s ...")
                time.sleep(wait)
            else:
                return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}, False


# ---------------------------------------------------------------------------
# Single-question worker (called inside thread pool)
# ---------------------------------------------------------------------------

def _process_question(client, provider, model, q):
    """Run one question through the model and return a prediction dict."""
    raw = call_model(client, provider, model, q["prompt"])
    parsed, ok = parse_response(raw)
    return {
        "question_id": q["id"],
        "question_type": q["type"],
        "model": model,
        "raw_response": raw,
        "parsed_response": parsed,
        "parse_success": ok,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(questions_path, model, provider, output_path, resume=True,
                  max_workers=40, save_every=50):
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    existing = {}
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            prev = json.load(f)
        for pred in prev.get("predictions", []):
            existing[pred["question_id"]] = pred
        print(f"Resuming -- {len(existing)} predictions already on disk")

    todo = [q for q in questions if q["id"] not in existing]
    total = len(questions)
    print(f"Questions: {total} total, {len(existing)} done, {len(todo)} remaining")
    print(f"Provider: {provider}  |  Model: {model}  |  Workers: {max_workers}")

    if not todo:
        print("Nothing to do.")
        return

    client = create_client(provider)

    predictions = list(existing.values())
    lock = threading.Lock()
    completed = 0

    def on_done(future):
        nonlocal completed
        pred = future.result()
        with lock:
            predictions.append(pred)
            completed += 1
            status = "ok" if pred["parse_success"] else "PARSE_FAIL"
            print(f"  [{completed}/{len(todo)}] {pred['question_type']:20s} "
                  f"{pred['question_id']}  [{status}]")
            if completed % save_every == 0:
                _save(output_path, model, provider, predictions)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for q in todo:
            fut = pool.submit(_process_question, client, provider, model, q)
            fut.add_done_callback(on_done)
            futures.append(fut)

        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"  Unexpected thread error: {exc}")

    _save(output_path, model, provider, predictions)
    parse_ok = sum(1 for p in predictions if p["parse_success"])
    print(f"\nDone -- {len(predictions)} predictions "
          f"({parse_ok} parsed, {len(predictions) - parse_ok} failed) "
          f"-> {output_path}")


def _save(path, model, provider, predictions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {
        "metadata": {
            "model": model,
            "provider": provider,
            "n_predictions": len(predictions),
            "n_parse_failures": sum(
                1 for p in predictions if not p["parse_success"]
            ),
            "last_updated": datetime.now().isoformat(),
        },
        "predictions": predictions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the recruitment benchmark against an LLM"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. gpt-4o, claude-3-5-sonnet-20241022, "
                             "gemini-2.5-pro-preview-05-06)")
    parser.add_argument("--provider", required=True,
                        choices=["azure_openai", "openai", "anthropic", "gemini"])
    parser.add_argument("--questions",
                        default=os.path.join(BASE_DIR, "benchmark", "data",
                                             "questions.json"))
    parser.add_argument("--output", default=None,
                        help="Output path (default: benchmark/results/<model>.json)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=40,
                        help="Max parallel threads (default: 40)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N completions (default: 50)")
    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(BASE_DIR, "benchmark", "results")
        safe = args.model.replace("/", "_").replace(":", "_")
        args.output = os.path.join(results_dir, f"{safe}.json")

    run_benchmark(
        args.questions,
        args.model,
        args.provider,
        args.output,
        resume=not args.no_resume,
        max_workers=args.workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
