"""Run celeris-1-decision and then mercury-decide on the Typed Decisions test split.

    uv run benchmark.py

Both providers speak the JEV API, so each case's `state` and `questions` are sent unchanged as one
`POST /v1/systemone` through the JEV SDK (`typesafe-sdk`), one kept-alive HTTPS connection per provider. First
celeris-1-decision answers every case, then mercury-decide. A failed request, or an answer that misses a
question, is retried up to three times (the SDK's own retries are off); latency counts from the first attempt.
Rows are appended to results/<model>-<timestamp>.jsonl, one file per model, as they come in. API keys are read from ../.env (repo root).
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

ROOT = Path(__file__).resolve().parent
ATTEMPTS = 3
PROVIDERS = {  # model: (base URL, API key in ../.env (repo root))
    "celeris-1-decision": ("https://inference.celeris.ai/celeris-1-decision", "CELERIS_API_KEY"),
    "mercury-decide": ("https://api.inceptionlabs.ai", "INCEPTION_API_KEY"),
}


def run(client: TypeSafeClient, model: str, case: dict) -> dict:
    row = {"id": case["id"], "model": model, "answers": None, "errors": []}
    t0 = time.perf_counter()
    for row["attempts"] in range(1, ATTEMPTS + 1):
        try:
            response = client.system_one(state=case["state"], questions=case["questions"])
            if set(response.answers) != set(case["questions"]):
                raise ValueError(f"answered {sorted(response.answers)}")
            row["answers"] = {name: a.model_dump(mode="json") for name, a in response.answers.items()}
            row["input_tokens"] = response.usage.input_tokens
            break
        except (TypeSafeError, ValueError) as e:
            row["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
    row["latency_ms"] = (time.perf_counter() - t0) * 1000
    return row


def main() -> None:
    load_dotenv(ROOT.parent / ".env")
    clients = {
        model: TypeSafeClient(api_key=os.environ[key], base_url=url, model=model, timeout=15,
                              retry=RetryPolicy(max_retries=0))
        for model, (url, key) in PROVIDERS.items()
    }
    cases = [json.loads(line) for line in (ROOT / "data" / "test.jsonl").read_text().splitlines()]
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    (ROOT / "results").mkdir(exist_ok=True)
    for model, client in clients.items():
        out = ROOT / "results" / f"{model}-{stamp}.jsonl"
        with out.open("w") as f:
            for i, case in enumerate(cases):
                row = run(client, model, case)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{i + 1:4}/{len(cases)}  {model:<20} {case['id']:<36} {row['latency_ms']:5.0f} ms  "
                      f"{row['errors'] or ''}")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
