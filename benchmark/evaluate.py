"""Score the two model files of a benchmark.py run against the Typed Decisions gold and write results/report.md.

    uv run evaluate.py results/<model-a>-<timestamp>.jsonl results/<model-b>-<timestamp>.jsonl

The report opens with a leaderboard: both models of the run ranked by accuracy among the models on the dataset
card. Then the two models head to head, with intervals, calibration and request statistics.

Metrics, per decision, then averaged:
    accuracy  argmax of the model distribution equals the gold label; a tie of k labels scores 1/k if it holds gold
    kl        KL(gold || model) in nats
    brier     sum over labels of (model - gold)^2
ECE: 10 equal-width bins on the probability of the predicted label. A failed request scores as the uniform
distribution. Intervals: 95% percentile bootstrap over cases, as the five decisions of a case share one state; both
models are resampled with the same cases, so the difference is paired.
"""

import json
import math
import random
import statistics
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRICE_IN = 0.04  # USD per 1M input tokens, both providers; output is not billed
EPS = 1e-6  # probability floor, so KL stays finite
BOOTSTRAP = 2000
METRICS = ("accuracy", "kl", "brier")
BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Dataset card, zero-shot leaderboard, fetched 2026-10-07 from
# https://huggingface.co/datasets/LocalLLaMA/typed-decisions: accuracy, KL, Brier, p50 latency.
# § self-reported. † measured on the model's own machine (M3 Max). ‡ reported by the submitter on their own hardware.
CARD = {
    "meraGPT Decider 1": (0.768, 0.096, 0.052, "526 ms"),
    "Liquid AI d1": (0.742, 0.475, 0.155, "525 ms"),
    "TypeSafe Jev 1.13.0": (0.727, 1.442, 0.148, "710 ms"),
    "Featherless Simple Jev": (0.716, 0.488, 0.176, "–"),
    "prima-ratio + 12B §": (0.702, 0.564, 0.234, "700 ms‡"),
    "OpenDecider-small §": (0.671, 0.211, 0.117, "40 ms‡"),
    "Bongard-mini §": (0.594, 0.256, 0.132, "225 ms‡"),
    "Jeff-Gemma4-E2B": (0.561, 0.403, 0.219, "2,272 ms†"),
    "Jeff-Qwen3.5-2B": (0.511, 0.460, 0.237, "1,346 ms†"),
    "Jeff-Qwen3.5-0.8B": (0.483, 0.679, 0.313, "662 ms†"),
    "Prior": (0.470, 0.347, 0.189, "–"),
    "Uniform": (0.308, 0.444, 0.238, "–"),
}


def labels(question: dict) -> list[str]:
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def distribution(question: dict, answer: dict | None) -> dict[str, float]:
    """The answer as {label: p}, floored and renormalised; uniform if there is no answer."""
    if answer is None:
        p = {label: 1.0 for label in labels(question)}
    elif question["type"] == "noul":
        p = {"true": answer["noul"], "false": 1 - answer["noul"]}
    else:
        p = {label: answer["probabilities"].get(label, 0.0) for label in labels(question)}
    p = {label: max(v, EPS) for label, v in p.items()}
    total = sum(p.values())
    return {label: v / total for label, v in p.items()}


def score(case: dict, answers: dict | None) -> list[dict]:
    out = []
    for name, question in case["questions"].items():
        p = distribution(question, None if answers is None else answers[name])
        gold = case["gold"][name]
        g = gold["probabilities"]
        top = max(p.values())
        tied = [k for k in p if p[k] == top]
        out.append({"case": case["id"], "top": top,
                    "accuracy": (gold["label"] in tied) / len(tied),
                    "kl": sum(g[k] * math.log(g[k] / p[k]) for k in p if g[k] > 0),
                    "brier": sum((p[k] - g[k]) ** 2 for k in p)})
    return out


def means(decisions: list[dict]) -> dict[str, float]:
    return {m: sum(d[m] for d in decisions) / len(decisions) for m in METRICS}


def ece(decisions: list[dict]) -> float:
    bins = {}
    for d in decisions:
        bins.setdefault(min(int(d["top"] * 10), 9), []).append(d)
    return sum(abs(sum(d["top"] - d["accuracy"] for d in b)) for b in bins.values()) / len(decisions)


def interval(values: list[float]) -> str:
    values = sorted(values)
    return f"[{values[int(0.025 * len(values))]:.3f}, {values[int(0.975 * len(values))]:.3f}]"


def table(header: list[str], rows: list[list]) -> list[str]:
    cells = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row] for row in rows]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in cells]


def latencies(rows: list[dict], model: str) -> list[float]:
    return [r["latency_ms"] for r in rows if r["model"] == model and r["answers"] is not None]


def leaderboard(scored: dict[str, list[dict]], rows: list[dict]) -> list[str]:
    """Both models of the run among the card models, ranked by accuracy; the best value of each metric in bold."""
    board = [[f"{m} (this run)", *means(d).values(), f"{statistics.median(latencies(rows, m)):,.0f} ms"]
             for m, d in scored.items()]
    board += [[name, *values] for name, values in CARD.items()]
    board.sort(key=lambda r: -r[1])
    best = {i: (max if m == "accuracy" else min)(r[i] for r in board) for i, m in enumerate(METRICS, 1)}
    cells = [[rank, f"**{r[0]}**" if "(this run)" in r[0] else r[0],
              *(f"**{r[i]:.3f}**" if r[i] == best[i] else f"{r[i]:.3f}" for i in best), r[4]]
             for rank, r in enumerate(board, 1)]
    return table(["#", "Model", "accuracy ↑", "KL ↓", "Brier ↓", "p50 latency"], cells)


def report(cases: list[dict], rows: list[dict]) -> str:
    by_id = {c["id"]: c for c in cases}
    models = list(dict.fromkeys(r["model"] for r in rows))
    a, b = models
    scored = {m: [d for r in rows if r["model"] == m for d in score(by_id[r["id"]], r["answers"])] for m in models}
    uniform = means([d for c in cases for d in score(c, None)])

    per_case = {m: {} for m in models}
    for m, decisions in scored.items():
        for d in decisions:
            per_case[m].setdefault(d["case"], []).append(d)
    rng, ids = random.Random(0), [c["id"] for c in cases]
    samples = []
    for _ in range(BOOTSTRAP):
        drawn = rng.choices(ids, k=len(ids))
        samples.append({m: means([d for c in drawn for d in per_case[m][c]]) for m in models})

    out = [f"# {a} vs. {b} on Typed Decisions", "",
           f"{len(cases)} cases, {len(scored[a])} decisions per model.", "",
           "## Leaderboard", ""]
    out += leaderboard(scored, rows)
    out += ["", ("Card: zero-shot leaderboard of the dataset card, same test split; models fitted on `train` are left "
                 "out, as the card does. Best value per metric in bold. Whether a gap to a card model is real: see the "
                 "intervals below. Latency: p50 end to end per case from a client, one request at a time; this run "
                 "from one machine with a kept-alive connection. § self-reported. † measured on the model's own "
                 "machine (M3 Max) and ‡ reported by the submitter on their own hardware, both not comparable with "
                 "hosted latency.")]

    out += ["", "## Head to head", "", "95% bootstrap intervals in brackets.", ""]
    result = [[m, *(f"{means(scored[m])[k]:.3f} {interval([s[m][k] for s in samples])}" for k in METRICS)]
              for m in models]
    result.append([f"Δ {b} − {a}", *(f"{means(scored[b])[k] - means(scored[a])[k]:+.3f} "
                                      f"{interval([s[b][k] - s[a][k] for s in samples])}" for k in METRICS)])
    result.append(["uniform (this code)", uniform["accuracy"], uniform["kl"], uniform["brier"]])
    out += table(["Model", *METRICS], result)
    out += ["", ("An interval of Δ that excludes 0 is a real difference. uniform (this code): the uniform distribution "
                 "scored by evaluate.py, to check the metrics against the card's Uniform row (its accuracy 0.308 "
                 "breaks ties differently; 1/k credit gives the expected value).")]

    out += ["", "## Calibration", "", "Decisions binned by the probability of the predicted label.", ""]
    calibration = []
    for lo, hi in pairwise(BINS):
        for m in models:
            hit = [d for d in scored[m] if lo <= d["top"] < hi or (hi == 1.0 and d["top"] == 1.0)]
            calibration.append([f"{lo:.1f}–{hi:.1f}", m, len(hit),
                                *((statistics.mean(d["top"] for d in hit), statistics.mean(d["accuracy"] for d in hit))
                                  if hit else ("–", "–"))])
    out += table(["p", "Model", "decisions", "mean p", "accuracy"], calibration)
    out += ["", "ECE (10 equal-width bins): " + ", ".join(f"{m} {ece(scored[m]):.3f}" for m in models)
            + ". The card's ECE definitions differ by submitter; compare the two models here, not with the card."]

    out += ["", "## Time per sample", "", ("One sample is one request with all five questions, client-side from the "
                                             "first attempt, ms. Failed requests are left out."), ""]
    timing = []
    for m in models:
        ms = sorted(latencies(rows, m))
        timing.append([m, f"{statistics.mean(ms):,.0f}", f"{statistics.median(ms):,.0f}",
                       f"{ms[int(0.9 * len(ms))]:,.0f}", f"{ms[int(0.99 * len(ms))]:,.0f}", f"{ms[-1]:,.0f}"])
    out += table(["Model", "mean", "p50", "p90", "p99", "max"], timing)

    out += ["", "## Requests", "", "One request per case with five questions.", ""]
    requests = []
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        ok = [r for r in mine if r["answers"] is not None]
        requests.append([m, len(mine), len(mine) - len(ok), sum(r["attempts"] > 1 for r in mine),
                         f"{statistics.mean(r['input_tokens'] for r in ok) * PRICE_IN / 1e3:.3f}"])
    out += table(["Model", "requests", "failed", "retried", "USD / 1,000 cases"], requests)
    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: uv run evaluate.py results/<model-a>-<timestamp>.jsonl results/<model-b>-<timestamp>.jsonl")
    rows = [json.loads(line) for path in sys.argv[1:] for line in Path(path).read_text().splitlines()]
    count = Counter(r["id"] for r in rows)
    ids = {i for i, n in count.items() if n == 2}  # cases answered by both models, if the run was cut short
    rows = [r for r in rows if r["id"] in ids]
    cases = [json.loads(line) for line in (ROOT / "data" / "test.jsonl").read_text().splitlines()]
    text = report([c for c in cases if c["id"] in ids], rows)
    (ROOT / "results" / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
