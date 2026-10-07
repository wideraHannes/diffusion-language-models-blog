# Benchmark: diffusion models as decision models

celeris-1-decision vs. mercury-decide on the test split of
[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (`data/test.jsonl`, 400 cases).

API keys `CELERIS_API_KEY` and `INCEPTION_API_KEY` go in the `.env` at the repo root.

```bash
uv run benchmark.py    # writes results/<model>-<timestamp>.jsonl, one file per model
uv run evaluate.py results/celeris-1-decision-<timestamp>.jsonl results/mercury-decide-<timestamp>.jsonl
                       # writes results/report.md
```

The report ranks both models among the models on the dataset card, then compares the two with bootstrap intervals,
calibration and latency.

## Metrics

Each question has a gold distribution over its labels; the model answers with a probability per label.

| Metric | Definition | Better |
|---|---|---|
| accuracy | Share of decisions where the label with the highest model probability is the gold label (a tie of k labels scores 1/k if it holds gold). Checks only the top answer. | higher, max 1 |
| KL | KL(gold ‖ model) in nats: how far the model distribution is from the gold distribution. Punishes overconfident wrong answers hard. | lower, min 0 |
| Brier | Sum over labels of (model − gold)²: squared error of the probabilities, gentler on single bad misses than KL. | lower, min 0 (uniform ≈ 0.24) |
| ECE | Expected calibration error, 10 equal-width bins on the probability of the predicted label: gap between stated confidence and actual accuracy. | lower, min 0 |
| latency | Time per case (one request, five questions), end to end from the client; p50 = median. | lower |

### What the metrics say about calibration

A decision model is calibrated when its probabilities can be taken at face value: of all answers given with 90 %, about
90 % are right. That is what makes the probabilities usable, e.g. "refund automatically above 0.9, else ask a human".
Accuracy says nothing about this; it only looks at the top answer. The other three do, each from a different angle:

- **ECE** is calibration and nothing else. It groups the decisions by stated confidence and compares each group's
  confidence with its accuracy. 0.07 means the stated confidence is off by 7 points on average. It looks only at the
  top answer, and a model that always says "60 %" and is right 60 % of the time scores perfectly, while being useless.
- **Brier** scores the whole distribution. It is low only if the probabilities are both calibrated and sharp, i.e.
  confident where the answer is clear. A wrong 90 % costs 0.81 on that label, a wrong 60 % only 0.36, so overconfidence
  shows, but bounded.
- **KL** scores the whole distribution too, but with a log: putting ~0 % on the right answer costs almost without
  limit. A high KL with decent accuracy is the signature of a model that is often very sure and wrong. It is the metric
  that hits overconfidence hardest.

Reading them together: high accuracy with low KL and Brier means the probabilities can be trusted for thresholds. High
accuracy with high KL means good top answers but probabilities that should not be thresholded without recalibration.
