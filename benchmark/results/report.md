# celeris-1-decision vs. mercury-decide on Typed Decisions

400 cases, 2000 decisions per model.

## Leaderboard

| # | Model | accuracy ↑ | KL ↓ | Brier ↓ | p50 latency |
|---|---|---|---|---|---|
| 1 | meraGPT Decider 1 | **0.768** | **0.096** | **0.052** | 526 ms |
| 2 | Liquid AI d1 | 0.742 | 0.475 | 0.155 | 525 ms |
| 3 | **celeris-1-decision (this run)** | 0.739 | 0.214 | 0.108 | 243 ms |
| 4 | TypeSafe Jev 1.13.0 | 0.727 | 1.442 | 0.148 | 710 ms |
| 5 | Featherless Simple Jev | 0.716 | 0.488 | 0.176 | – |
| 6 | **mercury-decide (this run)** | 0.715 | 0.967 | 0.271 | 409 ms |
| 7 | prima-ratio + 12B § | 0.702 | 0.564 | 0.234 | 700 ms‡ |
| 8 | OpenDecider-small § | 0.671 | 0.211 | 0.117 | 40 ms‡ |
| 9 | Bongard-mini § | 0.594 | 0.256 | 0.132 | 225 ms‡ |
| 10 | Jeff-Gemma4-E2B | 0.561 | 0.403 | 0.219 | 2,272 ms† |
| 11 | Jeff-Qwen3.5-2B | 0.511 | 0.460 | 0.237 | 1,346 ms† |
| 12 | Jeff-Qwen3.5-0.8B | 0.483 | 0.679 | 0.313 | 662 ms† |
| 13 | Prior | 0.470 | 0.347 | 0.189 | – |
| 14 | Uniform | 0.308 | 0.444 | 0.238 | – |

Card: zero-shot leaderboard of the dataset card, same test split; models fitted on `train` are left out, as the card does. Best value per metric in bold. Whether a gap to a card model is real: see the intervals below. Latency: p50 end to end per case from a client, one request at a time; this run from one machine with a kept-alive connection. § self-reported. † measured on the model's own machine (M3 Max) and ‡ reported by the submitter on their own hardware, both not comparable with hosted latency.

## Head to head

95% bootstrap intervals in brackets.

| Model | accuracy | kl | brier |
|---|---|---|---|
| celeris-1-decision | 0.739 [0.715, 0.761] | 0.214 [0.201, 0.228] | 0.108 [0.100, 0.117] |
| mercury-decide | 0.715 [0.694, 0.735] | 0.967 [0.904, 1.037] | 0.271 [0.255, 0.289] |
| Δ mercury-decide − celeris-1-decision | -0.024 [-0.048, 0.000] | +0.754 [0.691, 0.819] | +0.164 [0.148, 0.179] |
| uniform (this code) | 0.318 | 0.444 | 0.238 |

An interval of Δ that excludes 0 is a real difference. uniform (this code): the uniform distribution scored by evaluate.py, to check the metrics against the card's Uniform row (its accuracy 0.308 breaks ties differently; 1/k credit gives the expected value).

## Calibration

Decisions binned by the probability of the predicted label.

| p | Model | decisions | mean p | accuracy |
|---|---|---|---|---|
| 0.0–0.5 | celeris-1-decision | 0 | – | – |
| 0.0–0.5 | mercury-decide | 20 | 0.479 | 0.425 |
| 0.5–0.6 | celeris-1-decision | 935 | 0.585 | 0.681 |
| 0.5–0.6 | mercury-decide | 102 | 0.553 | 0.471 |
| 0.6–0.7 | celeris-1-decision | 313 | 0.627 | 0.572 |
| 0.6–0.7 | mercury-decide | 110 | 0.652 | 0.536 |
| 0.7–0.8 | celeris-1-decision | 80 | 0.700 | 0.738 |
| 0.7–0.8 | mercury-decide | 151 | 0.754 | 0.616 |
| 0.8–0.9 | celeris-1-decision | 555 | 0.833 | 0.886 |
| 0.8–0.9 | mercury-decide | 264 | 0.858 | 0.636 |
| 0.9–1.0 | celeris-1-decision | 117 | 0.910 | 0.949 |
| 0.9–1.0 | mercury-decide | 1353 | 0.976 | 0.778 |

ECE (10 equal-width bins): celeris-1-decision 0.072, mercury-decide 0.185. The card's ECE definitions differ by submitter; compare the two models here, not with the card.

## Time per sample

One sample is one request with all five questions, client-side from the first attempt, ms. Failed requests are left out.

| Model | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| celeris-1-decision | 258 | 243 | 316 | 453 | 629 |
| mercury-decide | 423 | 409 | 500 | 1,126 | 1,656 |

## Requests

One request per case with five questions.

| Model | requests | failed | retried | USD / 1,000 cases |
|---|---|---|---|---|
| celeris-1-decision | 400 | 0 | 0 | 0.054 |
| mercury-decide | 400 | 0 | 0 | 0.056 |
