# Dev decision: retrieval config for the single test run (2026-10-23)

Written before any test run of this experiment. Dev = `evals/dev_set.jsonl` (23 answerable, frozen).
LLM: Qwen2.5-7B Q4_K_M, prompt v1, temperature 0.3, seed 0 (one run per config, so noisy).

## Retrieval (all 6 configs)
Hit@5 = Recall@5 = 1.00 for every config, both doc-level (20 candidates) and on the final 5
context chunks. Dev retrieval is saturated and cannot rank configs, so all 3 interventions
(reranker only, smaller chunks only, both) plus a second chunk size went to the LLM check.
Rerank cost on dev: mean 351 ms (1200-char chunks), 124 ms (600), 83 ms (400).

## LLM faithfulness (dev, seed 0)
| config | small faith. | large faith. | halluc. small / large | neg. ctrl small / large | generate mean (s) |
|---|---|---|---|---|---|
| c1200 (baseline) | **0.877** | **0.804** | 0.261 / 0.304 | 0.116 / 0.116 | 17.7 |
| c1200_rr | 0.659 | 0.758 | 0.364 / 0.318 | 0.129 / 0.144 | 19.8 |
| c600 (600/100) | 0.754 | 0.601 | 0.348 / 0.478 | 0.152 / 0.123 | 14.2 |
| c600_rr | 0.849 | 0.689 | 0.273 / 0.409 | 0.159 / 0.076 | 18.9 |
| c400_rr (400/80) | 0.819 | 0.754 | 0.261 / 0.261 | 0.109 / 0.123 | 14.2 |

## Decision
No candidate beat the baseline on dev under either judge. The best candidate is **c400_rr**
(highest mean of both judges among candidates, 0.786; hallucination rate not above baseline
under either judge). It is the single config taken to test, against c1200, with seeds 0, 1, 2.
Expectation from dev: no improvement. The default changes only under the pre-registered rule in
`src/rag_production/experiment.py` (both judges: paired bootstrap 95% CI excludes 0 and the new
worst seed beats the baseline best seed).
Box note: other agents' jobs shared the 8 vCPU during dev runs, so dev latency is indicative only.
