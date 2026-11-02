| Agent model | runs | evaluations (total) | cost per run USD (mean ± sd) | output tokens per run (mean) | best dev mAP (mean) | best map test Δ vs raw (mean [min, max]) | best maps significant (of 3) | one-shot test Δ (mean [min, max]) | mean \|dev step\| | worst single-step dev drop |
|---|---|---|---|---|---|---|---|---|---|---|
| Haiku 4.5* | 3 | 45 | 0.070 ± 0.001 | 4694 | 0.2318 | +0.0093 [+0.0046, +0.0134] | 3 of 3 | -0.0039 [-0.0063, -0.0027] | 0.0070 | -0.0514 |
| Haiku 5.5 | 3 | 44 | 0.017 ± 0.006 | 13499 | 0.2291 | +0.0054 [+0.0040, +0.0062] | 3 of 3 | -0.0008 [-0.0008, -0.0008] | 0.0017 | -0.0031 |
| Sonnet 5.5 | 3 | 45 | 0.209 ± 0.031 | 8593 | 0.2285 | +0.0109 [+0.0099, +0.0119] | 3 of 3 | +0.0031 [+0.0026, +0.0041] | 0.0022 | -0.0115 |
| Opus 5.5 | 3 | 45 | 0.323 ± 0.025 | 8030 | 0.2329 | +0.0041 [+0.0017, +0.0061] | 2 of 3 | -0.0008 [-0.0008, -0.0008] | 0.0014 | -0.0024 |

- Dev baselines (kitti_dev, 500 images): raw names 0.2260, ngdet synonyms 0.2250. Test mAP of raw names 0.1946. ngdet synonyms test Δ vs raw names: -0.0032 [-0.0045, -0.0024].
- Test Δ = kitti_test mAP of the selected prompt map minus raw names (OWLv2). 'Significant' = the paired-bootstrap 95% CI (1000 image resamples) excludes 0; it covers image resampling only, not run-to-run variance of the agent. 3 runs per model, so mean [min, max] over 3 seeds.
- Evaluations: dev evaluations the agent used (budget 15; one Haiku 5.5 run stopped at 14). Dev step = change in dev mAP between consecutive evaluations within a run; the 'worst drop' is the most negative single step over all runs of that model.
- Cost per run = (input x p_in + output x p_out + cache_read x p_read + cache_write x p_write) / 1e6 from the usage the API reported, with USD per million tokens from PRICES in src/hw1/agent.py: claude-opus-5-5: in/out/read/write = 4.0/20.0/0.20/5.0; claude-sonnet-5-5: in/out/read/write = 2.0/10.0/0.20/2.5; claude-haiku-4-5: in/out/read/write = 1.0/5.0/0.10/1.25; claude-haiku-5-5: in/out/read/write = 0.10/0.50/0.01/0.125. Cost covers agent tokens only, not OWLv2 compute.
- * Haiku 4.5 ran without the effort setting; the others ran at effort medium. Its summary.json has no token block, so its output tokens are summed from the per-turn usage in transcript.jsonl.
