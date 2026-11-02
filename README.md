# class-cmpe-249-hw-1

CMPE249 Homework 1, Track A. This repository runs six COCO-pretrained detectors on COCO val2017 and, without retraining,
on the KITTI 2D validation split, scores them, and writes the report in `report/report.pdf`. It also contains the
prompt-tuning experiment, in which Claude models choose the text prompts of the OWLv2 open-vocabulary detector.

Open these first:

- `run.sh` runs the whole pipeline in resumable stages: parity, e1, cache, agent, arms, score, analyze, publish.
- `src/hw1/` holds the code. `run.py` runs one detector on one split, `score.py` scores saved predictions,
  `taxonomy.py` holds the KITTI to COCO label rules, and `agent.py` holds the prompt-tuning loop.
- `runs/` holds the results: `metrics/` for scores, `preds/*.meta.json` for the configuration, latency and exact
  command of each run, `analysis/` for the error breakdown, and `figures/` and `tables/` for the report.
  `runs/COMMANDS.md` logs every run.

To reproduce on an Apple silicon Mac (about 8 hours of compute and 45 GB of disk):

```sh
./setup.sh                                   # environment, COCO and KITTI downloads, tests
./run.sh parity e1 cache score analyze       # detector runs and scoring
export ANTHROPIC_API_KEY=...                 # needed only for the prompt-tuning stages
AGENT_MODEL=claude-sonnet-5-5 ./run.sh agent arms
uv run python -m hw1.figures && ./report/build.sh
```

`uv run pytest -q tests` runs the tests. `DRY_RUN=1 ./run.sh` lists the remaining work without running it.
Raw predictions and image caches stay on the machine that made them; only small results are committed.
