#!/usr/bin/env bash
# Full experiment pipeline. Resumable: rerun after any interruption and finished work is skipped.
#   ./run.sh                 all stages: parity e1 cache agent arms score analyze publish
#   ./run.sh e1 score        only the named stages
# Env: DEV (default mps), COOL (seconds between model runs, default 120), NO_PUSH=1 (commit results, don't push),
#      ANTHROPIC_API_KEY (the agent stage is skipped without it), AGENT_MODEL (default claude-haiku-4-5; e.g. claude-opus-5-5),
#      AGENT_DETECTORS (default "owlv2"; add gdino_t to also run the Grounding DINO agent),
#      DRY_RUN=1 (print the commands that would run, run nothing).
set -uo pipefail
cd "$(dirname "$0")"
if [[ $(uname) == Darwin && -z ${CAFFEINATED:-} ]]; then export CAFFEINATED=1; exec caffeinate -dims "$0" "$@"; fi
export PATH="$HOME/.local/bin:$PATH" PYTHONWARNINGS=ignore PYTORCH_ENABLE_MPS_FALLBACK=1
DEV=${DEV:-mps}; COOL=${COOL:-120}; AGENT_MODEL=${AGENT_MODEL:-claude-haiku-4-5}
AGENT_DETECTORS=${AGENT_DETECTORS:-owlv2}
STAGES=${*:-parity e1 cache agent arms score analyze publish}
MODELS="yolo11m yolo26m rtdetr_r50 frcnn_v2 owlv2 gdino_t"
mkdir -p runs/logs runs/parity
if [[ -n ${DRY_RUN:-} ]]; then py() { echo "  would run: python $*"; }; COOL=0; else py() { uv run python "$@"; }; fi
dev() { cat "runs/parity/$1.device" 2>/dev/null || echo "$DEV"; }
log() { echo "[$(date '+%a %H:%M:%S')] $*" | tee -a runs/logs/run.log; }

stage_parity() {  # MPS vs CPU on 20 images per model; a model that disagrees runs on CPU instead
  [[ $DEV == mps ]] || { log "parity skipped (DEV=$DEV)"; return; }
  for m in $MODELS; do
    [[ -f runs/parity/$m.device ]] && continue
    log "parity $m"
    out=$(py scripts/parity.py "$m" 2>&1 | tee -a runs/logs/parity.log | tail -1)
    [[ -n ${DRY_RUN:-} ]] && { echo "$out"; continue; }
    case $out in
      *"PARITY OK"*) echo mps > "runs/parity/$m.device" ;;
      *"PARITY FAIL"*) echo cpu > "runs/parity/$m.device" ;;
      *) log "  parity $m errored (see runs/logs/parity.log); undecided, will retry next run, using $DEV"; continue ;;
    esac
    echo "$out" > "runs/parity/$m.txt"
    log "  $out -> $(cat "runs/parity/$m.device")"
  done
}

run() {  # run <model> <split> [tag] [extra args...]
  local m=$1 s=$2 t=${3:-}
  shift $(( $# < 3 ? $# : 3 ))
  local name="${m}__${s}${t:+__$t}"
  [[ -f runs/preds/$name.meta.json ]] && return
  log "run $name on $(dev "$m")"
  py -m hw1.run --model "$m" --split "$s" --device "$(dev "$m")" ${t:+--tag "$t"} "$@" 2>&1 \
    | grep -vE 'Ultralytics|NNPACK|Loading weights' | tee -a runs/logs/e1.log | tail -1
  sleep "$COOL"
}

stage_e1() {
  for m in yolo11m yolo26m rtdetr_r50 frcnn_v2; do run $m coco_val; run $m kitti_test; done
  run yolo26m coco_val nms --set end2end=false
  run yolo26m kitti_test nms --set end2end=false
  run yolo11m kitti_test img1280 --set imgsz=1280
  for m in owlv2 gdino_t; do run $m coco_val1k; run $m kitti_test; done
  # latency on the parity-chosen device; gdino_t's main runs went to CPU after an interrupted parity check
  run gdino_t coco_val200 lat
  run gdino_t kitti_test200 lat
}

stage_cache() {  # OWLv2 image features: dev for the agent loop, test (sharded) for scoring the agent's arms
  if [[ ! -f runs/cache/owlv2__kitti_dev.pt ]]; then
    log "owlv2 cache kitti_dev"
    py -m hw1.owl_cache --split kitti_dev --device "$(dev owlv2)" 2>&1 | tail -1 | tee -a runs/logs/cache.log
  fi
  log "owlv2 cache kitti_test (sharded, resumable)"
  py -m hw1.owl_cache --split kitti_test --shard-size 250 --device "$(dev owlv2)" 2>&1 | grep -E 'shard|wrote' | tee -a runs/logs/cache.log
}

agent() {  # agent <detector> <seed>
  [[ -f runs/agent_logs/$1__kitti_dev__seed$2__${AGENT_MODEL#claude-}/summary.json ]] && return
  log "agent $1 seed $2 ($AGENT_MODEL)"
  py -m hw1.agent --detector "$1" --seed "$2" --device "$(dev "$1")" --model "$AGENT_MODEL" 2>&1 \
    | grep -v 'Loading weights' | tee -a runs/logs/agent.log | grep -E '^\[(iter|baseline)|refusal|Error'
}

stage_agent() {
  if [[ -z ${ANTHROPIC_API_KEY:-} ]]; then log "agent skipped: ANTHROPIC_API_KEY not set"; return; fi
  for s in 0 1 2; do agent owlv2 $s; done
  [[ " $AGENT_DETECTORS " == *" gdino_t "* ]] && agent gdino_t 0
  py scripts/agent_usage.py 2>&1 | tee -a runs/logs/agent.log
}

stage_arms() {
  for d in owlv2 gdino_t; do
    log "test arms $d"
    py -m hw1.arms --detector $d --device "$(dev $d)" 2>&1 | grep -v 'Loading weights' | tee -a runs/logs/arms.log | grep -E 'arms|mAP|done'
  done
}

stage_score() { log "score all"; py -m hw1.score_all 2>&1 | tee -a runs/logs/score.log; }

stage_analyze() {  # TIDE errors, recall bins, galleries on kitti_test; closed-set COCO rescored on the open-vocab models' 1000 images
  log "failure analysis"
  py -m hw1.failure 2>&1 | tee -a runs/logs/analyze.log
  for m in yolo11m yolo26m rtdetr_r50 frcnn_v2; do
    [[ -f runs/metrics/${m}__coco_val__default__sub-coco_val1k.json ]] && continue
    [[ -f runs/preds/${m}__coco_val.jsonl.gz || -n ${DRY_RUN:-} ]] || continue
    log "rescore $m on coco_val1k"
    py -m hw1.score "runs/preds/${m}__coco_val.jsonl.gz" --subset coco_val1k --fast 2>&1 | tee -a runs/logs/score.log
  done
}

stage_publish() {  # small outputs only (metrics, analysis, run metadata, agent logs, logs); predictions and caches stay local
  [[ -n ${DRY_RUN:-} ]] && { log "publish: would commit and push runs/ outputs"; return; }
  git add runs/metrics runs/preds/*.meta.json runs/agent_logs runs/parity runs/analysis runs/logs runs/COMMANDS.md 2>/dev/null
  if git diff --cached --quiet; then log "publish: nothing new"; return; fi
  git commit -qm "hw1 results: $(hostname -s) $(date '+%Y-%m-%d %H:%M')" && log "publish: committed"
  if [[ -n ${NO_PUSH:-} ]]; then log "publish: NO_PUSH set, not pushing"; return; fi
  git pull -q --rebase --autostash || log "publish: pull failed; resolve by hand, then 'git push'"
  if git push -q; then log "publish: pushed"; else log "publish: push failed; run 'git push' by hand"; fi
}

log "start stages: $STAGES  (DEV=$DEV)"
for st in $STAGES; do "stage_$st"; done
log "finished: $STAGES"
