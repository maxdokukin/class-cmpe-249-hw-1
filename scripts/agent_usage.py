"""Token use and cost per agent run, plus what the same token counts would cost on other models.

    python scripts/agent_usage.py

The projection reprices the measured token counts. A bigger model may differ in practice: Opus 5.5 always thinks
(more output tokens) and uses a different tokenizer (roughly 1.0-1.35x the token count of Haiku 4.5), so the
Opus figure is a lower bound; the 'x2 output' column is a rough allowance for thinking.
"""
import json
from pathlib import Path

from hw1.agent import PRICES

runs = sorted(Path("runs/agent_logs").glob("*__seed*/summary.json"))
if not runs:
    raise SystemExit("no finished agent runs in runs/agent_logs")


def price(model, t, out_mult=1.0):
    pin, pout, pread, pwrite = PRICES[model]
    return (t["input"] * pin + t["output"] * out_mult * pout + t["cache_read"] * pread + t["cache_write"] * pwrite) / 1e6


tot = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
print(f"{'run':44s} {'evals':>5s} {'input':>9s} {'output':>8s} {'c.read':>9s} {'c.write':>8s} {'cost $':>7s} {'best dev mAP':>12s}")
for f in runs:
    s = json.loads(f.read_text())
    t = s.get("tokens")
    if not t:  # runs from before token totals were added: sum the per-call usage in the transcript
        t = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        for line in (f.parent / "transcript.jsonl").read_text().splitlines():
            u = json.loads(line)["usage"]
            t["input"] += u.get("input_tokens") or 0
            t["output"] += u.get("output_tokens") or 0
            t["cache_read"] += u.get("cache_read_input_tokens") or 0
            t["cache_write"] += u.get("cache_creation_input_tokens") or 0
    for k in tot:
        tot[k] += t.get(k, 0)
    print(f"{f.parent.name:44s} {s['evaluations']:5d} {t.get('input', 0):9,d} {t.get('output', 0):8,d} "
          f"{t.get('cache_read', 0):9,d} {t.get('cache_write', 0):8,d} {s['cost_usd']:7.3f} {s['best_dev_mAP']:12.4f}")
print(f"\ntotal tokens: input {tot['input']:,}  output {tot['output']:,}  cache read {tot['cache_read']:,}  cache write {tot['cache_write']:,}")
print("same token counts repriced (all runs together):")
for m in PRICES:
    print(f"  {m:20s} ${price(m, tot):7.2f}   with x2 output for thinking: ${price(m, tot, 2.0):7.2f}")
