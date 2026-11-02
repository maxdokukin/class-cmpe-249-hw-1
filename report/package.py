"""Write dist/hw1_report.pdf and dist/hw1_code_and_results.zip (code, configs, logs and results; no datasets or weights)."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "pyproject.toml", "uv.lock", "setup.sh", "run.sh", ".gitignore", "src", "scripts", "tests", "doc",
           "report/report.typ", "report/refs.yml", "report/appendix_tables.typ", "report/build.sh", "report/make_appendix.py",
           "report/package.py", "runs/metrics", "runs/analysis", "runs/agent_logs", "runs/figures", "runs/tables",
           "runs/parity", "runs/logs", "runs/COMMANDS.md"]
SKIP = ("__pycache__", ".pytest_cache", ".pyc")

dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
shutil.copy(ROOT / "report/report.pdf", dist / "hw1_report.pdf")
n = 0
with zipfile.ZipFile(dist / "hw1_code_and_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = ROOT / item
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            if any(s in str(f) for s in SKIP):
                continue
            z.write(f, Path("class-cmpe-249-hw-1") / f.relative_to(ROOT)); n += 1
    for f in sorted((ROOT / "runs/preds").glob("*.meta.json")):
        z.write(f, Path("class-cmpe-249-hw-1") / f.relative_to(ROOT)); n += 1
print(f"{n} files -> {dist / 'hw1_code_and_results.zip'} ({(dist / 'hw1_code_and_results.zip').stat().st_size / 1e6:.1f} MB)")
