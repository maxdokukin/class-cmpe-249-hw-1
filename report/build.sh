#!/usr/bin/env bash
# Build report/report.pdf: regenerate the appendix tables from run metadata, then compile with Typst.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv run python report/make_appendix.py
uv run --no-project --with typst python -c "import typst; typst.compile('report/report.typ', output='report/report.pdf', root='.')"
echo "wrote report/report.pdf"
