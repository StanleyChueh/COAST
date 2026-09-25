#!/usr/bin/env bash
# Render the deck to PDF + one PNG per slide for visual QA.
# Usage (from presentation/): ./render_previews.sh [deck.pptx]
# LibreOffice substitutes Liberation Sans/Serif for Arial/Times (metric-compatible),
# so text fit in the previews matches PowerPoint closely; Cambria Math is approximated.
set -euo pipefail
cd "$(dirname "$0")"
DECK="${1:-COAST_Reproduction_Status.pptx}"
OUT=previews
PROFILE="$(mktemp -d)"
trap 'rm -rf "$PROFILE"' EXIT
mkdir -p "$OUT"
rm -f "$OUT"/slide-*.png
timeout 180 soffice -env:UserInstallation="file://$PROFILE" --headless --convert-to pdf --outdir "$OUT" "$DECK" >/dev/null
pdftoppm -png -r 80 "$OUT/$(basename "${DECK%.pptx}").pdf" "$OUT/slide"
ls -1 "$OUT"/slide-*.png | wc -l | xargs echo "rendered slides:"
