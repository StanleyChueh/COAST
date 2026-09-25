# COAST reproduction deck

A regenerable research presentation that summarizes the COAST reproduction study
(`research/reproduction/`, Phase 0 → latest phase). The deck is built from code;
normal updates need no manual PowerPoint editing.

## Layout

| Path | Role |
|---|---|
| `presentation_data.yaml` | **Content source of truth**: meta, phase list, paper-reported values, findings, open questions, author questions, per-slide text and speaker notes |
| `extract_data.py` | Reads `research/reproduction/experiments/*.{csv,yaml,json}` (read-only), recomputes every plotted/quoted measurement from per-episode records, **asserts it against the value each phase report states**, writes `generated_figures/chart_data.json` + `data_checks.txt` |
| `build_slides.js` | Layout only (PptxGenJS). Draws all charts as native vector shapes from `chart_data.json`; writes the `.pptx` and `speaker_notes.md` |
| `assets/theme.json` | Colour/typography tokens. Encoding is fixed: **orange = paper**, **cyan = ours**, gray = baseline/control, green = match, red = discrepancy, yellow = unresolved |
| `render_previews.sh` | LibreOffice → `previews/*.pdf` + one PNG per slide, for visual QA |
| `COAST_Reproduction_Status.pptx` | Current deck |
| `speaker_notes.md` | Generated notes (main point · what to say · likely question · answer); the same text is embedded as PowerPoint speaker notes |
| `archive/` | Frozen copies at phases that changed conclusions (`COAST_Reproduction_Status_phase7A.pptx`, …) |

## Build

```bash
cd presentation && npm install          # once (pptxgenjs, js-yaml)
cd .. && uv run --no-sync python presentation/extract_data.py   # numbers + cross-checks
cd presentation && node build_slides.js # deck + speaker notes
./render_previews.sh                    # optional: PDF/PNG previews
```

`extract_data.py` fails loudly if a record no longer matches the value the report
states; `build_slides.js` fails on any unresolved `{{…}}` placeholder.

## Placeholders in `presentation_data.yaml`

- `{{d.<path>}}` → `generated_figures/chart_data.json` (measured), e.g. `{{d.phase5a.global.label}}` → `19/30`.
  Append `|N` to round, e.g. `{{d.phase7a.mean_PGG|3}}`.
- `{{p.<path>}}` → `paper:` block (paper-reported values, tagged [P]).
- `{{m.<key>}}` → `meta:` block.
- Inline markup: `**bold**`, `` `code` ``, `^{sup}`, `_{sub}`, `<o>paper</o>`, `<c>ours</c>`, `<g>`, `<r>`, `<y>`, `<m>muted</m>`.

Measured numbers must never be typed into the YAML or the layout code; add them to
`extract_data.py` (with a cross-check) and reference them by placeholder.

## Adding a phase

1. Read the new phase report and its records in `research/reproduction/`.
2. `extract_data.py`: add a section that loads the new records and `check()`s them
   against the report's stated values.
3. `presentation_data.yaml`: bump `meta.latest_phase` / `meta.footer`, append to
   `phase_status`, update `major_findings`, `open_questions`, `author_questions.status`,
   `next_steps`, and edit or add only the affected slides (a new slide needs a layout
   function `L.<id>` in `build_slides.js`).
4. Keep the visual encoding in `assets/theme.json` unchanged.
5. Rebuild, render, inspect every changed slide.
6. If the conclusions changed, copy the previous deck to
   `archive/COAST_Reproduction_Status_phase<X>.pptx` before overwriting.

## Provenance notes

- Charts on slides 7, 8, 9, 10, 14 and 17 come entirely from `chart_data.json`.
- Paper values (slide 4, paper reference lines, Phase 7A paper Base/Global) come from
  the `paper:` block or from `phase7a_analysis.json`, which copies them from the paper.
- The provenance counts on slide 13 (152 configurations, "Selected Grid 18",
  P = 4.0 × 10⁻⁹) and the commit chronology on slide 12 are taken from the Phase 6C
  report text; they have no separate machine-readable record.
- Fonts: Arial / Courier New / Cambria Math. Previews use metric-compatible Liberation
  fonts, so text fit matches PowerPoint closely; equations (Cambria Math) are approximate.
