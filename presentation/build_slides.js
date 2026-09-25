// Build the COAST reproduction deck.
//
//   content  : presentation_data.yaml          (narrative, paper values, notes)
//   numbers  : generated_figures/chart_data.json (from extract_data.py; never hand-typed)
//   theme    : assets/theme.json
//   output   : COAST_Reproduction_Status.pptx, speaker_notes.md
//
// Usage (from presentation/):  node build_slides.js
"use strict";

const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const Y = yaml.load(fs.readFileSync(path.join(HERE, "presentation_data.yaml"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(HERE, "generated_figures", "chart_data.json"), "utf8"));
const T = JSON.parse(fs.readFileSync(path.join(HERE, "assets", "theme.json"), "utf8"));
const K = T.color;
const F = T.font;
const S = T.size;

const W = 13.333;
const MX = 0.6;
const CW = W - 2 * MX;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = Y.meta.author;
pres.title = `${Y.meta.title} — updated through Phase ${Y.meta.latest_phase}`;

// ------------------------------------------------------------------ templating
function get(obj, dotted) {
  return dotted.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function tpl(str) {
  return String(str).replace(/\{\{([dpm])\.([^}|]+)(?:\|(\d+))?\}\}/g, (_, ns, p, dec) => {
    const root = ns === "d" ? D : ns === "p" ? Y.paper : Y.meta;
    let v = get(root, p);
    if (v === undefined || v === null) throw new Error(`Unresolved placeholder {{${ns}.${p}}}`);
    if (dec !== undefined) v = Number(v).toFixed(Number(dec));
    return String(v);
  });
}

const TAG = { o: K.paper, c: K.ours, g: K.green, r: K.red, y: K.yellow, m: K.dim, w: K.text };

// Inline markup -> pptxgenjs runs: **bold**, `code`, ^{sup}, _{sub}, <o>..</o> colours.
function rich(str, base = {}) {
  const s = tpl(str);
  const out = [];
  const re = /(`[^`]*`|\*\*|\^\{[^}]*\}|_\{[^}]*\}|<\/?[ocgrymw]>)/g;
  const colors = [];
  let bold = false;
  let last = 0;
  let m;
  const push = (text, extra = {}) => {
    if (!text) return;
    out.push({
      text,
      options: { ...base, bold: bold || !!base.bold, color: colors.length ? colors[colors.length - 1] : base.color, ...extra },
    });
  };
  while ((m = re.exec(s))) {
    push(s.slice(last, m.index));
    const tok = m[0];
    if (tok === "**") bold = !bold;
    else if (tok[0] === "`") push(tok.slice(1, -1), { fontFace: F.mono, fontSize: Math.round((base.fontSize || S.body) * 0.88) });
    else if (tok.startsWith("^{")) push(tok.slice(2, -1), { superscript: true });
    else if (tok.startsWith("_{")) push(tok.slice(2, -1), { subscript: true });
    else if (tok.startsWith("</")) colors.pop();
    else colors.push(TAG[tok[1]]);
    last = m.index + tok.length;
  }
  push(s.slice(last));
  return out;
}

// Several paragraphs -> one run list. pOpts apply to every run (bullet, spacing, align).
function paras(list, base, pOpts = {}) {
  const runs = [];
  list.forEach((item, i) => {
    const r = rich(item, base);
    r.forEach((x) => Object.assign(x.options, pOpts));
    if (i < list.length - 1 && r.length) r[r.length - 1].options.breakLine = true;
    runs.push(...r);
  });
  return runs;
}

// ------------------------------------------------------------------ primitives
function text(slide, content, o) {
  const base = { fontFace: o.font || F.body, fontSize: o.size || S.body, color: o.color || K.text, bold: !!o.bold, italic: !!o.italic };
  const runs = Array.isArray(content) ? paras(content, base, o.p || {}) : rich(content, base);
  slide.addText(runs, {
    x: o.x, y: o.y, w: o.w, h: o.h,
    fontFace: base.fontFace, fontSize: base.fontSize, color: base.color,
    align: o.align || "left", valign: o.valign || "top", margin: o.margin === undefined ? 0 : o.margin,
    charSpacing: o.charSpacing, isTextBox: true, fit: "none", lineSpacingMultiple: o.lsm,
  });
}

function box(slide, x, y, w, h, o = {}) {
  slide.addShape(o.square ? pres.shapes.RECTANGLE : pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: o.fill === null ? { type: "none" } : { color: o.fill || K.card, transparency: o.transparency || 0 },
    line: o.line ? { color: o.line, width: o.lineW || 1, dashType: o.dash || "solid" } : { type: "none" },
    rectRadius: o.square ? undefined : (o.r === undefined ? 0.08 : o.r),
  });
}

function hline(slide, x, y, w, color, width = 1, dash = "solid") {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width, dashType: dash } });
}

function vline(slide, x, y, h, color, width = 1, dash = "solid") {
  slide.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color, width, dashType: dash } });
}

function arrow(slide, x, y, w, color = K.dim, width = 2) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width, endArrowType: "triangle" } });
}

function dot(slide, cx, cy, d, color, hollow = false, lineW = 2.25) {
  slide.addShape(pres.shapes.OVAL, {
    x: cx - d / 2, y: cy - d / 2, w: d, h: d,
    fill: hollow ? { color: K.bg } : { color },
    line: { color, width: hollow ? lineW : 0.75 },
  });
}

// Width estimate for Arial bold: caps ≈ 0.72 em, mixed case ≈ 0.56 em, plus charSpacing (pt).
function textWidth(label, size, cs) {
  const caps = label === label.toUpperCase();
  return label.length * size * (caps ? 0.0102 : 0.0080) + (label.length * cs) / 72;
}

function chip(slide, x, y, label, color, o = {}) {
  const cs = o.cs === undefined ? 1 : o.cs;
  const w = o.w || 0.34 + textWidth(label, o.size || 12, cs);
  const h = o.h || 0.34;
  box(slide, x, y, w, h, { fill: o.fill || K.card, line: color, lineW: 1, r: 0.17 });
  text(slide, label, { x, y, w, h, size: o.size || 12, color, bold: true, align: "center", valign: "middle", charSpacing: cs });
  return w;
}

function kicker(slide, s) {
  text(slide, s, { x: MX, y: 0.3, w: CW, h: 0.28, size: S.kicker, color: K.ours, bold: true, charSpacing: 3 });
}

function title(slide, s) {
  text(slide, s, { x: MX, y: 0.58, w: CW, h: 0.62, size: S.title, bold: true, font: F.head });
}

function source(slide, s) {
  if (!s) return;
  text(slide, `Source: ${s}`, { x: MX, y: 6.7, w: CW, h: 0.34, size: S.source, color: K.dim });
}

function footer(slide, n, N) {
  text(slide, Y.meta.footer, { x: MX, y: 7.1, w: 8, h: 0.24, size: S.footer, color: K.dim });
  text(slide, `${n} / ${N}`, { x: W - MX - 1.5, y: 7.1, w: 1.5, h: 0.24, size: S.footer, color: K.dim, align: "right" });
}

function legendItem(slide, x, y, color, label, kind = "dot", tw = 3.4) {
  if (kind === "dot") dot(slide, x + 0.09, y + 0.14, 0.17, color);
  else if (kind === "ring") dot(slide, x + 0.09, y + 0.14, 0.17, color, true, 2);
  else if (kind === "dash") hline(slide, x, y + 0.14, 0.34, color, 2.25, "dash");
  else if (kind === "bar") box(slide, x, y + 0.04, 0.2, 0.2, { fill: color, square: true });
  const off = kind === "dash" ? 0.44 : 0.28;
  text(slide, label, { x: x + off, y, w: tw, h: 0.28, size: S.label, color: K.muted, valign: "middle" });
}

const frac = (o) => o.label;

// ------------------------------------------------------------------ charts
// Vertical dot plot of success counts with Wilson whiskers and reference lines.
function dotPlot(slide, o) {
  const { x, y, w, h, ymin, ymax, ticks, cats } = o;
  const py = (v) => y + h - ((v - ymin) / (ymax - ymin)) * h;
  ticks.forEach((t) => {
    hline(slide, x, py(t), w, K.grid, 0.75);
    text(slide, String(t), { x: x - 0.55, y: py(t) - 0.14, w: 0.45, h: 0.28, size: S.label, color: K.dim, align: "right" });
  });
  if (o.yLabel) text(slide, o.yLabel, { x: x - 0.6, y: y - 0.5, w: 6, h: 0.28, size: S.label, color: K.dim });
  (o.refs || []).forEach((r) => {
    hline(slide, x, py(r.v), w, r.color, 2.25, "dash");
    text(slide, r.label, { x: x + w - 3.2, y: py(r.v) - 0.36, w: 3.2, h: 0.3, size: S.label, color: r.color, bold: true, align: "right" });
  });
  const bw = w / cats.length;
  cats.forEach((c, ci) => {
    const cx = x + bw * (ci + 0.5);
    const n = c.points.length;
    const sp = o.spacing || 1.0;
    c.points.forEach((p, pi) => {
      const px = cx + (pi - (n - 1) / 2) * sp - 0.28;
      if (p.lo !== undefined) vline(slide, px, py(p.hi), py(p.lo) - py(p.hi), p.color, 1.25);
      if (p.lo !== undefined) {
        hline(slide, px - 0.06, py(p.hi), 0.12, p.color, 1.25);
        hline(slide, px - 0.06, py(p.lo), 0.12, p.color, 1.25);
      }
      dot(slide, px, py(p.v), 0.26, p.color, p.hollow);
      text(slide, [`**${p.label}**`, `<m>${p.tag || ""}</m>`], { x: px + 0.2, y: py(p.v) - 0.25, w: 0.9, h: 0.5, size: S.label, color: p.color });
    });
    text(slide, [`**${c.label}**`, `<m>${c.sub || ""}</m>`], { x: x + bw * ci + 0.05, y: y + h + 0.12, w: bw - 0.1, h: 0.62, size: 15, align: "center" });
  });
}

// ------------------------------------------------------------------ slides
const L = {};

L.title = (s) => {
  s.background = { color: K.bgDeep };
  text(s, `RESEARCH MEETING · ${Y.meta.date}`, { x: 0.8, y: 1.35, w: 7, h: 0.3, size: 13, color: K.ours, bold: true, charSpacing: 3 });
  text(s, Y.meta.title, { x: 0.8, y: 1.8, w: 7.8, h: 0.8, size: 40, bold: true, font: F.head });
  text(s, Y.meta.subtitle, { x: 0.8, y: 2.7, w: 7.8, h: 0.5, size: 22, color: K.text });
  text(s, Y.meta.scope, { x: 0.8, y: 3.3, w: 7.8, h: 0.4, size: 19, color: K.ours });
  text(s, Y.meta.author, { x: 0.8, y: 4.75, w: 7, h: 0.4, size: 22, bold: true });
  text(s, [`Reproducing ${Y.meta.paper_ref}`, `<m>Paper authors: ${Y.meta.paper_authors}</m>`], { x: 0.8, y: 5.2, w: 7.2, h: 0.75, size: 14, color: K.muted });
  chip(s, 0.8, 6.25, `UPDATED THROUGH PHASE ${Y.meta.latest_phase}`, K.green, { size: 12 });

  // Motif: the 50 LIBERO initial states of one task, coloured by how we use them.
  const gx = 8.85, gy = 1.95, sp = 0.4;
  for (let i = 0; i < 50; i++) {
    const cx = gx + (i % 10) * sp, cy = gy + Math.floor(i / 10) * sp;
    if (i < 15) dot(s, cx, cy, 0.26, K.text, true, 1.75);
    else if (i < 45) dot(s, cx, cy, 0.26, K.ours);
    else dot(s, cx, cy, 0.26, K.line);
  }
  text(s, "50 LIBERO initial states per task", { x: 8.62, y: 4.05, w: 4.2, h: 0.3, size: 13, color: K.muted });
  legendItem(s, 8.62, 4.45, K.text, "fit / selection · states 0–14", "ring");
  legendItem(s, 8.62, 4.8, K.ours, "held-out test · states 15–44", "dot");
  legendItem(s, 8.62, 5.15, K.line, "unused · states 45–49", "dot");
};

L.objective = (s, c) => {
  text(s, `“${c.question}”`, { x: 1.2, y: 1.55, w: W - 2.4, h: 1.3, size: 30, bold: true, align: "center", valign: "middle", font: F.head });
  const cw = 3.55, gap = 0.74, y0 = 3.35, ch = 2.75;
  const colors = { red: K.red, cyan: K.ours, yellow: K.yellow };
  c.steps.forEach((st, i) => {
    const x = MX + 0.05 + i * (cw + gap);
    box(s, x, y0, cw, ch, { fill: K.card });
    dot(s, x + 0.45, y0 + 0.45, 0.46, K.cardHi);
    text(s, String(i + 1), { x: x + 0.22, y: y0 + 0.22, w: 0.46, h: 0.46, size: 18, bold: true, color: K.ours, align: "center", valign: "middle" });
    text(s, st.head, { x: x + 0.85, y: y0 + 0.25, w: cw - 1.0, h: 0.45, size: 20, bold: true, valign: "middle" });
    text(s, st.body, { x: x + 0.3, y: y0 + 0.95, w: cw - 0.6, h: 1.1, size: 17, color: K.muted });
    chip(s, x + 0.3, y0 + ch - 0.6, st.status, colors[st.color], { size: 12.5, w: cw - 0.6, cs: 0 });
    if (i < 2) arrow(s, x + cw + 0.12, y0 + ch / 2, gap - 0.24, K.dim, 2.5);
  });
};

L.coast = (s, c) => {
  const stages = [
    { head: "1 · Collect", lines: ["H^{+}  ∈  ℝ^{N₊ × d}", "H^{−}  ∈  ℝ^{N₋ × d}"], note: "successful / failed rollouts; rows = token-pooled h at layer ℓ" },
    { head: "2 · Correlate", lines: ["R = (1/N) Σ_{i} x_{i} x_{i}^{T}", "x_{i} = h_{i} − μ", "R ∈ ℝ^{d × d}"], note: "one R per class, mean-centered" },
    { head: "3 · Conceptor", lines: ["C = R (R + α^{−2} I)^{−1}"], note: "eigenvalues λ / (λ + α^{−2}) ∈ [0, 1): a soft projection" },
    { head: "4 · Contrast", lines: ["C_{steer} = C^{+} ∧ ¬C^{−}", "¬C = I − C"], note: ["A ∧ B =", "(A^{−1} + B^{−1} − I)^{−1}"] },
    { head: "5 · Steer", lines: ["M = (1 − β) I + β C_{steer}", "h′ = M h"], note: "at every denoising step" },
  ];
  const gap = 0.34, cw = (CW - 4 * gap) / 5, y0 = 1.6, ch = 2.85;
  stages.forEach((st, i) => {
    const x = MX + i * (cw + gap);
    box(s, x, y0, cw, ch, { fill: i === 4 ? K.oursDeep : K.card, line: i === 4 ? K.ours : undefined });
    text(s, st.head, { x: x + 0.18, y: y0 + 0.18, w: cw - 0.3, h: 0.34, size: 16, bold: true, color: K.ours });
    text(s, st.lines, { x: x + 0.18, y: y0 + 0.68, w: cw - 0.3, h: 1.2, size: 15, font: F.math, p: { paraSpaceAfter: 6 } });
    text(s, st.note, { x: x + 0.18, y: y0 + 1.95, w: cw - 0.3, h: 0.85, size: 13, color: K.muted });
    if (i < 4) arrow(s, x + cw + 0.04, y0 + ch / 2, gap - 0.08, K.dim, 2);
  });
  box(s, MX, 4.7, CW, 0.62, { fill: K.card });
  text(s, "d = 1024 (action-expert width)  ·  10 action tokens  ·  10 Euler denoising steps  ·  layer ℓ ∈ {0, 5, 11, 17} of 18", {
    x: MX + 0.25, y: 4.7, w: CW - 0.5, h: 0.62, size: 16, color: K.muted, valign: "middle",
  });
  text(s, "Strategies", { x: MX, y: 5.6, w: 1.6, h: 0.4, size: 16, bold: true, valign: "middle" });
  const strat = [["Global", "one C_{steer} for all steps"], ["Per-step", "one C per step"], ["Positive-only", "C = C^{+}"]];
  strat.forEach(([h, b], i) => {
    const x = 2.3 + i * 3.55;
    box(s, x, 5.58, 3.35, 0.46, { fill: K.card, line: K.line });
    text(s, `**${h}** — ${b}`, { x: x + 0.15, y: 5.58, w: 3.1, h: 0.46, size: 15, valign: "middle" });
  });
  text(s, "Paper description. Implementations differ on several of these steps (slides 11, 15).", { x: MX, y: 6.2, w: CW, h: 0.3, size: 13, color: K.yellow });
};

L.paper = (s) => {
  const P = Y.paper.ks3;
  chip(s, MX, 1.38, "PAPER-REPORTED VALUES", K.paper, { size: 12 });
  const rowsP = [
    ["Base (unsteered)", "", P.base.value, true],
    ["Global", P.global.config, P.global.value, false],
    ["Per-step", P.per_step.config, P.per_step.value, false],
    ["Positive-only", P.pos_only.config, P.pos_only.value, false],
  ];
  const bx = 3.05, bwMax = 4.2, y0 = 2.05, rh = 1.02;
  [0, 0.5, 1].forEach((t) => {
    vline(s, bx + t * bwMax, y0 - 0.1, rh * 4, K.grid, 0.75);
    text(s, t.toFixed(1), { x: bx + t * bwMax - 0.3, y: y0 + rh * 4 - 0.02, w: 0.6, h: 0.28, size: S.label, color: K.dim, align: "center" });
  });
  rowsP.forEach(([name, cfg, v, isBase], i) => {
    const y = y0 + i * rh;
    text(s, [`**${name}**`, `<m>${cfg}</m>`], { x: MX, y: y + 0.08, w: 2.35, h: 0.7, size: 17 });
    box(s, bx, y + 0.14, bwMax * Number(v), 0.5, { fill: isBase ? K.paperDeep : K.paper, line: K.paper, square: true });
    text(s, v, { x: bx + bwMax * Number(v) + 0.12, y: y + 0.06, w: 1.0, h: 0.66, size: 30, bold: true, color: K.paper, valign: "middle" });
  });
  text(s, "KS3 success rate · π0.5 · LIBERO-10 · Table 4 (Base: Table 1)", { x: MX, y: y0 + rh * 4 + 0.32, w: 7.6, h: 0.3, size: S.label, color: K.dim });

  const px = 8.55, pw = W - MX - px;
  box(s, px, 1.95, pw, 2.95, { fill: K.card });
  text(s, "How to read these numbers", { x: px + 0.25, y: 2.1, w: pw - 0.5, h: 0.36, size: 17, bold: true });
  text(s, [
    `Paper text: 30 held-out test rollouts ⇒ <o>**${P.global.value} = ${P.global.count30}**</o>`,
    `Every Table 4 cell is also k/15 ⇒ <o>**${P.global.value} = ${P.global.count15}**</o> (slide 13)`,
    `Base <o>**${P.base.value} = ${P.base.count15}**</o> equals the Table 21 fit-set rate`,
  ], { x: px + 0.25, y: 2.6, w: pw - 0.45, h: 2.5, size: 15, color: K.muted, p: { bullet: { indent: 14 }, paraSpaceAfter: 10 } });
  const M = Y.paper.libero10_mean;
  text(s, [`**LIBERO-10 mean (10 tasks)**`, `Base ${M.base} · Global ${M.global}`, `Per-step ${M.per_step} · Positive-only ${M.pos_only}`], {
    x: px, y: 5.15, w: pw, h: 1.0, size: 15, color: K.paper,
  });
};

L.protocol = (s, c) => {
  const gap = 0.42, bw = (CW - 3 * gap) / 4, y0 = 1.45, bh = 1.15;
  c.flow.forEach((f, i) => {
    const x = MX + i * (bw + gap);
    box(s, x, y0, bw, bh, { fill: i === 3 ? K.oursDeep : K.card, line: i === 3 ? K.ours : i === 0 ? K.text : undefined });
    text(s, [`**${f.head}**`, `<m>${f.body}</m>`], { x: x + 0.2, y: y0 + 0.14, w: bw - 0.4, h: bh - 0.25, size: 17, valign: "middle" });
    if (i < 3) arrow(s, x + bw + 0.06, y0 + bh / 2, gap - 0.12, K.dim, 2);
  });

  const sx = 0.85, sw = CW - 0.5, sp = sw / 49, cy = 3.72;
  const px = (i) => sx + i * sp;
  const bracket = (a, b, yb, label, color, up) => {
    hline(s, px(a) - 0.1, yb, px(b) - px(a) + 0.2, color, 1.5);
    vline(s, px(a) - 0.1, up ? yb : yb - 0.12, 0.12, color, 1.5);
    vline(s, px(b) + 0.1, up ? yb : yb - 0.12, 0.12, color, 1.5);
    text(s, label, { x: px(a) - 0.1, y: up ? yb - 0.4 : yb + 0.06, w: px(b) - px(a) + 0.2, h: 0.32, size: 14, color, bold: true, align: "center" });
  };
  bracket(0, 14, 3.3, "Fit + selection · 0–14", K.text, true);
  bracket(15, 44, 3.3, "Held-out test · 15–44 (Phases 4A, 5A, 5C)", K.ours, true);
  bracket(45, 49, 3.3, "unused", K.dim, true);
  for (let i = 0; i < 50; i++) {
    if (i < 15) dot(s, px(i), cy, 0.19, K.text, true, 1.5);
    else if (i < 45) dot(s, px(i), cy, 0.19, K.ours);
    else dot(s, px(i), cy, 0.19, K.line);
    if (i % 5 === 0) text(s, String(i), { x: px(i) - 0.25, y: cy + 0.16, w: 0.5, h: 0.24, size: 11, color: K.dim, align: "center" });
  }
  bracket(15, 29, 4.55, "Development · 15–29 (1C–2B)", K.muted, false);
  bracket(30, 44, 4.55, "Held-out · 30–44 (3A)", K.muted, false);
  text(s, "LIBERO init state index · KS3 · episode k → state (seed + k) mod 50", { x: sx - 0.1, y: 5.2, w: 8, h: 0.28, size: 13, color: K.dim });
  box(s, MX, 5.7, CW, 0.62, { fill: K.yellowDeep, line: K.yellow });
  text(s, c.caveat, { x: MX + 0.25, y: 5.7, w: CW - 0.5, h: 0.62, size: 17, valign: "middle" });
};

L.timeline = (s) => {
  const ph = Y.phase_status;
  const n = ph.length, bw = 2.3, x0 = MX + bw / 2 + 0.02, x1 = W - MX - bw / 2 - 0.02, ly = 4.05;
  const step = (x1 - x0) / (n - 1);
  hline(s, x0 - 0.3, ly, x1 - x0 + 0.6, K.line, 2.5);
  ph.forEach((p, i) => {
    const cx = x0 + i * step, up = i % 2 === 0, latest = i === n - 1;
    const by = up ? 1.5 : 4.5, bh = 2.05;
    vline(s, cx, up ? by + bh : ly, up ? ly - by - bh : by - ly, K.line, 1.25);
    dot(s, cx, ly, latest ? 0.34 : 0.24, latest ? K.green : K.ours);
    box(s, cx - bw / 2, by, bw, bh, { fill: latest ? K.greenDeep : K.card, line: latest ? K.green : undefined });
    text(s, [`<c>**Phase ${p.id}**</c>`, `**${p.group}**`], { x: cx - bw / 2 + 0.15, y: by + 0.12, w: bw - 0.3, h: 0.7, size: 15 });
    text(s, p.outcome, { x: cx - bw / 2 + 0.15, y: by + 0.85, w: bw - 0.3, h: bh - 0.95, size: 14, color: K.muted });
  });
};

L.gap = (s, c) => {
  const g5a = D.phase5a.global, g4a = D.phase4a.S_coast_b01, g5c = D.phase5c.test.global;
  const b5a = D.phase5a.baseline, b4a = D.phase4a.S_baseline, b5c = D.phase5c.test.baseline;
  const pt = (f, color, tag, hollow) => ({ v: f.k, lo: f.wilson95[0] * f.n, hi: f.wilson95[1] * f.n, label: f.label, tag, color, hollow });
  dotPlot(s, {
    x: 1.25, y: 2.15, w: 6.9, h: 3.45, ymin: 10, ymax: 30, ticks: [10, 15, 20, 25, 30], spacing: 1.05,
    yLabel: "Successes / 30 · held-out states 15–44 · whiskers: Wilson 95%",
    refs: [{ v: 28, label: `Paper Global ${Y.paper.ks3.global.count30}`, color: K.paper }],
    cats: [
      { label: "Global · paper config", sub: "L5 · α 0.5 · β 0.1", points: [pt(g5a, K.ours, "5A"), pt(g4a, K.ours, "4A", true)] },
      { label: "Global · oracle", sub: g5c.config.replace("α", "· α ").replace("β", "· β ").replace(/ +/g, " "), points: [pt(g5c, K.ours, "5C")] },
      { label: "Unsteered baseline", sub: "same states", points: [pt(b5a, K.base, "5A"), pt(b4a, K.base, "4A", true), pt(b5c, K.base, "5C")] },
    ],
  });
  legendItem(s, 0.65, 1.3, K.muted, "released default, unseeded", "dot");
  legendItem(s, 3.55, 1.3, K.muted, "paired noise, seed 100 (Phase 4A)", "ring");

  const px = 8.85, pw = W - MX - px;
  text(s, c.message, { x: px, y: 1.55, w: pw, h: 0.95, size: 24, bold: true, color: K.ours });
  text(s, c.stats, { x: px, y: 2.65, w: pw, h: 3.7, size: 15, color: K.muted, p: { bullet: { indent: 14 }, paraSpaceAfter: 9 } });
};

L.noise = (s, c) => {
  const nf = D.phase1d.noise_floor;
  const rowsN = [
    { label: "States 15–29", sub: "15 episodes", pts: [[nf["100"], true], [nf["200"], true], [nf["300"], true], [D.phase1c.baseline, false]] },
    { label: "States 0–14", sub: "15 episodes (fit states)", pts: [[D.phase1a.fit, false], ...D.phase5c.fit_baselines.map((k) => [{ k, n: 15, rate: k / 15, label: `${k}/15` }, false])] },
    { label: "States 15–44", sub: "30 episodes", pts: [[D.phase4a.S_baseline, true], [D.phase5a.baseline, false], [D.phase5c.test.baseline, false]] },
  ];
  const lx = MX, ax = 3.0, aw = 5.3, y0 = 2.35, rh = 1.18;
  const pxv = (r) => ax + r * aw;
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    vline(s, pxv(t), y0 - 0.35, rh * 3 - 0.1, K.grid, 0.75);
    text(s, t.toFixed(2), { x: pxv(t) - 0.3, y: y0 + rh * 3 - 0.4, w: 0.6, h: 0.26, size: S.label, color: K.dim, align: "center" });
  });
  text(s, "Unsteered success rate (same policy, same states)", { x: ax, y: y0 + rh * 3 - 0.12, w: aw, h: 0.28, size: S.label, color: K.dim, align: "center" });
  rowsN.forEach((r, i) => {
    const cy = y0 + i * rh + 0.12;
    text(s, [`**${r.label}**`, `<m>${r.sub}</m>`], { x: lx, y: cy - 0.32, w: 2.3, h: 0.66, size: 16 });
    const rates = r.pts.map(([f]) => f.k / f.n);
    const lo = Math.min(...rates), hi = Math.max(...rates);
    box(s, pxv(lo), cy - 0.05, pxv(hi) - pxv(lo), 0.1, { fill: K.line, square: true });
    const sorted = r.pts.map(([f, hollow]) => ({ f, hollow, x: pxv(f.k / f.n) })).sort((a, b) => a.x - b.x);
    sorted.forEach((p, j) => {
      dot(s, p.x, cy, 0.24, K.base, p.hollow);
      const above = j % 2 === 0;
      text(s, p.f.label, { x: p.x - 0.4, y: above ? cy - 0.47 : cy + 0.15, w: 0.8, h: 0.3, size: S.label, color: K.text, align: "center", bold: true });
    });
    text(s, `range ${Math.round(lo * rowsN[i].pts[0][0].n)}–${Math.round(hi * rowsN[i].pts[0][0].n)}`, {
      x: ax + aw + 0.12, y: cy - 0.14, w: 1.3, h: 0.28, size: S.label, color: K.muted,
    });
  });
  legendItem(s, lx, 1.35, K.base, "paired noise seed (1D: 100/200/300; 4A: 100)", "ring", 4.4);
  legendItem(s, lx + 4.6, 1.35, K.base, "released default, unseeded", "dot");

  const px = 9.9, pw = W - MX - px;
  text(s, c.message, { x: px, y: 1.5, w: pw, h: 1.3, size: 20, bold: true, color: K.ours });
  text(s, c.stats, { x: px, y: 2.95, w: pw, h: 3.4, size: 15, color: K.muted, p: { bullet: { indent: 14 }, paraSpaceAfter: 10 } });
};

L.mechanism = (s, c) => {
  const E = D.phase1e;
  box(s, MX, 1.45, 6.35, 4.35, { fill: K.card });
  text(s, [
    "M = (1 − β) I + β C",
    "code:  h′ = h M^{T}  (row)   ⇔   h′ = M h  (column)",
    "h′ = (1 − β) h + β C h",
    "Δh = h′ − h = β (C h − h)",
    "ρ = h^{T}C h / ‖h‖^{2} ≈ 0   ⇒   h′ ≈ (1 − β) h",
  ], { x: MX + 0.3, y: 1.62, w: 5.9, h: 2.75, size: 19, font: F.math, p: { paraSpaceAfter: 9 } });
  text(s, [
    "h ∈ ℝ^{1024} per action token  ·  M, C ∈ ℝ^{1024 × 1024}",
    "applied to all 10 tokens at all 10 denoising steps, layer 5",
    `C symmetric ⇒ ‖βCh‖ ≤ β λ_{max} ‖h‖;  λ_{max} = ${E.lambda_max.toFixed(3)} ⇒ ≤ ${(0.1 * E.lambda_max * 100).toFixed(1)} % of ‖h‖`,
  ], { x: MX + 0.3, y: 4.45, w: 5.9, h: 1.25, size: 13.5, color: K.muted, p: { paraSpaceAfter: 4 } });

  const rx = 7.25, rw = W - MX - rx, tw = (rw - 0.3) / 3;
  const tiles = [
    [E.rel_delta_mean.toFixed(4), "‖Δh‖ / ‖h‖", "β = 0.1"],
    [E.cos_mean.toFixed(4).replace("-", "−"), "cos(Δh, h)", "−1 = pure shrink"],
    [E.rayleigh_text, "ρ = hᵀCh / ‖h‖²", `trace/d = ${E.trace_over_d.toFixed(4)}`],
  ];
  tiles.forEach(([v, l, sub], i) => {
    const x = rx + i * (tw + 0.15);
    box(s, x, 1.45, tw, 1.55, { fill: K.card });
    text(s, v, { x, y: 1.55, w: tw, h: 0.62, size: 28, bold: true, color: K.ours, align: "center", valign: "middle" });
    text(s, [`**${l}**`, `<m>${sub}</m>`], { x: x + 0.05, y: 2.22, w: tw - 0.1, h: 0.7, size: 13, align: "center" });
  });
  text(s, `${E.hook_applications.toLocaleString("en-US")} hook applications · V0 · L5 α 0.5 β 0.1 · states 15–29 (1E)`, {
    x: rx, y: 3.08, w: rw, h: 0.28, size: 12.5, color: K.dim,
  });

  // Geometry sketch: h, (1-β)h and a magnified βCh component.
  const ox = rx + 0.15, L1 = 4.9, gy1 = 3.75, gy2 = 4.45;
  arrow(s, ox, gy1, L1, K.text, 2.5);
  text(s, "h", { x: ox + L1 + 0.05, y: gy1 - 0.18, w: 0.4, h: 0.36, size: 18, font: F.math, bold: true });
  arrow(s, ox, gy2, L1 * 0.9, K.ours, 2.5);
  text(s, "(1 − β) h", { x: ox + 0.1, y: gy2 + 0.04, w: 2, h: 0.32, size: 15, font: F.math, color: K.ours });
  s.addShape(pres.shapes.LINE, { x: ox + L1 * 0.9, y: gy2 - 0.28, w: 0, h: 0.28, flipV: true, line: { color: K.yellow, width: 1.75, dashType: "dash", endArrowType: "triangle" } });
  text(s, "+ β C h  (tiny)", { x: ox + L1 * 0.9 - 1.9, y: gy2 - 0.4, w: 1.8, h: 0.3, size: 13, font: F.math, color: K.yellow, align: "right" });
  text(s, "sketch, not to scale: the βCh component is magnified", { x: ox, y: gy2 + 0.38, w: rw, h: 0.26, size: 11.5, color: K.dim });
  const cr = D.phase7a.global_cos_range;
  box(s, rx, 5.25, rw, 1.2, { fill: K.oursDeep, line: K.ours });
  const fc = (v) => v.toFixed(4).replace("-", "−");
  text(s, [c.interpretation, `<m>7A, candidate builder, 4/4 tasks: cos(Δh, h) from ${fc(cr[1])} to ${fc(cr[0])}</m>`], {
    x: rx + 0.2, y: 5.28, w: rw - 0.4, h: 1.14, size: 14.5, valign: "middle",
  });
};

function barPanel(s, x, y, w, h, heading, vals, maxV, chips) {
  text(s, heading, { x, y, w, h: 0.34, size: 16, bold: true });
  const ch = h - 1.5, cy = y + 0.65;
  const py = (v) => cy + ch - (v / maxV) * ch;
  [0, 5, 10, 15].forEach((t) => {
    hline(s, x + 0.35, py(t), w - 0.35, K.grid, 0.75);
    text(s, String(t), { x: x - 0.1, y: py(t) - 0.13, w: 0.35, h: 0.26, size: 12, color: K.dim, align: "right" });
  });
  const n = vals.length, slot = (w - 0.45) / n, bw = slot * 0.62;
  vals.forEach((v, i) => {
    const bx = x + 0.45 + i * slot + (slot - bw) / 2;
    box(s, bx, py(v.f.k), bw, py(0) - py(v.f.k), { fill: v.color, square: true, line: v.outline, dash: v.dash });
    text(s, v.f.label, { x: bx - 0.2, y: py(v.f.k) - 0.34, w: bw + 0.4, h: 0.3, size: 15, bold: true, align: "center", color: v.color === K.ours ? K.ours : K.text });
    text(s, v.name, { x: bx - 0.25, y: py(0) + 0.06, w: bw + 0.5, h: 0.28, size: 12.5, align: "center", color: K.muted });
  });
  chips.forEach((t, i) => text(s, t, { x, y: y + h - 0.36 + i * 0.3, w, h: 0.3, size: 14, color: K.text }));
}

L.ablation = (s, c) => {
  const mk = (P) => [
    { f: P.baseline, name: "Baseline", color: K.base },
    { f: P.shrinkage, name: "Shrink", color: K.control },
    { f: P.coast, name: "COAST", color: K.ours },
    { f: P.random_matched, name: "Random", color: K.bg, outline: K.base, dash: "dash" },
  ];
  const B = D.phase2b, H = D.phase3a;
  barPanel(s, MX, 1.45, 4.55, 4.15, "Development · states 15–29 (2A/2B)", mk(B), 15, [
    `COAST ↔ shrink **${B.agree_coast_shrink}/15** · COAST ↔ random **${B.agree_coast_random}/15**`,
  ]);
  barPanel(s, 5.45, 1.45, 4.55, 4.15, "Held-out · states 30–44 (3A)", mk(H), 15, [
    `COAST ↔ shrink **${H.agree_coast_shrink}/15** · COAST ↔ random **${H.agree_coast_random}/15**`,
  ]);
  const rx = 10.35, rw = W - MX - rx;
  box(s, rx, 1.45, rw, 4.15, { fill: K.card });
  text(s, "Realized ‖h′‖ / ‖h‖", { x: rx + 0.2, y: 1.6, w: rw - 0.4, h: 0.3, size: 15, bold: true });
  const nr = B.norm_ratio;
  [["Shrinkage", nr.shrinkage], ["COAST", nr.coast], ["Random", nr.random_matched]].forEach(([n, v], i) => {
    text(s, n, { x: rx + 0.2, y: 2.05 + i * 0.42, w: 1.2, h: 0.36, size: 15, color: K.muted, valign: "middle" });
    text(s, v.toFixed(4), { x: rx + 1.2, y: 2.05 + i * 0.42, w: rw - 1.4, h: 0.36, size: 17, bold: true, align: "right", valign: "middle", color: n === "COAST" ? K.ours : K.text });
  });
  text(s, "Across all 30 states, real COAST was never the only condition that differed from the other steered conditions.", {
    x: rx + 0.2, y: 3.45, w: rw - 0.4, h: 1.9, size: 14, color: K.muted,
  });
  box(s, MX, 5.72, CW, 0.7, { fill: K.oursDeep, line: K.ours });
  text(s, c.conclusion, { x: MX + 0.25, y: 5.72, w: CW - 0.5, h: 0.7, size: 16.5, valign: "middle" });
  text(s, c.caveat, { x: MX, y: 6.47, w: CW, h: 0.26, size: 11, color: K.dim });
};

const MARK = { ok: ["✓", K.green], no: ["✗", K.red], part: ["!", K.yellow] };

L.why_history = (s, c) => {
  const tw = (CW - 3 * 0.18) / 4;
  c.triggers.forEach((t, i) => {
    const x = MX + i * (tw + 0.18);
    box(s, x, 1.4, tw, 0.62, { fill: K.yellowDeep, line: K.yellow });
    text(s, t, { x: x + 0.14, y: 1.4, w: tw - 0.28, h: 0.62, size: 13.5, valign: "middle" });
  });
  const rowsW = [
    ["Token pooling", "mean over 10 action tokens", ["no", "every token a row"], ["ok", "mean-pooled"]],
    ["Centering", "mean-centered R", ["no", "uncentered"], ["ok", "centered"]],
    ["Contrastive formula", "C^{+} ∧ ¬C^{−}  (pinv AND)", ["no", "A(A + B − AB)^{−1}B"], ["no", "C_{s}(I − C_{f})"]],
    ["Per-step α", "α = 10 (KS3)", ["no", "fixed α = 1"], ["part", "label a10.0, math α = 1"]],
    ["Episodes per evaluation", "30 test rollouts", ["part", "configurable (docs: 15)"], ["no", "15 (NUM_EPISODES)"]],
    ["State semantics", "different seeds / initial conditions", ["ok", "(seed + k) mod 50"], ["no", "initial_states[k], k = 0…14"]],
    ["Evaluation split", "15 fit + 30 held-out", ["ok", "possible via --seed offset"], ["no", "evaluated on fit states 0–14"]],
  ];
  const cols = [[MX, 2.45], [3.15, 3.05], [6.3, 3.15], [9.55, 3.18]];
  const hy = 2.22, rh = 0.53;
  ["", "Paper text", "Released main", "miranda-v2"].forEach((h, i) => {
    text(s, h, { x: cols[i][0] + 0.12, y: hy, w: cols[i][1], h: 0.36, size: 15, bold: true, color: i === 1 ? K.paper : i ? K.text : K.muted, valign: "middle" });
  });
  rowsW.forEach((r, j) => {
    const y = hy + 0.42 + j * rh;
    text(s, r[0], { x: cols[0][0], y, w: cols[0][1], h: rh - 0.06, size: 15, bold: true, valign: "middle" });
    box(s, cols[1][0], y, cols[1][1] - 0.08, rh - 0.07, { fill: K.card });
    text(s, r[1], { x: cols[1][0] + 0.12, y, w: cols[1][1] - 0.25, h: rh - 0.07, size: 14, color: K.muted, valign: "middle" });
    [r[2], r[3]].forEach(([k, t], ci) => {
      const [x, w] = cols[ci + 2];
      box(s, x, y, w - 0.08, rh - 0.07, { fill: K.card });
      text(s, MARK[k][0], { x: x + 0.1, y, w: 0.3, h: rh - 0.07, size: 17, bold: true, color: MARK[k][1], valign: "middle", align: "center" });
      text(s, t, { x: x + 0.45, y, w: w - 0.6, h: rh - 0.07, size: 14, valign: "middle" });
    });
  });
  text(s, c.legend, { x: MX, y: hy + 0.42 + 7 * rh + 0.05, w: CW, h: 0.28, size: 13, color: K.muted });
};

L.archaeology = (s, c) => {
  const B = D.phase6c.branches;
  const fun = [
    [`${B.public_branches} branches + ${B.pr_heads} PR heads`, "every public ref of COAST-VLA/COAST"],
    [`${B.deep_inspected} branches in depth`, "commit-by-commit + pickaxe search"],
    ["miranda-v2", "closest candidate pipeline"],
  ];
  const fw = (CW - 2 * 0.5) / 3;
  fun.forEach(([h, b], i) => {
    const x = MX + i * (fw + 0.5);
    box(s, x, 1.4, fw, 1.0, { fill: i === 2 ? K.oursDeep : K.card, line: i === 2 ? K.ours : undefined });
    text(s, [`**${h}**`, `<m>${b}</m>`], { x: x + 0.2, y: 1.4, w: fw - 0.4, h: 1.0, size: 16, valign: "middle", color: i === 2 ? K.ours : K.text });
    if (i < 2) arrow(s, x + fw + 0.08, 1.9, 0.34, K.dim, 2);
  });
  const ev = [
    ["Apr 9", "HF release", "checkpoint + fit activations (states 0–14)", K.text],
    ["Apr 13", "2fdc5ad", "miranda-v2 LIBERO sweep · 15 episodes", K.ours],
    ["Apr 15", "29059a5", "first public conceptor builder (for_subin)", K.ours],
    ["Apr 21", "ec62cad", "oracle_gap_table.py → Table 15", K.ours],
    ["Apr 23", "PR #48", "--seed now selects init states", K.text],
    ["May 16", "arXiv v1", "paper submitted", K.paper],
    ["Jun 1", "6c34a11 · #50", "steering merged to main", K.text],
  ];
  const x0 = 1.55, x1 = W - 1.55, ly = 4.3, st = (x1 - x0) / (ev.length - 1), bw = 2.55;
  hline(s, x0 - 0.5, ly, x1 - x0 + 1.0, K.line, 2.5);
  text(s, "≈", { x: x0 + 4.5 * st - 0.2, y: ly - 0.25, w: 0.4, h: 0.4, size: 20, color: K.dim, align: "center", bold: true });
  ev.forEach(([d, sha, desc, col], i) => {
    const cx = x0 + i * st, up = i % 2 === 0, by = up ? 2.8 : 4.75, bh = 1.12;
    vline(s, cx, up ? by + bh : ly, up ? ly - by - bh : by - ly, K.line, 1.25);
    dot(s, cx, ly, 0.24, col);
    text(s, [`**${d}**  \`${sha}\``, `<m>${desc}</m>`], { x: cx - bw / 2, y: by, w: bw, h: bh, size: 14, color: col, align: "center", valign: up ? "bottom" : "top" });
  });
  legendItem(s, MX, 6.05, K.ours, "miranda-v2 lineage", "dot");
  legendItem(s, MX + 2.6, 6.05, K.paper, "paper", "dot");
  legendItem(s, MX + 4.0, 6.05, K.text, "release / main", "dot");
  text(s, c.message, { x: 6.2, y: 5.98, w: W - MX - 6.2, h: 0.62, size: 14.5, color: K.text, valign: "middle" });
};

L.evidence = (s, c) => {
  chip(s, MX, 1.3, c.badge.toUpperCase(), K.yellow, { size: 12 });
  const cards = [
    ["152", "configurations per task", "Paper: “up to 152 configurations” = the miranda-v2 sweep: 135 steered + 9 random + 8 positive-only"],
    ["18", "“Selected Grid” (π0.5 LIBERO)", "= the count `oracle_gap_table.py` derives from the sweep's condition names"],
    ["a10.0", "per-step “α = 10”", "A loop label in the condition name `per_step_*_L5_a10.0_b0.3`; α does not enter the per-step conceptor"],
    ["10 / 10", "positive-only cells", "All inside the sweep grid L{5,11} × α{0.5,1} × β{0.1,0.3} (8 conditions), not Table 14's 60"],
    ["k / 15", "30 / 30 Table 4 cells", "If N = 30: P(all 30 numerators even) = 4.0 × 10^{−9}. Control: Table 8 (16-env pipeline) shows k/16"],
    ["Caption", "Table 15", "Near-verbatim rendering of the LaTeX f-string in `oracle_gap_table.py`, same benchmark order"],
  ];
  const gw = 0.2, cw = (CW - 2 * gw) / 3, ch = 2.28;
  cards.forEach(([big, lab, desc], i) => {
    const x = MX + (i % 3) * (cw + gw), y = 1.85 + Math.floor(i / 3) * (ch + 0.2);
    box(s, x, y, cw, ch, { fill: K.card });
    text(s, big, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.7, size: 34, bold: true, color: K.green, valign: "middle" });
    text(s, lab, { x: x + 0.25, y: y + 0.88, w: cw - 0.5, h: 0.34, size: 16, bold: true });
    text(s, desc, { x: x + 0.25, y: y + 1.26, w: cw - 0.45, h: ch - 1.35, size: 13.5, color: K.muted });
  });
};

L.historical = (s, c) => {
  const H = D.phase6c;
  const pt = (f, color) => ({ v: f.k, lo: f.wilson95[0] * f.n, hi: f.wilson95[1] * f.n, label: f.label, tag: f.run, color });
  dotPlot(s, {
    x: 1.25, y: 2.15, w: 6.9, h: 3.45, ymin: 5, ymax: 15, ticks: [5, 7.5, 10, 12.5, 15], spacing: 1.0,
    yLabel: "Successes / 15 · states 0–14 (= fit states) · whiskers: Wilson 95%",
    refs: [{ v: 14, label: `Paper Global ${Y.paper.ks3.global.value} ≙ ${Y.paper.ks3.global.count15}`, color: K.paper }],
    cats: [
      { label: "Global", sub: "L5 · α 0.5 · β 0.1", points: H.global.map((f) => pt(f, K.ours)) },
      { label: "Unsteered baseline", sub: "same sessions", points: H.baseline.map((f) => pt(f, K.base)) },
      { label: "Random control", sub: "candidate, L5 · β 0.1", points: H.random_b01.map((f) => pt(f, K.control)) },
    ],
  });
  chip(s, 0.65, 1.3, "HISTORICAL FIT-STATE SEMANTICS · NOT HELD-OUT", K.yellow, { size: 11.5 });
  const px = 8.85, pw = W - MX - px;
  text(s, c.message, { x: px, y: 1.45, w: pw, h: 1.5, size: 19, bold: true, color: K.ours });
  const ps = (v) => v.toFixed(2);
  text(s, [
    `Pooled: Global **${H.pooled.global}** · baseline **${H.pooled.baseline}** · random **${H.pooled.random_b01}**`,
    `Fisher: vs baseline p = ${ps(H.p_global_vs_baseline)} · vs random p = ${ps(H.p_global_vs_random)} · vs 14/15 p = ${ps(H.p_global_vs_paper_native)}`,
    `Per-step cell as coded: ${H.per_step_0[0].label}, ${H.per_step_9[0].label} (paper ${H.paper_native.per_step}) · Positive-only ${H.pos_only[0].label} (paper ${H.paper_native.pos_only})`,
    "Evaluated states 0–14 are the authors' fit-rollout states",
  ], { x: px, y: 3.1, w: pw, h: 3.3, size: 14.5, color: K.muted, p: { bullet: { indent: 14 }, paraSpaceAfter: 9 } });
};

L.discrepancies = (s, c) => {
  const R = [
    ["Evaluation episodes", "30 per condition", ["part", "configurable (docs: 15)"], ["no", "15 (NUM_EPISODES = 15)"]],
    ["Fit / test separation", "15 fit + 30 disjoint test", ["ok", "via --seed offset (post-#48)"], ["no", "none: eval states 0–14 = fit"]],
    ["Token pooling", "mean over 10 action tokens", ["no", "every token is a row"], ["ok", "mean-pooled"]],
    ["Centering", "mean-centered R", ["no", "uncentered R = XᵀX / N"], ["ok", "centered"]],
    ["Contrastive operation", "(A^{−1} + B^{−1} − I)^{−1}, pinv", ["no", "A (A + B − AB)^{−1} B"], ["no", "C_{s} (I − C_{f})"]],
    ["Global aggregation", "all denoising steps", ["ok", "all steps × all tokens"], ["no", "denoising step 0 only"]],
    ["Per-step α", "α = 10 (KS3)", ["no", "fixed α = 1.0"], ["no", "label a10.0; conceptor α = 1"]],
    ["Per-step mechanism", "separate C per step", ["ok", "time-varying"], ["no", "fixed ds-0 or ds-9 C"]],
    ["Grid size", "240 (Table 14); text: up to 152", ["part", "configurable; per-step α fixed"], ["ok", "152 (matches text + cells)"]],
  ];
  const cols = [[MX, 2.5], [3.15, 3.1], [6.3, 3.2], [9.55, 3.18]];
  const hy = 1.34, rh = 0.47;
  ["Component", "Paper (text)", "Released main · 2afa10e", "miranda-v2 · 29059a5"].forEach((h, i) => {
    text(s, h, { x: cols[i][0] + (i ? 0.12 : 0), y: hy, w: cols[i][1], h: 0.4, size: 15, bold: true, valign: "middle", color: i === 1 ? K.paper : i ? K.text : K.muted });
  });
  const fill = { ok: K.greenDeep, no: K.redDeep, part: K.yellowDeep };
  R.forEach((r, j) => {
    const y = hy + 0.45 + j * rh;
    text(s, r[0], { x: cols[0][0], y, w: cols[0][1], h: rh - 0.06, size: 15, bold: true, valign: "middle" });
    box(s, cols[1][0], y, cols[1][1] - 0.08, rh - 0.06, { fill: K.card, r: 0.05 });
    text(s, r[1], { x: cols[1][0] + 0.12, y, w: cols[1][1] - 0.24, h: rh - 0.06, size: 14, color: K.muted, valign: "middle" });
    [r[2], r[3]].forEach(([k, t], ci) => {
      const [x, w] = cols[ci + 2];
      box(s, x, y, w - 0.08, rh - 0.06, { fill: fill[k], r: 0.05 });
      text(s, MARK[k][0], { x: x + 0.08, y, w: 0.3, h: rh - 0.06, size: 16, bold: true, color: MARK[k][1], valign: "middle", align: "center" });
      text(s, t, { x: x + 0.42, y, w: w - 0.55, h: rh - 0.06, size: 14, valign: "middle" });
    });
  });
  const fy = hy + 0.45 + 9 * rh + 0.04;
  text(s, "<g>✓</g> as in the paper text    <r>✗</r> differs    <y>!</y> partial / unclear", { x: MX, y: fy, w: CW, h: 0.28, size: 12.5, color: K.muted });
  text(s, c.footnote, { x: MX, y: fy + 0.3, w: CW, h: 0.28, size: 12, color: K.muted });
};

L.phase7a_design = (s, c) => {
  chip(s, W - MX - 3.1, 0.62, c.status, K.green, { size: 12, w: 3.1 });
  const T7 = D.phase7a.tasks;
  // column 1: task selection
  const c1x = MX, c1w = 4.25;
  box(s, c1x, 1.45, c1w, 4.95, { fill: K.card });
  text(s, [`**Task selection · paper values only**`, "<m>Eligible: Table 21 S ≥ 3 and F ≥ 3; KS3 excluded</m>"], { x: c1x + 0.22, y: 1.58, w: c1w - 0.4, h: 0.7, size: 15 });
  const sel = [["A", "LR2a", "largest paper gain"], ["B", "KS4", "median paper gain"], ["C", "LR1", "smallest paper gain"]];
  sel.forEach(([slot, t, rule], i) => {
    const y = 2.55 + i * 1.25;
    box(s, c1x + 0.2, y, c1w - 0.4, 1.12, { fill: K.cardHi });
    text(s, slot, { x: c1x + 0.3, y: y + 0.1, w: 0.45, h: 0.45, size: 22, bold: true, color: K.ours });
    const cfg = T7[t].config.replace(/a/, "α ").replace(/ b/, " · β ").replace(/^L(\d+) /, "L$1 · ");
    text(s, [`**${t}** <m>· ${rule}</m>`, `<o>Paper Base ${T7[t].paper_base.toFixed(2)} → Global ${T7[t].paper_global.toFixed(2)}</o>`, `<m>${cfg}</m>`], {
      x: c1x + 0.8, y: y + 0.08, w: c1w - 1.1, h: 1.0, size: 14,
    });
  });
  // column 2: arms and conditions
  const c2x = 5.05, c2w = 3.85;
  box(s, c2x, 1.45, c2w, 4.95, { fill: K.card });
  text(s, "**Two arms, one conceptor**", { x: c2x + 0.22, y: 1.58, w: c2w - 0.4, h: 0.36, size: 15 });
  const arms = [["Arm H · historical", "states 0–14 · 15 episodes", "= the fit states", K.text], ["Arm T · held-out", "states 15–44 · 30 episodes", "disjoint from the fit states", K.ours]];
  arms.forEach(([h, a, b, col], i) => {
    const y = 2.05 + i * 1.1;
    box(s, c2x + 0.2, y, c2w - 0.4, 0.98, { fill: K.cardHi, line: col });
    text(s, [`**${h}**`, a, `<m>${b}</m>`], { x: c2x + 0.35, y: y + 0.05, w: c2w - 0.7, h: 0.9, size: 14, color: col });
  });
  text(s, "**Conditions**", { x: c2x + 0.22, y: 4.3, w: c2w - 0.4, h: 0.3, size: 14 });
  legendItem(s, c2x + 0.22, 4.62, K.base, "Baseline (unsteered)", "dot");
  legendItem(s, c2x + 0.22, 4.9, K.ours, "Paper Global (exact config)", "dot");
  legendItem(s, c2x + 0.22, 5.18, K.control, "Random control (candidate)*", "ring");
  text(s, [`2 repeats · ${D.phase7a.n_episodes} episodes · unseeded`, "* not spectrum-matched"], { x: c2x + 0.22, y: 5.6, w: c2w - 0.4, h: 0.62, size: 12.5, color: K.dim });
  // column 3: metric
  const c3x = 9.1, c3w = W - MX - c3x;
  box(s, c3x, 1.45, c3w, 4.95, { fill: K.oursDeep, line: K.ours });
  text(s, "**Primary quantity**", { x: c3x + 0.22, y: 1.58, w: c3w - 0.4, h: 0.36, size: 15 });
  text(s, ["G_{H} = SR_{COAST,H} − SR_{BASE,H}", "G_{T} = SR_{COAST,T} − SR_{BASE,T}", "**PGG = G_{H} − G_{T}**"], {
    x: c3x + 0.22, y: 2.05, w: c3w - 0.4, h: 1.5, size: 16, font: F.math, p: { paraSpaceAfter: 8 },
  });
  text(s, c.pgg_note, { x: c3x + 0.22, y: 3.55, w: c3w - 0.4, h: 1.0, size: 14 });
  text(s, [
    "**Case rule (pre-registered)**",
    "A: G_{H} > G_{T} in most tasks, no clear held-out advantage",
    "B: mixed · C: COAST beats baseline and random on T",
    "<m>clear = pooled Fisher p < 0.05 and same sign in both repeats</m>",
  ], { x: c3x + 0.22, y: 4.55, w: c3w - 0.4, h: 1.8, size: 12.5, color: K.muted, p: { paraSpaceAfter: 3 } });
};

L.phase7a_results = (s, c) => {
  const P = D.phase7a, order = P.task_order;
  const lx = MX, px0 = 2.0, pw = 4.4, rmin = 0.2, rmax = 1.0, cx0 = 6.55;
  const px = (r) => px0 + ((r - rmin) / (rmax - rmin)) * pw;
  const rowH = 0.44;
  const block = (y0, arm, headline, rowsB, showPaper) => {
    text(s, headline, { x: lx, y: y0, w: 6.2, h: 0.32, size: 15, bold: true });
    text(s, "Base · Global · Random", { x: cx0, y: y0, w: 2.0, h: 0.32, size: 12, color: K.dim });
    rowsB.forEach((r, i) => {
      const cy = y0 + 0.55 + i * rowH;
      text(s, r.label, { x: lx, y: cy - 0.16, w: 1.3, h: 0.32, size: 15, bold: r.pooled, valign: "middle" });
      hline(s, px(rmin), cy, pw, K.grid, 0.75);
      const pts = [["BASELINE", K.base, false], ["RANDOM_CONTROL", K.control, true], ["PAPER_GLOBAL", K.ours, false]];
      const rates = pts.map(([k]) => r.v[k].k / r.v[k].n);
      hline(s, px(Math.min(...rates)), cy, px(Math.max(...rates)) - px(Math.min(...rates)), K.line, 3);
      pts.forEach(([k, col, hollow]) => dot(s, px(r.v[k].k / r.v[k].n), cy, 0.22, col, hollow));
      if (showPaper && r.paper !== undefined) {
        vline(s, px(r.paper), cy - 0.17, 0.34, K.paper, 3);
      }
      text(s, `${r.v.BASELINE.k} · <c>${r.v.PAPER_GLOBAL.k}</c> · ${r.v.RANDOM_CONTROL.k} /${r.v.BASELINE.n}`, { x: cx0, y: cy - 0.16, w: 2.0, h: 0.32, size: 14, valign: "middle", bold: r.pooled });
    });
  };
  const hRows = order.map((t) => ({ label: t, v: P.tasks[t].H, paper: P.tasks[t].paper_global }));
  const tRows = order.map((t) => ({ label: t, v: P.tasks[t].T }));
  const parse = (str) => { const [k, n] = str.split("/").map(Number); return { k, n }; };
  const pt = P.pooled.T;
  tRows.push({ label: "Pooled", pooled: true, v: { BASELINE: parse(pt.BASELINE), PAPER_GLOBAL: parse(pt.PAPER_GLOBAL), RANDOM_CONTROL: parse(pt.RANDOM_CONTROL) } });
  block(1.35, "H", "Arm H · states 0–14 (fit states) · 2 × 15", hRows, true);
  block(3.2, "T", "Arm T · held-out states 15–44 · 2 × 30", tRows, false);
  const axY = 3.2 + 0.55 + 4 * rowH - 0.1;
  [0.2, 0.4, 0.6, 0.8, 1.0].forEach((t) => {
    vline(s, px(t), axY - 0.06, 0.06, K.dim, 0.75);
    text(s, t.toFixed(1), { x: px(t) - 0.3, y: axY + 0.02, w: 0.6, h: 0.24, size: 12, color: K.dim, align: "center" });
  });
  text(s, "success rate (pooled over repeats)", { x: px0, y: axY + 0.26, w: pw, h: 0.26, size: 12, color: K.dim, align: "center" });
  const ly = 6.1;
  legendItem(s, lx, ly, K.base, "Baseline", "dot");
  legendItem(s, lx + 1.35, ly, K.ours, "Paper Global", "dot");
  legendItem(s, lx + 3.0, ly, K.control, "Random control", "ring");
  s.addShape(pres.shapes.LINE, { x: lx + 4.85, y: ly + 0.02, w: 0, h: 0.26, line: { color: K.paper, width: 3 } });
  text(s, "Paper Global (as /15)", { x: lx + 4.97, y: ly, w: 2.2, h: 0.28, size: S.label, color: K.muted, valign: "middle" });

  // PGG bars
  const rx = 8.85, rw = W - MX - rx;
  text(s, "PGG = G_{H} − G_{T}", { x: rx, y: 1.35, w: rw, h: 0.32, size: 15, bold: true, font: F.math });
  const zx = rx + rw / 2, scale = (rw / 2 - 0.35) / 0.2;
  vline(s, zx, 1.8, 1.45, K.dim, 1);
  order.forEach((t, i) => {
    const v = P.tasks[t].PGG, y = 1.85 + i * 0.46, bw = Math.abs(v) * scale;
    box(s, v >= 0 ? zx : zx - bw, y, bw, 0.32, { fill: v >= 0 ? K.ours : K.control, square: true });
    const lab = `${t} ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
    text(s, lab, { x: v >= 0 ? zx - 1.75 : zx + 0.08, y, w: 1.65, h: 0.32, size: 13, align: v >= 0 ? "right" : "left", valign: "middle", color: K.text });
  });
  text(s, c.stats, { x: rx, y: 3.45, w: rw, h: 2.9, size: 14, color: K.muted, p: { bullet: { indent: 12 }, paraSpaceAfter: 7 } });
  text(s, c.caveat, { x: rx, y: 6.2, w: rw, h: 0.5, size: 10.5, color: K.dim });
};

L.conclusion = (s, c) => {
  const cols = [
    ["SUPPORTED", K.green, K.greenDeep, c.supported, 5.3, 15],
    ["OPEN", K.yellow, K.yellowDeep, Y.open_questions, 3.65, 14],
    ["NOT CLAIMED", K.red, K.redDeep, Y.not_claimed.map((t) => t[0].toUpperCase() + t.slice(1)), 2.8, 16],
  ];
  let x = MX;
  cols.forEach(([h, col, deep, items, w, sz]) => {
    box(s, x, 1.4, w, 5.0, { fill: K.card, line: col });
    box(s, x, 1.4, w, 0.5, { fill: deep, line: col });
    text(s, h, { x: x + 0.2, y: 1.4, w: w - 0.4, h: 0.5, size: 15, bold: true, color: col, valign: "middle", charSpacing: 2 });
    const list = items.map((t, i) => (h === "SUPPORTED" ? `**${i + 1}.** ${t}` : t));
    text(s, list, { x: x + 0.22, y: 2.05, w: w - 0.42, h: 4.25, size: sz, color: h === "SUPPORTED" ? K.text : K.muted, p: h === "SUPPORTED" ? { paraSpaceAfter: 10 } : { bullet: { indent: 12 }, paraSpaceAfter: 8 } });
    x += w + 0.19;
  });
};

L.authors = (s) => {
  const A = Y.author_questions;
  chip(s, MX, 1.3, `STATUS · ${A.status.split(" (")[0].toUpperCase()}`, K.yellow, { size: 12 });
  const gw = 0.22, cw = (CW - gw) / 2, ch = 1.85;
  A.items.forEach((it, i) => {
    const x = MX + (i % 2) * (cw + gw), y = 1.85 + Math.floor(i / 2) * (ch + 0.2);
    box(s, x, y, cw, ch, { fill: K.card });
    text(s, String(i + 1), { x: x + 0.22, y: y + 0.2, w: 0.5, h: 0.5, size: 26, bold: true, color: K.ours });
    text(s, it.q, { x: x + 0.8, y: y + 0.22, w: cw - 1.0, h: 0.4, size: 18, bold: true });
    text(s, it.text, { x: x + 0.8, y: y + 0.7, w: cw - 1.0, h: ch - 0.8, size: 15.5, color: K.muted });
  });
  text(s, A.framing, { x: MX, y: 6.05, w: CW, h: 0.4, size: 18, italic: true, color: K.text });
};

L.next = (s, c) => {
  const lw = 5.75;
  box(s, MX, 1.4, lw, 5.0, { fill: K.card, line: K.ours });
  chip(s, MX + 0.25, 1.6, "PLANNED · PHASE 7B", K.ours, { size: 12 });
  text(s, Y.next_steps, { x: MX + 0.25, y: 2.15, w: lw - 0.5, h: 2.6, size: 16, p: { bullet: { indent: 14 }, paraSpaceAfter: 10 } });
  box(s, MX + 0.25, 4.85, lw - 0.5, 1.3, { fill: K.cardHi });
  text(s, "**Success criterion for any future method:** beat baseline **and** matched random / shrinkage controls on disjoint held-out states, with repeats.", {
    x: MX + 0.45, y: 4.9, w: lw - 0.9, h: 1.2, size: 14.5, color: K.text, valign: "middle",
  });
  const rx = MX + lw + 0.35, rw = W - MX - rx;
  box(s, rx, 1.4, rw, 5.0, { fill: K.bg, line: K.yellow, dash: "dash", lineW: 1.5 });
  chip(s, rx + 0.25, 1.6, c.future_badge, K.yellow, { size: 12 });
  text(s, c.future_title, { x: rx + 0.25, y: 2.1, w: rw - 0.5, h: 0.45, size: 20, bold: true });
  text(s, ["**Current**   h′ = (1 − β) h + β C h  ≈  (1 − β) h", "**Possible**   h′ = h + γ D(h)"], {
    x: rx + 0.25, y: 2.65, w: rw - 0.5, h: 0.95, size: 17, font: F.math, p: { paraSpaceAfter: 8 },
  });
  text(s, [
    "preserve the hidden-state norm (‖h′‖ ≈ ‖h‖)",
    "isolate a success-specific direction D(h)",
    "avoid uniform contraction",
    "evaluate on held-out states against matched controls",
  ], { x: rx + 0.25, y: 3.75, w: rw - 0.5, h: 2.1, size: 15, color: K.muted, p: { bullet: { indent: 14 }, paraSpaceAfter: 7 } });
  text(s, "A direction for discussion — no such method has been implemented or tested.", { x: rx + 0.25, y: 5.85, w: rw - 0.5, h: 0.4, size: 12.5, color: K.yellow });
};

L.takeaway = (s, c) => {
  s.background = { color: K.bgDeep };
  kicker(s, c.kicker);
  c.lines.forEach((l, i) => {
    const y = 1.2 + i * 1.8;
    text(s, String(i + 1), { x: MX, y, w: 0.8, h: 0.9, size: 44, bold: true, color: K.ours });
    text(s, l, { x: MX + 1.0, y: y + 0.08, w: CW - 1.0, h: 1.55, size: 22, color: K.text });
  });
};

// ------------------------------------------------------------------ assemble
const N = Y.slides.length;
const notesMd = [`# Speaker notes — ${Y.meta.title}`, "", `Updated through Phase ${Y.meta.latest_phase} · generated by build_slides.js from presentation_data.yaml`, ""];
Y.slides.forEach((c, i) => {
  if (!L[c.id]) throw new Error(`No layout for slide id "${c.id}"`);
  const s = pres.addSlide();
  s.background = { color: K.bg };
  if (c.id !== "title" && c.id !== "takeaway") {
    kicker(s, c.kicker);
    title(s, c.title);
  }
  L[c.id](s, c);
  if (c.source) source(s, c.source);
  if (c.id !== "title") footer(s, i + 1, N);
  const n = c.notes;
  const heading = c.title || (c.id === "title" ? Y.meta.title : c.kicker);
  s.addNotes(`MAIN POINT: ${tpl(n.point)}\n\nSAY: ${tpl(n.say)}\n\nLIKELY QUESTION: ${tpl(n.q)}\nANSWER: ${tpl(n.a)}`);
  notesMd.push(`## ${i + 1}. ${tpl(heading)}`, "", `**Main point.** ${tpl(n.point)}`, "", `**What to say.** ${tpl(n.say)}`, "", `**Likely question.** “${tpl(n.q)}”`, "", `**Answer.** ${tpl(n.a)}`, "");
});

fs.writeFileSync(path.join(HERE, "speaker_notes.md"), notesMd.join("\n"));
pres.writeFile({ fileName: path.join(HERE, Y.meta.output_file) }).then((f) => console.log(`wrote ${path.relative(HERE, f)} (${N} slides)`));
