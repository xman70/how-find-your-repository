/* AI Text Analysis - front end. All DOM text is set with textContent (no HTML injection). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { result: null, file: null, mode: "paste", selected: null };

  // ------------------------------------------------------------ helpers
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "style") n.setAttribute("style", v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    (children || []).forEach((c) => n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${Math.round(100 * v)}%`);
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : (typeof v === "number" ? v.toFixed(d) : String(v)));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function hexToRgb(h) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16)); }
  function mix(a, b, t) { const A = hexToRgb(a), B = hexToRgb(b); return "rgb(" + A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(",") + ")"; }
  function diverging(p) {
    p = Math.min(1, Math.max(0, p));
    return p < 0.5 ? mix(css("--div-low"), css("--div-mid"), p / 0.5) : mix(css("--div-mid"), css("--div-high"), (p - 0.5) / 0.5);
  }
  function highlight(p) {
    const a = Math.min(0.55, Math.abs(p - 0.5) * 2 * 0.55);
    return `rgba(${p >= 0.5 ? css("--hl-high") : css("--hl-low")},${a.toFixed(3)})`;
  }
  async function api(path, opts) {
    const r = await fetch(path, opts);
    let body = null;
    try { body = await r.json(); } catch (e) { body = null; }
    if (!r.ok) throw new Error((body && (body.detail || body.message)) || `Request failed (${r.status})`);
    return body;
  }
  function showError(msg) { const b = $("error-box"); b.textContent = msg; b.classList.remove("hidden"); }
  function clearError() { $("error-box").classList.add("hidden"); }

  // ------------------------------------------------------------ input
  document.querySelectorAll("[data-input]").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll("[data-input]").forEach((x) => x.classList.toggle("active", x === t));
    state.mode = t.dataset.input;
    $("pane-paste").classList.toggle("hidden", state.mode !== "paste");
    $("pane-upload").classList.toggle("hidden", state.mode !== "upload");
  }));
  $("text-input").addEventListener("input", () => {
    const n = ($("text-input").value.match(/[\p{L}\p{M}']+/gu) || []).length;
    $("word-count").textContent = `${n} words`;
  });
  function setFile(f) { state.file = f; $("file-label").textContent = f ? `${f.name} (${Math.round(f.size / 1024)} KB)` : "Choose a file or drop it here"; }
  $("file-input").addEventListener("change", (e) => setFile(e.target.files[0] || null));
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

  $("analyze-btn").addEventListener("click", async () => {
    clearError();
    const btn = $("analyze-btn");
    btn.disabled = true; $("progress").classList.remove("hidden");
    try {
      let res;
      if (state.mode === "upload") {
        if (!state.file) throw new Error("Choose a file first.");
        const fd = new FormData();
        fd.append("file", state.file);
        if ($("title-input").value) fd.append("title", $("title-input").value);
        res = await api("/api/analyze-file", { method: "POST", body: fd });
      } else {
        const text = $("text-input").value;
        if (!text.trim()) throw new Error("Paste some text first.");
        res = await api("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, title: $("title-input").value || null }) });
      }
      render(res);
    } catch (e) { showError(e.message); }
    finally { btn.disabled = false; $("progress").classList.add("hidden"); }
  });

  // ------------------------------------------------------------ rendering
  function render(res) {
    state.result = res; state.selected = null;
    const out = $("results");
    out.classList.remove("hidden");
    const warn = $("warnings"); warn.replaceChildren();
    (res.warnings || []).forEach((w) => warn.appendChild(el("div", { class: "notice", text: w })));
    renderLimitations(res);
    if (res.status !== "ok") {
      warn.appendChild(el("div", { class: "notice", text: `${statusLabel(res.status)}: ${res.message || ""}` }));
      $("tiles").replaceChildren(); $("interpretation").replaceChildren();
      document.querySelectorAll(".results-tabs, .tab-pane").forEach((n) => n.classList.add("hidden"));
      return;
    }
    document.querySelector(".results-tabs").classList.remove("hidden");
    selectTab("heatmap");
    renderTiles(res); renderDoc(res); renderModels(res); renderStats(res); renderTransitions(res); renderExport(res);
    drawCharts();
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function statusLabel(s) {
    return { insufficient_text: "Insufficient text", unsupported_language: "Unsupported language", model_missing: "Model missing", error: "Error" }[s] || s;
  }
  function tile(v, l, sub, isText) {
    return el("div", { class: "tile" }, [el("div", { class: "v" + (isText ? " text" : ""), text: v }), el("div", { class: "l", text: l }), sub ? el("div", { class: "sub", text: sub }) : el("span")]);
  }
  function renderTiles(res) {
    const r = res.result, m = res.meta;
    $("tiles").replaceChildren(
      tile(pct(r.ai_associated_share), "AI-associated text", "share of words"),
      tile(pct(r.human_associated_share), "Human-associated text", "share of words"),
      tile(pct(r.uncertain_share), "Uncertain / mixed", "share of words"),
      tile(pct(r.document_probability), "Document-level AI probability", "calibrated"),
      tile(pct(r.confidence), `Confidence: ${r.confidence_level}`, `Length evidence: ${r.evidence_text}`),
      tile(r.hybrid.label, "Category", `est. AI share ${pct(r.estimated_ai_share)}`, true),
      tile(String(m.word_count), "Words", m.language ? `language: ${m.language}` : ""),
      tile(String(m.sentence_count), "Sentences", ""),
      tile(String(m.paragraph_count), "Paragraphs", "")
    );
    const box = $("interpretation");
    box.replaceChildren(
      el("p", {}, [el("strong", { text: "Interpretation. " }), r.interpretation]),
      el("p", { class: "muted small", text: `${r.hybrid.rationale} ${r.share_definition}` }),
      el("p", { class: "muted small", text: "Confidence is reported separately from likelihood: a high AI likelihood with low confidence means the text resembles AI-generated writing but the evidence is limited." })
    );
  }
  function renderDoc(res) {
    const view = $("doc-view"); view.replaceChildren();
    const paras = new Map();
    res.sentences.forEach((s) => { if (!paras.has(s.paragraph)) paras.set(s.paragraph, []); paras.get(s.paragraph).push(s); });
    for (const sents of paras.values()) {
      const p = el("p");
      sents.forEach((s) => {
        const span = el("span", { class: "sent", tabindex: "0", role: "button", "data-i": String(s.index),
          title: `Sentence ${s.number} · AI likelihood ${pct(s.probability)} · confidence ${s.confidence}` },
          [el("sup", { text: String(s.number) }), s.text]);
        span.style.background = highlight(s.probability);
        span.addEventListener("click", () => selectSentence(s.index));
        span.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectSentence(s.index); } });
        p.appendChild(span); p.appendChild(document.createTextNode(" "));
      });
      view.appendChild(p);
    }
    $("sentence-detail").replaceChildren(el("p", { class: "muted", text: "Select a sentence to see its AI likelihood, confidence and the statistical signals behind it." }));
  }
  function selectSentence(i) {
    const res = state.result, s = res.sentences[i];
    state.selected = i;
    document.querySelectorAll(".sent").forEach((n) => n.classList.toggle("selected", Number(n.dataset.i) === i));
    const d = $("sentence-detail");
    const kids = [el("h3", { text: `Sentence ${s.number}` }),
      el("div", { class: "big", text: `AI likelihood: ${pct(s.probability)}` }),
      el("div", { text: `Confidence: ${s.confidence} · ${s.level}` }),
      el("p", { class: "muted small", text: `"${s.text.length > 220 ? s.text.slice(0, 220) + "…" : s.text}"` })];
    const ai = s.signals.toward_ai || [], hu = s.signals.toward_human || [];
    if (ai.length) {
      kids.push(el("strong", { text: s.band === "ai" ? "Factors contributing to the elevated score:" : "Strongest signals toward AI-associated patterns (score not elevated):" }));
      kids.push(el("ul", {}, ai.map((f) => el("li", { text: f.text + (f.human_percentile !== null ? ` (human percentile ${Math.round(f.human_percentile)})` : "") }))));
    }
    if (hu.length) {
      kids.push(el("strong", { text: "Signals pointing toward human-associated writing:" }));
      kids.push(el("ul", {}, hu.map((f) => el("li", { text: f.text }))));
    }
    kids.push(el("p", { class: "muted small", text: "These are statistical indicators, not proof of AI authorship. Sentence scores use a 5-sentence context window." }));
    const tbl = el("table", { class: "data" }, [el("tr", {}, [el("th", { text: "Detector (this sentence)" }), el("th", { class: "num", text: "AI prob." })])]);
    Object.entries(s.detectors).forEach(([k, v]) => tbl.appendChild(el("tr", {}, [el("td", { text: k }), el("td", { class: "num", text: pct(v) })])));
    kids.push(tbl);
    d.replaceChildren(...kids);
  }
  function renderModels(res) {
    $("disagreement-note").textContent = res.disagreement.note + ` Highest: ${res.disagreement.most_ai}; lowest: ${res.disagreement.most_human}.`;
    const t = $("models-table");
    t.replaceChildren(el("tr", {}, ["Detector", "AI probability", "± error", "Confidence", "Weight", "Validation AUC"].map((h, i) => el("th", { class: i ? "num" : "", text: h }))));
    res.detectors.forEach((d) => t.appendChild(el("tr", {}, [el("td", { text: d.label }), el("td", { class: "num", text: pct(d.probability) }),
      el("td", { class: "num", text: fmt(d.error_estimate) }), el("td", { class: "num", text: pct(d.confidence) }),
      el("td", { class: "num", text: fmt(d.weight, 2) }), el("td", { class: "num", text: fmt(d.validation_auc) })])));
    t.appendChild(el("tr", {}, [el("td", {}, [el("strong", { text: "Ensemble (calibrated)" })]), el("td", { class: "num" }, [el("strong", { text: pct(res.result.document_probability) })]), el("td"), el("td"), el("td"), el("td")]));
  }
  function renderStats(res) {
    const c = $("stats-container"); c.replaceChildren();
    c.appendChild(el("p", { class: "muted small", text: "Percentiles compare each value with human-written reference documents used in training (50 = typical human value)." }));
    for (const [group, g] of Object.entries(res.statistics)) {
      const card = el("div", { class: "card" }, [el("h3", { text: group.charAt(0).toUpperCase() + group.slice(1) })]);
      if (g.summary) card.appendChild(el("p", { class: "small", text: Object.entries(g.summary).map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "number" ? fmt(v) : v}`).join(" · ") }));
      const t = el("table", { class: "data" }, [el("tr", {}, ["Measure", "Value", "Human percentile", "Human median", "AI median"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))]);
      g.items.forEach((it) => t.appendChild(el("tr", {}, [el("td", { text: it.label }), el("td", { class: "num", text: it.display }),
        el("td", { class: "num", text: it.human_percentile === null ? "n/a" : String(Math.round(it.human_percentile)) }),
        el("td", { class: "num", text: fmt(it.human_median) }), el("td", { class: "num", text: fmt(it.ai_median) })])));
      card.appendChild(el("div", { class: "table-wrap" }, [t]));
      if (g.note) card.appendChild(el("p", { class: "muted small", text: g.note }));
      c.appendChild(card);
    }
  }
  function renderTransitions(res) {
    const box = $("transitions-list"); box.replaceChildren();
    const cps = res.transitions.changepoints || [];
    if (!cps.length) { box.appendChild(el("p", { class: "muted", text: "No statistically significant style transition was detected." })); return; }
    box.appendChild(el("p", { class: "muted small", text: "A potential authorship/style transition marks a statistically unusual change in writing characteristics (permutation test). It does not mean that AI use begins or ends there." }));
    const t = el("table", { class: "data" }, [el("tr", {}, ["Before sentence", "p-value", "Style (JS)", "Perplexity Δ", "Vocabulary Δ", "Syntax Δ", "Semantic distance", "AI likelihood before → after"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))]);
    cps.forEach((c) => { const d = c.differences; t.appendChild(el("tr", {}, [el("td", { text: `Potential transition before sentence ${c.boundary_before_sentence + 1}` }),
      el("td", { class: "num", text: fmt(c.p_value) }), el("td", { class: "num", text: fmt(d.style_js) }), el("td", { class: "num", text: fmt(d.perplexity_bits) }),
      el("td", { class: "num", text: fmt(d.vocabulary_mattr) }), el("td", { class: "num", text: fmt(d.syntax_depth) }), el("td", { class: "num", text: fmt(d.semantic_distance) }),
      el("td", { class: "num", text: `${pct(c.ai_probability_before)} → ${pct(c.ai_probability_after)}` })])); });
    box.appendChild(el("div", { class: "table-wrap" }, [t]));
  }
  function renderLimitations(res) {
    $("limitations").replaceChildren(...(res.limitations || []).map((l) => el("li", { text: l })), el("li", { text: res.disclaimer || "" }));
  }
  function renderExport(res) {
    ["pdf", "html", "json"].forEach((f) => { $(`export-${f}`).href = res.id ? `/api/analyses/${res.id}/report.${f}` : "#"; });
  }

  // ------------------------------------------------------------ charts
  function baseLayout(title, xTitle, yTitle) {
    return {
      title: { text: title, x: 0, xanchor: "left", font: { size: 14, color: css("--ink") } },
      paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
      font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css("--ink2"), size: 12 },
      margin: { l: 56, r: 16, t: 44, b: 44 }, height: 280, showlegend: false,
      xaxis: { title: { text: xTitle }, gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") } },
      yaxis: { title: { text: yTitle }, gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") } },
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { color: css("--ink") } },
    };
  }
  const plotCfg = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  function lineChart(id, ys, title, yTitle, opts = {}) {
    const xs = opts.x || ys.map((_, i) => i + 1);
    const tr = { x: xs, y: ys, type: "scatter", mode: opts.markerColors ? "lines+markers" : "lines", line: { color: opts.markerColors ? css("--ink2") : css("--series-1"), width: 2 },
      hovertemplate: opts.hover || `Sentence %{x}<br>${yTitle}: %{y:.3f}<extra></extra>` };
    if (opts.markerColors) tr.marker = { size: 9, color: opts.markerColors, line: { color: css("--surface"), width: 2 } };
    const layout = baseLayout(title, opts.xTitle || "Sentence", yTitle);
    if (opts.yrange) layout.yaxis.range = opts.yrange;
    layout.shapes = (opts.refs || []).map((r) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: r.y, y1: r.y, line: { color: css("--muted"), width: 1, dash: "dot" } }));
    layout.annotations = (opts.refs || []).map((r) => ({ xref: "paper", x: 1, y: r.y, text: r.label, showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: css("--ink2") } }));
    Plotly.react(id, [tr], layout, plotCfg);
  }
  function drawCharts() {
    const res = state.result; if (!res || res.status !== "ok") return;
    const s = res.series, b = res.result.bands;
    const sp = s.sentence_probability;
    const colors = sp.map(diverging);
    lineChart("chart-sentprob", sp, "Sentence AI likelihood (calibrated)", "AI probability",
      { markerColors: colors, yrange: [0, 1], refs: [{ y: b.t_high, label: "AI-associated band" }, { y: b.t_low, label: "human-associated band" }],
        hover: "Sentence %{x}<br>AI likelihood: %{y:.0%}<extra></extra>" });
    const unit = res.model.predictability_backend === "unigram" ? "bits/word (unigram proxy)" : "bits/token";
    lineChart("chart-perplexity", s.sentence_surprisal, "Perplexity per sentence (mean surprisal)", unit);
    lineChart("chart-burstiness", s.rolling_perplexity_std, "Burstiness: local variation of predictability", "std of surprisal");
    if (s.style_drift.values.length) lineChart("chart-drift", s.style_drift.values, "Style drift between consecutive windows", "JS divergence",
      { x: s.style_drift.centers.map((c) => c + 1), xTitle: "Window centre (sentence)" });
    else Plotly.react("chart-drift", [], Object.assign(baseLayout("Style drift (text too short)", "", ""), {}), plotCfg);
    lineChart("chart-vocab", s.rolling_mattr, "Vocabulary diversity (MATTR, 5-sentence window)", "MATTR");
    const lens = s.sentence_lengths;
    Plotly.react("chart-lengths", [{ x: lens, type: "histogram", xbins: { size: 5 }, marker: { color: css("--series-1"), line: { color: css("--surface"), width: 2 } },
      hovertemplate: "%{x} words: %{y} sentences<extra></extra>" }], Object.assign(baseLayout("Sentence-length distribution", "Words per sentence", "Sentences"), { bargap: 0.05 }), plotCfg);
    // heatmap grid
    const cols = Math.min(25, sp.length), rows = Math.ceil(sp.length / cols);
    const z = [], txt = [], hover = [];
    for (let r = 0; r < rows; r++) { z.push([]); txt.push([]); hover.push([]);
      for (let c = 0; c < cols; c++) { const i = r * cols + c;
        z[r].push(i < sp.length ? sp[i] : null); txt[r].push(i < sp.length ? String(i + 1) : "");
        hover[r].push(i < sp.length ? `Sentence ${i + 1}<br>AI likelihood: ${pct(sp[i])}<br>Confidence: ${res.sentences[i].confidence}` : ""); } }
    const hl = baseLayout("AI-likelihood heatmap (one cell per sentence; click a cell)", "", "");
    hl.height = 60 * rows + 110; hl.xaxis.visible = false; hl.yaxis.visible = false; hl.yaxis.autorange = "reversed";
    hl.margin = { l: 16, r: 16, t: 44, b: 16 };
    Plotly.react("chart-heatmap", [{ z, text: txt, texttemplate: "%{text}", customdata: hover, hovertemplate: "%{customdata}<extra></extra>", type: "heatmap",
      zmin: 0, zmax: 1, xgap: 3, ygap: 3, colorscale: [[0, css("--div-low")], [0.5, css("--div-mid")], [1, css("--div-high")]],
      colorbar: { title: { text: "AI likelihood", side: "right" }, tickformat: ".0%", thickness: 12, len: 1, lenmode: "fraction", outlinewidth: 0, tickvals: [0, 0.5, 1] } }], hl, plotCfg);
    const hm = $("chart-heatmap");
    hm.removeAllListeners && hm.removeAllListeners("plotly_click");
    hm.on("plotly_click", (ev) => { const p = ev.points[0]; const i = p.y * cols + p.x; if (i < sp.length) { selectSentence(i);
      const n = document.querySelector(`.sent[data-i="${i}"]`); if (n) n.scrollIntoView({ block: "center", behavior: "smooth" }); } });
    // disagreement
    const dets = res.detectors.slice().reverse();
    const dl = baseLayout("Model disagreement: individual detector outputs", "Calibrated AI probability", "");
    const ens = res.result.document_probability;
    const labels = dets.map((d) => d.label.split(" (")[0]);
    dl.height = 90 + 34 * dets.length; dl.margin.l = 230; dl.margin.r = 56; dl.margin.t = 64; dl.xaxis.range = [0, 1]; dl.xaxis.tickformat = ".0%";
    dl.shapes = [{ type: "line", x0: ens, x1: ens, yref: "paper", y0: 0, y1: 1, line: { color: css("--ink"), width: 1.5, dash: "dash" } }];
    dl.annotations = [{ x: ens, yref: "paper", y: 1.0, yanchor: "bottom", text: `ensemble ${pct(ens)}`, showarrow: false, font: { size: 11, color: css("--ink") } }]
      .concat(dets.map((d, i) => ({ xref: "paper", x: 1.01, xanchor: "left", y: labels[i], text: pct(d.probability), showarrow: false, font: { size: 11, color: css("--ink2") } })));
    const dtr = [{ type: "bar", orientation: "h", y: labels, x: dets.map((d) => d.probability),
      error_x: { type: "data", array: dets.map((d) => d.error_estimate), color: css("--ink2"), thickness: 1 },
      marker: { color: css("--series-1"), cornerradius: 4 },
      customdata: dets.map((d) => [d.weight, d.confidence]), hovertemplate: "%{y}<br>AI probability: %{x:.0%}<br>weight: %{customdata[0]:.2f}<br>confidence: %{customdata[1]:.0%}<extra></extra>" }];
    Plotly.react("chart-disagreement", dtr, dl, plotCfg);
    Plotly.react("chart-disagreement-2", dtr, dl, plotCfg);
    // change-point statistic
    const cp = res.transitions;
    const cl = baseLayout("Change-point statistic (style/feature change at each boundary)", "Boundary before sentence", "Statistic");
    if (cp.threshold) { cl.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: cp.threshold, y1: cp.threshold, line: { color: css("--muted"), dash: "dot", width: 1 } }];
      cl.annotations = [{ xref: "paper", x: 1, y: cp.threshold, text: "significance threshold (permutation test)", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: css("--ink2") } }]; }
    Plotly.react("chart-cp", [{ x: (cp.series || []).map((_, i) => i + 1), y: cp.series || [], type: "scatter", mode: "lines", line: { color: css("--series-1"), width: 2 },
      hovertemplate: "Before sentence %{x}<br>statistic %{y:.2f}<extra></extra>" }], cl, plotCfg);
  }

  // ------------------------------------------------------------ tabs
  function selectTab(name) {
    document.querySelectorAll(".results-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${name}`));
    if (name === "similarity") loadReferences();
    window.dispatchEvent(new Event("resize"));
  }
  document.querySelectorAll(".results-tabs .tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));

  // ------------------------------------------------------------ similarity
  function currentText() {
    const r = state.result;
    if (r && r.sentences) return r.sentences.map((s) => s.text).join(" ");
    return $("text-input").value;
  }
  $("similarity-btn").addEventListener("click", async () => {
    const box = $("similarity-results"); box.replaceChildren(el("p", { class: "muted", text: "Checking…" }));
    try {
      const r = await api("/api/similarity", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: currentText() }) });
      const kids = [el("p", {}, [el("strong", { text: `Exact-overlap share: ${pct(r.overall_exact_overlap)}` }), ` (reference corpus: ${r.reference_corpus_size} documents)`]),
        el("p", { class: "muted small", text: r.note })];
      if (!r.sources.length) kids.push(el("p", { class: "muted", text: "No overlapping reference documents found." }));
      r.sources.forEach((s) => {
        const m = el("div", { class: "match" }, [el("strong", { text: s.title }), el("div", { class: "small" }, [
          el("span", { class: "pill", text: `exact words ${s.exact_match_words}` }), el("span", { class: "pill", text: `5-gram containment ${pct(s.containment_5gram)}` }),
          el("span", { class: "pill", text: `3-gram ${pct(s.containment_3gram)}` }), el("span", { class: "pill", text: `document cosine ${fmt(s.document_cosine, 2)}` }),
          el("span", { class: "pill", text: `similar sentences ${s.semantic_sentence_matches.length}` })])]);
        s.exact_matches.slice(0, 5).forEach((x) => m.appendChild(el("div", { class: "small muted", text: `“${x.text}”` })));
        kids.push(m);
      });
      box.replaceChildren(...kids);
    } catch (e) { box.replaceChildren(el("p", { class: "error", text: e.message })); }
  });
  async function loadReferences() {
    try {
      const refs = await api("/api/references");
      const ul = $("ref-list");
      ul.replaceChildren(...(refs.length ? refs.map((r) => el("li", {}, [el("span", { class: "meta", text: `${r.title} · ${r.word_count} words · ${r.added_at}` }),
        el("button", { class: "ghost", type: "button", text: "Remove", onclick: async () => { await api(`/api/references/${r.id}`, { method: "DELETE" }); loadReferences(); } })]))
        : [el("li", { class: "muted", text: "The reference corpus is empty. Add documents you want to compare against." })]));
    } catch (e) { /* ignore */ }
  }
  $("ref-add-current").addEventListener("click", async () => {
    await api("/api/references", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: currentText(), title: $("ref-title").value || (state.result && state.result.meta.title) || "Reference" }) });
    loadReferences();
  });
  $("ref-file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    try { await api("/api/references/upload", { method: "POST", body: fd }); } catch (err) { alert(err.message); }
    e.target.value = ""; loadReferences();
  });

  // ------------------------------------------------------------ history, settings, theme
  $("history-btn").addEventListener("click", async () => {
    $("history-drawer").classList.remove("hidden");
    const items = await api("/api/analyses");
    $("history-list").replaceChildren(...(items.length ? items.map((a) => el("li", {}, [
      el("span", { class: "meta", text: `${a.created_at} · ${a.title || "Untitled"} · ${a.word_count || 0} words · ${a.status === "ok" ? "AI-assoc. " + pct(a.ai_share) : a.status}` }),
      el("span", {}, [el("button", { class: "ghost", type: "button", text: "Open", onclick: async () => { const r = await api(`/api/analyses/${a.id}`); render(r); $("history-drawer").classList.add("hidden"); } }),
        el("button", { class: "ghost", type: "button", text: "Delete", onclick: async () => { await api(`/api/analyses/${a.id}`, { method: "DELETE" }); $("history-btn").click(); } })])]))
      : [el("li", { class: "muted", text: "No analyses yet." })]));
  });
  $("history-close").addEventListener("click", () => $("history-drawer").classList.add("hidden"));
  $("device-select").addEventListener("change", async (e) => {
    try { const r = await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ device: e.target.value }) });
      if (r.note) showError(r.note); else clearError(); } catch (err) { showError(err.message); }
  });
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme");
    try { if (t) localStorage.setItem("theme", t); else localStorage.removeItem("theme"); } catch (e) { /* storage unavailable */ }
    if (state.result && state.result.status === "ok") { renderDoc(state.result); drawCharts(); if (state.selected !== null) selectSentence(state.selected); }
  }
  $("theme-btn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    applyTheme(cur === "dark" ? "light" : "dark");
  });
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) { /* ignore */ }

  api("/api/health").then((h) => {
    $("privacy-badge").textContent = h.privacy.split(".")[0];
    $("disclaimer-text").textContent = h.disclaimer;
    return api("/api/settings");
  }).then((s) => { $("device-select").value = s.device; if (s.note) showError(s.note); }).catch(() => {});
})();
