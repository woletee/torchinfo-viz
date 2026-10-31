(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("torchinfo-data").textContent);
  const layers = data.layers;
  const T = data.totals;
  const $ = (id) => document.getElementById(id);

  // ---------- Model ----------
  layers.forEach((l) => { l.children = []; });
  const roots = [];
  layers.forEach((l) => (l.parent == null ? roots : layers[l.parent].children).push(l));
  const root = roots.length === 1 ? roots[0] : {
    id: -1, name: data.model, var_name: "", depth: -1, children: roots, virtual: true,
    self_params: 0, num_params: T.total_params, macs: 0, output_bytes: 0, flags: { leaf: false },
  };
  const parentOf = (l) => {
    if (l === root) return null;
    if (l.parent == null) return root.virtual ? root : null;
    return layers[l.parent];
  };
  const byId = (id) => (id === -1 ? root : layers[id]);
  const pathOf = (l) => { const p = []; for (let n = l; n; n = parentOf(n)) p.unshift(n); return p; };
  const leaves = layers.filter((l) => l.flags.leaf);
  const outputBytes = leaves.reduce((s, l) => s + l.output_bytes, 0);

  const METRICS = {
    params: { label: "params", self: (l) => l.self_params || 0, fmt: fmtCount, total: T.total_params },
    macs: { label: "mult-adds", self: (l) => (l.flags.leaf ? l.macs : 0), fmt: fmtCount, total: T.total_mult_adds },
    memory: { label: "output memory", self: (l) => (l.flags.leaf ? l.output_bytes : 0), fmt: fmtBytes, total: outputBytes },
  };
  const cache = {};
  function subtree(metric) {
    if (cache[metric]) return cache[metric];
    const values = new Map();
    const self = METRICS[metric].self;
    const visit = (n) => {
      let v = self(n);
      for (const c of n.children) v += visit(c);
      values.set(n, v);
      return v;
    };
    visit(root);
    const maxSelf = layers.reduce((m, l) => Math.max(m, self(l)), 0) || 1;
    return (cache[metric] = { values, maxSelf, rootValue: values.get(root) || 1 });
  }

  const state = { metric: "params", view: "flow", selected: null, zoom: root, query: "", sort: null };
  const VIEWS = ["graph", "flow", "treemap", "table"];
  const expanded = new Set(layers.filter((l) => l.depth < 2).map((l) => l.id));

  // ---------- Formatting ----------
  function trim(n, d) { return n.toFixed(d).replace(/\.?0+$/, ""); }
  function fmtCount(n) {
    if (!n) return "0";
    const a = Math.abs(n);
    if (a >= 1e12) return trim(n / 1e12, 2) + "T";
    if (a >= 1e9) return trim(n / 1e9, 2) + "B";
    if (a >= 1e6) return trim(n / 1e6, 2) + "M";
    if (a >= 1e3) return trim(n / 1e3, 1) + "K";
    return String(n);
  }
  function fmtBytes(n) {
    if (!n) return "0 B";
    if (n >= 1e9) return trim(n / 1e9, 2) + " GB";
    if (n >= 1e6) return trim(n / 1e6, 2) + " MB";
    if (n >= 1e3) return trim(n / 1e3, 1) + " kB";
    return n + " B";
  }
  const pct = (v, t) => (t ? (100 * v) / t : 0);
  const fmtPct = (p) => (p >= 10 ? p.toFixed(0) : p >= 0.1 ? p.toFixed(1) : p > 0 ? "<0.1" : "0") + "%";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const shape = (s) => (Array.isArray(s) && s.length ? JSON.stringify(s).replace(/,/g, ", ") : "—");
  const varName = (l) => (l.var_name && l.var_name !== l.name ? l.var_name : "");
  const label = (l) => l.name + (varName(l) ? ` (${varName(l)})` : "");
  // Dotted attribute path as in model.named_modules(), e.g. "layer4.1.conv2".
  const modulePath = (l) => pathOf(l).filter((n) => n !== root && n.depth > 0).map((n) => n.var_name).join(".");
  const kernel = (l) => {
    const k = [].concat(l.kernel_size == null ? [] : l.kernel_size);
    return k.length ? k.join("×") : "";
  };
  const labelHtml = (l) => esc(l.name) + (varName(l) ? ` <i>${esc(varName(l))}</i>` : "");
  // Square-root scale so small layers stay visible next to a dominant one.
  const heat = (v, max) => `color-mix(in oklab, var(--heat) ${v > 0 ? Math.round(10 + 60 * Math.sqrt(Math.min(1, v / max))) : 0}%, var(--surface-2))`;
  // Treemap shade 0..1: the largest layer is solid ink (black, or white in dark mode).
  const shade = (v, max) => (v > 0 ? +(0.06 + 0.9 * Math.sqrt(Math.min(1, v / max))).toFixed(3) : 0);

  // ---------- Header ----------
  $("model-name").textContent = data.model;
  const stats = [["Total params", fmtCount(T.total_params)], ["Trainable", fmtCount(T.trainable_params)],
    ["Frozen", fmtCount(T.total_params - T.trainable_params)], ["Params size", fmtBytes(T.total_param_bytes)]];
  if (data.has_input) {
    stats.push(["Mult-adds", fmtCount(T.total_mult_adds)]);
    stats.push(["Est. total size", fmtBytes(T.total_input_bytes + T.total_output_bytes + T.total_param_bytes)]);
  }
  $("stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  document.querySelectorAll("#metrics button").forEach((b) => { b.disabled = !METRICS[b.dataset.metric].total; });

  // ---------- Tree ----------
  function renderTree() {
    const { values, rootValue } = subtree(state.metric);
    const fmt = METRICS[state.metric].fmt;
    const q = state.query.trim().toLowerCase();
    let keep = null;
    if (q) {
      keep = new Set();
      for (const l of layers) {
        if ((l.name + " " + l.var_name).toLowerCase().includes(q)) pathOf(l).forEach((n) => keep.add(n));
      }
    }
    const out = [];
    const walk = (n, d) => {
      if (keep && !keep.has(n)) return;
      const open = keep ? true : expanded.has(n.id);
      const caret = n.children.length ? `<button class="caret" data-toggle="${n.id}" aria-label="${open ? "Collapse" : "Expand"}">${open ? "▾" : "▸"}</button>` : `<span class="caret"></span>`;
      out.push(`<div class="row" data-id="${n.id}" style="--d:${d};--h:${heat(values.get(n), rootValue)}" title="${esc(label(n))}">${caret}<span class="dot"></span><span class="nm">${labelHtml(n)}</span><span class="num">${fmt(values.get(n))}</span></div>`);
      if (open) n.children.forEach((c) => walk(c, d + 1));
    };
    (root.virtual ? root.children : [root]).forEach((n) => walk(n, 0));
    $("tree").innerHTML = out.join("") || `<div class="empty">No layers match “${esc(state.query)}”</div>`;
  }
  $("tree").addEventListener("click", (e) => {
    const toggle = e.target.closest("[data-toggle]");
    if (toggle) {
      const id = Number(toggle.dataset.toggle);
      expanded.has(id) ? expanded.delete(id) : expanded.add(id);
      renderTree();
      markSelected();
      return;
    }
    const row = e.target.closest("[data-id]");
    if (row) select(byId(Number(row.dataset.id)), { from: "tree" });
  });
  $("search").addEventListener("input", (e) => { state.query = e.target.value; renderTree(); markSelected(); });

  // ---------- Flow ----------
  function renderFlow() {
    const m = METRICS[state.metric];
    const { maxSelf } = subtree(state.metric);
    if (!leaves.length) { $("view-flow").innerHTML = `<div class="empty">This model has no layers.</div>`; return; }
    const out = ['<div class="flow">'];
    let group = null;
    let prev = null;
    for (const l of leaves) {
      // Edges carry the tensor shape flowing between consecutive layers.
      const flowing = prev ? prev.output_size : l.input_size;
      if (prev || data.has_input) out.push(`<div class="edge">${data.has_input ? esc((prev ? "" : "input ") + shape(flowing)) : ""}</div>`);
      const g = pathOf(l).slice(0, -1).map(label).join(" › ");
      if (g !== group) { out.push(`<div class="group">${esc(g || data.model)}</div>`); group = g; }
      const meta = [modulePath(l) || null, kernel(l) ? `kernel ${kernel(l)}` : null,
        l.num_params ? `trainable: ${l.trainable}` : "no params", l.flags.recursive ? "shared weights" : null];
      out.push(`<div class="card" data-id="${l.id}"><div class="bar" style="--h:${heat(m.self(l), maxSelf)}"></div>
        <div><h3>${labelHtml(l)}</h3><div class="meta">${esc(meta.filter(Boolean).join(" · "))}</div></div>
        <div class="val">${m.fmt(m.self(l))}<small>${m.label}</small></div></div>`);
      prev = l;
    }
    out.push("</div>");
    $("view-flow").innerHTML = out.join("");
  }
  $("view-flow").addEventListener("click", (e) => {
    const card = e.target.closest("[data-id]");
    if (card) select(byId(Number(card.dataset.id)), { from: "flow" });
  });

  // ---------- Treemap ----------
  function worst(row, side) {
    let sum = 0, max = 0, min = Infinity;
    for (const r of row) { sum += r.area; max = Math.max(max, r.area); min = Math.min(min, r.area); }
    return Math.max((side * side * max) / (sum * sum), (sum * sum) / (side * side * min));
  }
  function layoutRow(row, box, out) {
    const sum = row.reduce((s, r) => s + r.area, 0);
    if (box.w >= box.h) {
      const w = sum / box.h;
      let y = box.y;
      for (const r of row) { const h = r.area / w; out.push({ ...r, x: box.x, y, w, h }); y += h; }
      return { x: box.x + w, y: box.y, w: box.w - w, h: box.h };
    }
    const h = sum / box.w;
    let x = box.x;
    for (const r of row) { const w = r.area / h; out.push({ ...r, x, y: box.y, w, h }); x += w; }
    return { x: box.x, y: box.y + h, w: box.w, h: box.h - h };
  }
  function squarify(items, box) {
    const total = items.reduce((s, i) => s + i.value, 0);
    const out = [];
    if (!total || box.w <= 0 || box.h <= 0) return out;
    const scale = (box.w * box.h) / total;
    const rest = items.map((i) => ({ ...i, area: i.value * scale }));
    let row = [];
    while (rest.length) {
      const side = Math.min(box.w, box.h);
      if (!row.length || worst(row.concat(rest[0]), side) <= worst(row, side)) row.push(rest.shift());
      else { box = layoutRow(row, box, out); row = []; }
    }
    if (row.length) layoutRow(row, box, out);
    return out;
  }
  function treemapItems(n, values) {
    const items = n.children.map((c) => ({ node: c, value: values.get(c) })).filter((i) => i.value > 0);
    const own = METRICS[state.metric].self(n);
    if (own > 0 && items.length) items.push({ node: n, value: own, own: true });
    if (own > 0 && !items.length) items.push({ node: n, value: own, own: false });
    return items.sort((a, b) => b.value - a.value);
  }
  function layoutBox(n, box, depth, out, ctx) {
    for (const r of squarify(treemapItems(n, ctx.values), box)) {
      const c = r.node;
      const style = `left:${r.x.toFixed(1)}px;top:${r.y.toFixed(1)}px;width:${Math.max(0, r.w - 2).toFixed(1)}px;height:${Math.max(0, r.h - 2).toFixed(1)}px`;
      const name = r.own ? `${labelHtml(c)} <small>own</small>` : labelHtml(c);
      const lb = r.w > 44 && r.h > 18 ? `<div class="lb">${name}<small>${ctx.fmt(r.value)}</small></div>` : "";
      const title = `${label(c)}${r.own ? " (own)" : ""}: ${ctx.fmt(r.value)} · ${fmtPct(pct(r.value, ctx.total))}`;
      const nested = !r.own && c !== n && c.children.length && r.w > 70 && r.h > 48 && depth < 3;
      if (nested) {
        out.push(`<div class="tm grp" data-id="${c.id}" data-zoom="1" style="${style}" title="${esc(title)}">${lb}</div>`);
        layoutBox(c, { x: r.x + 4, y: r.y + 21, w: r.w - 8, h: r.h - 25 }, depth + 1, out, ctx);
      } else {
        const value = r.own || !c.children.length ? ctx.self(c) : r.value;
        const t = shade(value, ctx.maxSelf);
        out.push(`<div class="tm${t > 0.5 ? " ink" : ""}" data-id="${c.id}" ${!r.own && c.children.length ? 'data-zoom="1"' : ""} style="${style};--t:${t}" title="${esc(title)}">${lb}</div>`);
      }
    }
  }
  function renderTreemap() {
    const el = $("treemap");
    const { width, height } = el.getBoundingClientRect();
    const crumbs = pathOf(state.zoom).map((n, i, all) => (i === all.length - 1
      ? `<strong>${esc(label(n))}</strong>` : `<button data-zoom-to="${n.id}">${esc(label(n))}</button><span>›</span>`));
    $("crumbs").innerHTML = crumbs.join("");
    if (!width || !height) return;
    const { values, maxSelf } = subtree(state.metric);
    const m = METRICS[state.metric];
    const ctx = { values, maxSelf, fmt: m.fmt, self: m.self, total: m.total || 1 };
    const out = [];
    layoutBox(state.zoom, { x: 0, y: 0, w: width, h: height }, 0, out, ctx);
    el.innerHTML = out.join("") || `<div class="empty">Nothing to show for ${m.label} here.</div>`;
    markSelected();
  }
  $("treemap").addEventListener("click", (e) => {
    const box = e.target.closest("[data-id]");
    if (!box) return;
    const node = byId(Number(box.dataset.id));
    if (box.dataset.zoom && node !== state.zoom) { state.zoom = node; renderTreemap(); }
    select(node, { from: "treemap" });
  });
  $("crumbs").addEventListener("click", (e) => {
    const b = e.target.closest("[data-zoom-to]");
    if (b) { state.zoom = byId(Number(b.dataset.zoomTo)); renderTreemap(); }
  });
  new ResizeObserver(() => { if (state.view === "treemap") renderTreemap(); }).observe($("treemap"));

  // ---------- Table ----------
  const COLUMNS = [
    { key: "layer", label: "Layer", value: (l) => l.id },
    { key: "input", label: "Input shape", value: (l) => shape(l.input_size), input: true },
    { key: "output", label: "Output shape", value: (l) => shape(l.output_size), input: true },
    { key: "params", label: "Params", value: (l) => l.num_params, num: true, show: (l) => l.num_params.toLocaleString() },
    { key: "share", label: "% params", value: (l) => subtree("params").values.get(l), num: true, show: (l) => fmtPct(pct(subtree("params").values.get(l), T.total_params)) },
    { key: "macs", label: "Mult-adds", value: (l) => subtree("macs").values.get(l), num: true, input: true, show: (l) => fmtCount(subtree("macs").values.get(l)) },
    { key: "memory", label: "Output memory", value: (l) => l.output_bytes, num: true, input: true, show: (l) => fmtBytes(l.output_bytes) },
    { key: "trainable", label: "Trainable", value: (l) => l.trainable },
  ].filter((c) => data.has_input || !c.input);
  function renderTable() {
    const sort = state.sort;
    const rows = layers.slice();
    if (sort) {
      const col = COLUMNS.find((c) => c.key === sort.key);
      rows.sort((a, b) => {
        const x = col.value(a), y = col.value(b);
        return (x < y ? -1 : x > y ? 1 : a.id - b.id) * sort.dir;
      });
    }
    const arrow = (c) => (sort && sort.key === c.key ? (sort.dir > 0 ? " ↑" : " ↓") : "");
    const head = COLUMNS.map((c) => `<th class="${c.num ? "n" : ""}" data-sort="${c.key}">${c.label}${arrow(c)}</th>`).join("");
    const body = rows.map((l) => `<tr data-id="${l.id}">${COLUMNS.map((c) => {
      if (c.key === "layer") {
        const indent = sort ? 0 : l.depth * 14;
        return `<td title="${esc(pathOf(l).map(label).join(" › "))}"><span style="padding-left:${indent}px">${labelHtml(l)}</span></td>`;
      }
      return `<td class="${c.num ? "n" : ""}">${esc(c.show ? c.show(l) : c.value(l))}</td>`;
    }).join("")}</tr>`).join("");
    $("view-table").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    markSelected();
  }
  $("view-table").addEventListener("click", (e) => {
    const th = e.target.closest("[data-sort]");
    if (th) {
      const key = th.dataset.sort;
      const cur = state.sort;
      if (key === "layer") state.sort = null;
      else if (!cur || cur.key !== key) state.sort = { key, dir: COLUMNS.find((c) => c.key === key).num ? -1 : 1 };
      else if (cur.dir === -1) state.sort = { key, dir: 1 };
      else state.sort = null;
      renderTable();
      return;
    }
    const tr = e.target.closest("tr[data-id]");
    if (tr) select(byId(Number(tr.dataset.id)), { from: "table" });
  });

  // ---------- Graph ----------
  const G = data.graph;
  const gv = { k: 1, x: 0, y: 0, layout: null, byKey: new Map(), dirty: true, fitted: false };
  const gnodes = G ? G.nodes.map((n) => ({
    ...n,
    layerObj: n.kind === "layer" ? layers[n.layer] : null,
    parentObj: n.kind === "op" && n.parent != null ? layers[n.parent] : null,
  })) : [];
  const gByKey = new Map(gnodes.map((n) => [n.key, n]));
  const graphOpen = new Set();
  const maxDepth = layers.reduce((m, l) => Math.max(m, l.depth), 1);
  let graphLevel = 1;
  const FONT_T = "600 13px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  const FONT_S = "11px ui-monospace, 'SF Mono', 'Cascadia Mono', Consolas, monospace";
  const CATS = [["conv", /conv/i], ["norm", /norm/i], ["attn", /attention|attn|mha/i],
    ["act", /relu|gelu|sigmoid|tanh|silu|swish|mish|softmax|softplus|elu|act/i],
    ["pool", /pool/i], ["linear", /linear|embedding|dense/i], ["drop", /dropout/i]];
  const LEGEND = { conv: "Convolution", norm: "Normalization", act: "Activation", pool: "Pooling",
    linear: "Linear / Embedding", attn: "Attention", drop: "Dropout", other: "Other layer",
    op: "Tensor op", group: "Collapsed block", io: "Input / Output" };
  const measure = document.createElement("canvas").getContext("2d");
  const textWidth = (text, font) => { measure.font = font; return measure.measureText(text).width; };
  const leafCounts = new Map();
  const leafCount = (l) => {
    if (!leafCounts.has(l)) leafCounts.set(l, l.children.length ? l.children.reduce((s, c) => s + leafCount(c), 0) : 1);
    return leafCounts.get(l);
  };
  const isContainer = (l) => !!l && !l.virtual && l !== root && l.children.length > 0;
  // First closed container on the path to `l`: that is the box `l` is drawn inside.
  const collapsedAncestor = (l) => pathOf(l).find((c) => isContainer(c) && !graphOpen.has(c.id)) || null;
  function graphKeyOf(n) {
    const owner = n.kind === "layer" ? n.layerObj : n.parentObj;
    const box = owner ? collapsedAncestor(owner) : null;
    return box ? `L${box.id}` : n.key;
  }
  function graphKeyOfLayer(l) {
    const box = collapsedAncestor(l);
    if (box) return `L${box.id}`;
    return gByKey.has(`L${l.id}`) ? `L${l.id}` : null;
  }
  const selectedGraphKey = () => {
    const s = state.selected;
    if (!s || !G) return null;
    return s.kind ? graphKeyOf(s) : graphKeyOfLayer(s);
  };
  const gLabel = (n) => {
    if (n.kind === "layer") return modulePath(n.layerObj) || label(n.layerObj);
    if (n.kind === "op") return n.name;
    return `${n.kind === "input" ? "Input" : "Output"} ${n.key.replace(/\D/g, "")}`;
  };
  function category(n) {
    if (n.kind === "input" || n.kind === "output") return "io";
    if (n.kind === "op" || n.kind === "group") return n.kind;
    const hit = CATS.find(([, re]) => re.test(n.layerObj.name));
    return hit ? hit[0] : "other";
  }
  function setGraphLevel(level) {
    graphLevel = Math.max(1, Math.min(maxDepth, level));
    graphOpen.clear();
    layers.forEach((l) => { if (isContainer(l) && l.depth < graphLevel) graphOpen.add(l.id); });
  }
  if (G) {
    // Open as many levels as possible while keeping the first view readable.
    let best = 1;
    for (let level = 1; level <= maxDepth; level++) {
      setGraphLevel(level);
      if (new Set(gnodes.map(graphKeyOf)).size > 45) break;
      best = level;
    }
    setGraphLevel(best);
  }

  function decorate(n) {
    const m = METRICS[state.metric];
    const { values } = subtree(state.metric);
    n.cat = category(n);
    n.value = "";
    if (n.layerObj) {
      const l = n.layerObj;
      const v = n.kind === "group" ? values.get(l) : m.self(l);
      n.title = l.name;
      n.sub = n.kind === "group" ? [varName(l), `${leafCount(l)} layers`].filter(Boolean).join(" · ")
        : [varName(l), kernel(l)].filter(Boolean).join(" · ");
      n.value = v ? m.fmt(v) : "";
    } else if (n.kind === "op") {
      n.title = n.name;
      n.sub = "";
    } else {
      n.title = gLabel(n);
      n.sub = shape(n.shape);
    }
    const extra = n.kind === "group" ? 20 : 0;
    const top = textWidth(n.title, FONT_T) + (n.value ? textWidth(n.value, FONT_S) + 14 : 0);
    const bottom = n.sub ? textWidth(n.sub, FONT_S) + extra : extra;
    n.w = Math.round(Math.min(300, Math.max(n.kind === "op" ? 56 : 112, Math.max(top, bottom) + 28)));
    n.h = n.kind === "op" ? 30 : n.sub || n.kind === "group" ? 46 : 34;
  }
  function visibleGraph() {
    const nodes = new Map();
    for (const n of gnodes) {
      const key = graphKeyOf(n);
      if (nodes.has(key)) continue;
      nodes.set(key, key === n.key ? { ...n, order: nodes.size }
        : { key, kind: "group", layerObj: byId(Number(key.slice(1))), order: nodes.size });
    }
    const edges = new Map();
    for (const e of G.edges) {
      const s = graphKeyOf(gByKey.get(e.src));
      const d = graphKeyOf(gByKey.get(e.dst));
      if (s !== d && !edges.has(`${s}>${d}`)) edges.set(`${s}>${d}`, { s, d, shape: e.shape });
    }
    nodes.forEach(decorate);
    return { nodes: [...nodes.values()], edges: [...edges.values()] };
  }

  // Layered (Sugiyama-style) layout: rank by longest path, route long edges through
  // invisible dummy nodes, reduce crossings with barycenter sweeps, then place x.
  function layoutGraph(vg) {
    const GAP_X = 26, GAP_Y = 46;
    const nodes = vg.nodes;
    const byKey = new Map(nodes.map((n) => [n.key, n]));
    const succ = new Map(nodes.map((n) => [n, []]));
    const pred = new Map(nodes.map((n) => [n, []]));
    for (const e of vg.edges) {
      e.sn = byKey.get(e.s); e.dn = byKey.get(e.d);
      succ.get(e.sn).push(e.dn); pred.get(e.dn).push(e.sn);
    }
    const indeg = new Map(nodes.map((n) => [n, pred.get(n).length]));
    const done = new Set();
    const ready = nodes.filter((n) => !indeg.get(n));
    let t = 0;
    while (done.size < nodes.length) {
      let n;
      if (ready.length) {
        let best = 0;
        for (let i = 1; i < ready.length; i++) if (ready[i].order < ready[best].order) best = i;
        n = ready.splice(best, 1)[0];
      } else n = nodes.find((x) => !done.has(x)); // a cycle from collapsing: break it
      if (done.has(n)) continue;
      done.add(n); n.t = t++;
      for (const d of succ.get(n)) {
        indeg.set(d, indeg.get(d) - 1);
        if (indeg.get(d) === 0 && !done.has(d)) ready.push(d);
      }
    }
    const topo = nodes.slice().sort((a, b) => a.t - b.t);
    for (const n of topo) {
      n.rank = 0;
      for (const p of pred.get(n)) if (p.t < n.t) n.rank = Math.max(n.rank, p.rank + 1);
    }
    for (const n of topo) { // side inputs (ids, masks) sit right above their first consumer
      const later = succ.get(n).filter((d) => d.t > n.t);
      if (!pred.get(n).length && later.length) n.rank = Math.max(0, Math.min(...later.map((d) => d.rank)) - 1);
    }
    const ranks = [];
    const put = (n) => { (ranks[n.rank] = ranks[n.rank] || []).push(n); };
    nodes.forEach(put);
    const up = new Map(), down = new Map();
    const link = (a, b) => {
      if (!down.has(a)) down.set(a, []);
      if (!up.has(b)) up.set(b, []);
      down.get(a).push(b); up.get(b).push(a);
    };
    for (const e of vg.edges) {
      e.chain = [e.sn];
      if (e.dn.rank <= e.sn.rank) { e.back = true; e.chain.push(e.dn); continue; }
      let prev = e.sn;
      for (let r = e.sn.rank + 1; r < e.dn.rank; r++) {
        const dummy = { dummy: true, rank: r, w: 0, h: 0, t: e.sn.t + 0.5 };
        put(dummy); link(prev, dummy); e.chain.push(dummy); prev = dummy;
      }
      link(prev, e.dn); e.chain.push(e.dn);
    }
    for (let r = 0; r < ranks.length; r++) ranks[r] = ranks[r] || [];
    ranks.forEach((r) => { r.sort((a, b) => a.t - b.t); r.forEach((n, i) => { n.o = i; }); });
    const sweep = (list, nb) => {
      for (const r of list) {
        for (const n of r) {
          const ns = nb.get(n);
          n.b = ns && ns.length ? ns.reduce((s, m) => s + m.o, 0) / ns.length : n.o;
        }
        r.sort((a, b) => a.b - b.b || a.o - b.o);
        r.forEach((n, i) => { n.o = i; });
      }
    };
    for (let i = 0; i < 6; i++) { sweep(ranks.slice(1), up); sweep(ranks.slice(0, -1).reverse(), down); }
    const gap = (a, b) => (a.w + b.w) / 2 + (a.dummy || b.dummy ? 14 : GAP_X);
    const place = (r, want) => {
      const L = want.slice(), R = want.slice();
      for (let i = 1; i < r.length; i++) L[i] = Math.max(L[i], L[i - 1] + gap(r[i - 1], r[i]));
      for (let i = r.length - 2; i >= 0; i--) R[i] = Math.min(R[i], R[i + 1] - gap(r[i], r[i + 1]));
      const X = L.map((x, i) => (x + R[i]) / 2);
      for (let i = 1; i < r.length; i++) X[i] = Math.max(X[i], X[i - 1] + gap(r[i - 1], r[i]));
      r.forEach((n, i) => { n.x = X[i]; });
    };
    for (const r of ranks) {
      let x = 0;
      const want = r.map((n, i) => { if (i) x += gap(r[i - 1], n); return x; });
      place(r, want.map((v) => v - x / 2));
    }
    const mean = (ns, fallback) => (ns && ns.length ? ns.reduce((s, m) => s + m.x, 0) / ns.length : fallback);
    for (let i = 0; i < 8; i++) {
      const downward = i % 2 === 0;
      for (const r of downward ? ranks : ranks.slice().reverse()) {
        place(r, r.map((n) => mean((downward ? up : down).get(n), n.x)));
      }
    }
    let y = 0;
    for (const r of ranks) {
      const h = r.reduce((m, n) => Math.max(m, n.h), 0);
      r.forEach((n) => { n.y = y + h / 2; });
      y += h + GAP_Y;
    }
    // Spread edge endpoints along a box's edge so parallel edges do not overlap.
    const outs = new Map(), ins = new Map();
    for (const e of vg.edges) {
      if (!outs.has(e.sn)) outs.set(e.sn, []);
      if (!ins.has(e.dn)) ins.set(e.dn, []);
      outs.get(e.sn).push(e); ins.get(e.dn).push(e);
    }
    const spread = (n, list, next, side) => {
      list.sort((a, b) => next(a).x - next(b).x);
      const step = Math.min(16, (n.w - 24) / Math.max(1, list.length));
      list.forEach((e, i) => { e[side] = n.x + (i - (list.length - 1) / 2) * step; });
    };
    outs.forEach((list, n) => spread(n, list, (e) => e.chain[1], "x0"));
    ins.forEach((list, n) => spread(n, list, (e) => e.chain[e.chain.length - 2], "x1"));
    for (const e of vg.edges) {
      const pts = e.chain.map((p) => ({ x: p.x, y: p.y }));
      const last = pts.length - 1;
      pts[0] = { x: e.x0, y: e.sn.y + e.sn.h / 2 };
      pts[last] = { x: e.x1, y: e.dn.y - e.dn.h / 2 - 2 };
      e.y0 = pts[0].y;
      if (e.back) {
        const bend = Math.max(pts[0].x, pts[last].x) + Math.max(e.sn.w, e.dn.w) / 2 + 40;
        e.path = `M${pts[0].x},${pts[0].y} C${bend},${pts[0].y + 80} ${bend},${pts[last].y - 80} ${pts[last].x},${pts[last].y}`;
        continue;
      }
      let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i], my = ((a.y + b.y) / 2).toFixed(1);
        d += ` C${a.x.toFixed(1)},${my} ${b.x.toFixed(1)},${my} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;
      }
      e.path = d;
    }
    const minX = nodes.reduce((m, n) => Math.min(m, n.x - n.w / 2), 0);
    const maxX = nodes.reduce((m, n) => Math.max(m, n.x + n.w / 2), 0);
    return { nodes, edges: vg.edges, minX, maxX, height: y };
  }

  function tooltip(n) {
    if (n.layerObj) return `${label(n.layerObj)}${modulePath(n.layerObj) ? ` — ${modulePath(n.layerObj)}` : ""}`;
    return `${n.title} → ${shape(n.shape)}`;
  }
  function drawGraph(L) {
    const out = [];
    for (const e of L.edges) out.push(`<path class="ge" data-s="${e.s}" data-d="${e.d}" d="${e.path}" marker-end="url(#ga)"/>`);
    const labelled = new Set();
    for (const e of L.edges) {
      const text = shape(e.shape);
      if (!e.shape || !e.shape.length || labelled.has(e.s + text)) continue;
      labelled.add(e.s + text);
      out.push(`<text class="gl" data-s="${e.s}" x="${(e.x0 + 6).toFixed(1)}" y="${(e.y0 + 14).toFixed(1)}">${esc(text)}</text>`);
    }
    for (const n of L.nodes) {
      const cls = n.kind === "input" || n.kind === "output" ? "io" : n.kind;
      const rx = n.kind === "op" ? n.h / 2 : 9;
      let body = `<title>${esc(tooltip(n))}</title><rect width="${n.w}" height="${n.h}" rx="${rx}"/>`;
      if (n.layerObj) body += `<rect class="stripe" x="5" y="7" width="4" height="${n.h - 14}" rx="2"/>`;
      if (n.kind === "op") {
        body += `<text class="t" x="${n.w / 2}" y="${n.h / 2 + 4.5}" text-anchor="middle">${esc(n.title)}</text>`;
      } else {
        const x = n.layerObj ? 16 : 12;
        body += `<text class="t" x="${x}" y="${n.sub || n.kind === "group" ? 19 : n.h / 2 + 4.5}">${esc(n.title)}</text>`;
        if (n.sub) body += `<text class="s" x="${x}" y="36">${esc(n.sub)}</text>`;
        if (n.value) body += `<text class="v" x="${n.w - 10}" y="19" text-anchor="end">${esc(n.value)}</text>`;
        if (n.kind === "group") body += `<text class="plus" x="${n.w - 10}" y="37" text-anchor="end">⊕</text>`;
      }
      out.push(`<g class="gn ${cls}" data-key="${n.key}" style="--c:var(--cat-${n.cat})" transform="translate(${(n.x - n.w / 2).toFixed(1)},${(n.y - n.h / 2).toFixed(1)})">${body}</g>`);
    }
    $("graph-root").innerHTML = out.join("");
    const present = new Set(L.nodes.map((n) => n.cat));
    $("graph-legend").innerHTML = Object.entries(LEGEND).filter(([c]) => present.has(c))
      .map(([c, text]) => `<span style="--c:var(--cat-${c})">${text}</span>`).join("");
  }
  const svg = $("graph-svg");
  const applyView = () => $("graph-root").setAttribute("transform", `translate(${gv.x.toFixed(1)},${gv.y.toFixed(1)}) scale(${gv.k.toFixed(4)})`);
  function fitGraph(whole) {
    const L = gv.layout;
    const { width, height } = svg.getBoundingClientRect();
    if (!L || !width) return false;
    const byWidth = (width - 40) / (L.maxX - L.minX + 40);
    const byHeight = (height - 90) / (L.height + 20);
    gv.k = Math.max(0.04, Math.min(1, byWidth, whole ? byHeight : Infinity));
    gv.x = (width - (L.maxX + L.minX) * gv.k) / 2;
    gv.y = 56;
    applyView();
    return true;
  }
  function zoomAt(mx, my, factor) {
    const k = Math.min(3, Math.max(0.04, gv.k * factor));
    gv.x = mx - (mx - gv.x) * (k / gv.k);
    gv.y = my - (my - gv.y) * (k / gv.k);
    gv.k = k;
    applyView();
  }
  function centerOn(key, force) {
    const n = key && gv.byKey.get(key);
    const { width, height } = svg.getBoundingClientRect();
    if (!n || !width) return;
    const sx = gv.x + n.x * gv.k, sy = gv.y + n.y * gv.k;
    if (!force && sx > 40 && sx < width - 40 && sy > 60 && sy < height - 40) return;
    if (gv.k < 0.55) gv.k = 0.85;
    gv.x = width / 2 - n.x * gv.k;
    gv.y = height / 2 - n.y * gv.k;
    applyView();
  }
  function markGraph() {
    if (!G) return;
    const g = $("graph-root");
    g.querySelectorAll(".sel, .hl").forEach((el) => el.classList.remove("sel", "hl"));
    g.querySelectorAll('.ge[marker-end="url(#gah)"]').forEach((el) => el.setAttribute("marker-end", "url(#ga)"));
    const key = selectedGraphKey();
    if (!key) return;
    const node = g.querySelector(`.gn[data-key="${key}"]`);
    if (node) node.classList.add("sel");
    g.querySelectorAll(`.ge[data-s="${key}"], .ge[data-d="${key}"]`).forEach((el) => {
      el.classList.add("hl");
      el.setAttribute("marker-end", "url(#gah)");
    });
    g.querySelectorAll(`.gl[data-s="${key}"]`).forEach((el) => el.classList.add("hl"));
  }
  function renderGraph() {
    if (!G) return;
    gv.layout = layoutGraph(visibleGraph());
    gv.byKey = new Map(gv.layout.nodes.map((n) => [n.key, n]));
    gv.dirty = false;
    drawGraph(gv.layout);
    $("graph-level").textContent = `Level ${graphLevel} of ${maxDepth}`;
    if (!gv.fitted) gv.fitted = fitGraph(false);
    markGraph();
  }
  function toggleGroup(l, open) {
    if (open) graphOpen.add(l.id); else graphOpen.delete(l.id);
    renderGraph();
    centerOn(selectedGraphKey() || graphKeyOfLayer(l), false);
  }
  function showInGraph(l) {
    for (const c of pathOf(l).slice(0, -1)) if (isContainer(c)) graphOpen.add(c.id);
    gv.dirty = true;
    setView("graph");
    centerOn(selectedGraphKey(), true);
  }
  if (!G) {
    $("graph-wrap").innerHTML = `<div class="gempty">${data.has_input
      ? "The data-flow graph could not be recorded for this model."
      : "The graph comes from a forward pass: pass input_size or input_data to summary() to see it."}</div>`;
  } else {
    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const r = svg.getBoundingClientRect();
        zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-Math.max(-60, Math.min(60, e.deltaY)) * 0.01));
      } else {
        gv.x -= e.deltaX; gv.y -= e.deltaY;
        applyView();
      }
    }, { passive: false });
    let drag = null;
    svg.addEventListener("pointerdown", (e) => {
      if (e.button === 0) drag = { x: e.clientX, y: e.clientY, gx: gv.x, gy: gv.y, moved: false };
    });
    window.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; svg.classList.add("dragging"); }
      if (drag.moved) { gv.x = drag.gx + dx; gv.y = drag.gy + dy; applyView(); }
    });
    window.addEventListener("pointerup", () => {
      svg.classList.remove("dragging");
      setTimeout(() => { drag = null; }, 0);
    });
    svg.addEventListener("click", (e) => {
      if (drag && drag.moved) return;
      const g = e.target.closest(".gn");
      if (!g) return;
      const n = gv.byKey.get(g.dataset.key);
      select(n.layerObj || gByKey.get(n.key), { from: "graph" });
    });
    svg.addEventListener("dblclick", (e) => {
      const g = e.target.closest(".gn.group");
      if (g) toggleGroup(gv.byKey.get(g.dataset.key).layerObj, true);
    });
    document.querySelector(".gtools").addEventListener("click", (e) => {
      const b = e.target.closest("[data-gact]");
      if (!b) return;
      const r = svg.getBoundingClientRect();
      const act = b.dataset.gact;
      if (act === "fit") fitGraph(true);
      else if (act === "in" || act === "out") zoomAt(r.width / 2, r.height / 2, act === "in" ? 1.25 : 0.8);
      else {
        setGraphLevel(graphLevel + (act === "more" ? 1 : -1));
        gv.fitted = false;
        renderGraph();
      }
    });
  }
  function graphLinks(key) {
    const item = (k, shp) => `<li><button class="link" data-gpick="${k}">${esc(gLabel(gByKey.get(k)))}</button><span>${esc(shape(shp))}</span></li>`;
    const ins = G.edges.filter((e) => e.dst === key).map((e) => item(e.src, e.shape)).join("");
    const outs = G.edges.filter((e) => e.src === key).map((e) => item(e.dst, e.shape)).join("");
    return (ins ? `<h4>Inputs from</h4><ul class="list">${ins}</ul>` : "")
      + (outs ? `<h4>Feeds into</h4><ul class="list">${outs}</ul>` : "");
  }
  function graphNodeDetails(n) {
    const title = n.kind === "op" ? `${esc(n.name)} <i>tensor op</i>` : esc(gLabel(n));
    const where = n.parentObj
      ? `<div class="share">Runs in the forward() of <button class="link" data-pick="${n.parentObj.id}">${esc(modulePath(n.parentObj) || label(n.parentObj))}</button></div>`
      : "";
    return `<h2>${title}</h2>${where}<dl><dt>Output shape</dt><dd>${esc(shape(n.shape))}</dd></dl>${graphLinks(n.key)}`;
  }

  // ---------- Details ----------
  function overview() {
    const m = METRICS[state.metric];
    const top = leaves.filter((l) => m.self(l) > 0).sort((a, b) => m.self(b) - m.self(a)).slice(0, 8);
    const list = top.map((l) => `<li><button class="link" data-pick="${l.id}" title="${esc(label(l))}">${esc(modulePath(l) || label(l))}</button><span>${m.fmt(m.self(l))} · ${fmtPct(pct(m.self(l), m.total))}</span></li>`).join("");
    return `<h2>${esc(data.model)}</h2>
      <div class="share">${layers.length} layers · ${leaves.length} leaf layers${data.has_input ? ` · input ${esc(shape(data.input_size))}` : ""}</div>
      <h4>Heaviest layers by ${m.label}</h4>
      ${list ? `<ul class="list">${list}</ul>` : `<div class="share">No ${m.label} recorded.</div>`}
      <div class="share" style="margin-top:16px">Select a layer in any view to inspect it.</div>`;
  }
  function renderDetails() {
    const l = state.selected;
    if (!l) { $("details").innerHTML = overview(); return; }
    if (l.kind) { $("details").innerHTML = graphNodeDetails(l); return; }
    const m = METRICS[state.metric];
    const { values } = subtree(state.metric);
    const share = pct(values.get(l), m.total);
    const crumbs = pathOf(l).slice(0, -1).map((n) => `<button class="link" data-pick="${n.id}">${esc(label(n))}</button><span>›</span>`).join("");
    const rows = [
      ["Type", l.name], ["Module path", modulePath(l) || "(root)"],
      ["Position", l.depth_index != null ? `depth ${l.depth}, #${l.depth_index}` : `depth ${l.depth}`],
    ];
    if (data.has_input) rows.push(["Input", shape(l.input_size)], ["Output", shape(l.output_size)]);
    if (kernel(l)) rows.push(["Kernel", kernel(l)]);
    rows.push(["Params", l.num_params.toLocaleString()]);
    if (!l.flags.leaf) rows.push(["Own params", (l.self_params || 0).toLocaleString()]);
    rows.push(["Trainable", l.trainable], ["Param memory", fmtBytes(l.param_bytes)]);
    if (data.has_input) rows.push(["Mult-adds", subtree("macs").values.get(l).toLocaleString()], ["Output memory", fmtBytes(l.output_bytes)]);
    const tags = [];
    if (l.flags.leaf) tags.push("leaf layer"); else tags.push(`${l.children.length} children`);
    if (l.flags.recursive) tags.push("shared / repeated module");
    const kids = l.children.slice(0, 40).map((c) => `<li><button class="link" data-pick="${c.id}">${esc(label(c))}</button><span>${m.fmt(values.get(c))}</span></li>`).join("");
    $("details").innerHTML = `${crumbs ? `<div class="path">${crumbs}</div>` : ""}
      <h2>${labelHtml(l)}</h2>
      <div class="share">${fmtPct(share)} of total ${m.label} (${m.fmt(values.get(l))})<div class="track"><div class="fill" style="width:${Math.min(100, share)}%"></div></div></div>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      <div>${tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
      ${kids ? `<h4>Children</h4><ul class="list">${kids}</ul>${l.children.length > 40 ? `<div class="share">…and ${l.children.length - 40} more</div>` : ""}` : ""}
      ${G && gByKey.has(`L${l.id}`) ? graphLinks(`L${l.id}`) : ""}
      <p class="actions">
        ${G && l !== root && !l.virtual ? `<button class="link" data-show-graph="${l.id}">Show in graph →</button>` : ""}
        ${G && isContainer(l) ? `<button class="link" data-gtoggle="${l.id}">${graphOpen.has(l.id) ? "Collapse" : "Expand"} this block in the graph</button>` : ""}
        ${l.children.length ? `<button class="link" data-zoom-in="${l.id}">Open in treemap →</button>` : ""}
      </p>`;
  }
  $("details").addEventListener("click", (e) => {
    const pick = e.target.closest("[data-pick]");
    if (pick) { select(byId(Number(pick.dataset.pick)), { from: "details" }); return; }
    const gpick = e.target.closest("[data-gpick]");
    if (gpick) { const n = gByKey.get(gpick.dataset.gpick); select(n.layerObj || n, { from: "details" }); return; }
    const show = e.target.closest("[data-show-graph]");
    if (show) { showInGraph(byId(Number(show.dataset.showGraph))); return; }
    const gtoggle = e.target.closest("[data-gtoggle]");
    if (gtoggle) {
      const l = byId(Number(gtoggle.dataset.gtoggle));
      toggleGroup(l, !graphOpen.has(l.id));
      renderDetails();
      return;
    }
    const zoom = e.target.closest("[data-zoom-in]");
    if (zoom) { state.zoom = byId(Number(zoom.dataset.zoomIn)); setView("treemap"); }
  });

  // ---------- Linking ----------
  function markSelected() {
    document.querySelectorAll(".sel").forEach((el) => el.classList.remove("sel"));
    markGraph();
    if (!state.selected || state.selected.kind) return;
    document.querySelectorAll(`[data-id="${state.selected.id}"]`).forEach((el) => el.classList.add("sel"));
  }
  function reveal(container) {
    const el = container.querySelector(`[data-id="${state.selected.id}"]`);
    if (el) el.scrollIntoView({ block: "nearest" });
  }
  function select(l, { from } = {}) {
    state.selected = l;
    if (l && !l.kind) {
      let hidden = false;
      for (const n of pathOf(l).slice(0, -1)) if (!expanded.has(n.id)) { expanded.add(n.id); hidden = true; }
      if (hidden) renderTree();
    }
    renderDetails();
    markSelected();
    if (!l) return;
    if (state.view === "graph" && from !== "graph") centerOn(selectedGraphKey(), false);
    if (l.kind) return; // graph-only node (tensor op, input, output)
    if (from !== "tree") reveal($("tree"));
    if (from !== state.view && (state.view === "flow" || state.view === "table")) reveal($(`view-${state.view}`));
  }
  function setView(view) {
    state.view = view;
    document.querySelectorAll("#views button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
    document.querySelectorAll(".view-pane").forEach((p) => p.classList.toggle("on", p.id === `view-${view}`));
    if (view === "treemap") renderTreemap();
    if (view === "graph") {
      if (gv.dirty || !gv.layout) renderGraph();
      else if (!gv.fitted) gv.fitted = fitGraph(false);
      centerOn(selectedGraphKey(), false);
    }
    if (state.selected && !state.selected.kind && (view === "flow" || view === "table")) reveal($(`view-${view}`));
  }
  $("views").addEventListener("click", (e) => { const b = e.target.closest("[data-view]"); if (b) setView(b.dataset.view); });
  $("metrics").addEventListener("click", (e) => {
    const b = e.target.closest("[data-metric]");
    if (!b || b.disabled) return;
    state.metric = b.dataset.metric;
    document.querySelectorAll("#metrics button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderAll();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== $("search")) { e.preventDefault(); $("search").focus(); }
    if (e.key === "Escape") { if (document.activeElement === $("search")) $("search").blur(); else select(null); }
  });
  window.addEventListener("hashchange", () => {
    const view = location.hash.slice(1);
    if (VIEWS.includes(view) && view !== state.view) setView(view);
  });
  function renderAll() {
    renderTree();
    renderFlow();
    renderTable();
    if (state.view === "treemap") renderTreemap();
    if (state.view === "graph") renderGraph(); else gv.dirty = true;
    renderDetails();
    markSelected();
  }
  renderAll();
  const initial = location.hash.slice(1);
  setView(VIEWS.includes(initial) ? initial : G ? "graph" : "flow");
})();
