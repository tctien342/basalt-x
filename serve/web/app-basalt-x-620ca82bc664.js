"use strict";
const $ = s => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 0) => num(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "-";
const p2 = n => String(n).padStart(2, "0");
const clock = ms => { const d = new Date(ms); return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`; };
const gib = n => num(n) ? (n / 1024 ** 3).toFixed(1) : "-";
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ESC[c]);
const drawn = new WeakMap();
const html = (el, h) => { if (drawn.get(el) !== h) { el.innerHTML = h; drawn.set(el, h); } };   // skip unchanged redraws
const text = (el, v) => { if (el.textContent !== v) el.textContent = v; };
const HIST = 60;
const kt = n => n >= 100000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);   // 41.2k, 262k
const lim = n => n >= 10000 ? kt(n) : fmt(n);   // token limit: 8,192, 41.2k, 262k
const tick = n => n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(Math.round(n));
const API = { openai: "OpenAI", anthropic: "Anthropic" };
const shortGpu = s => String(s || "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
const shortCpu = s => String(s || "").replace(/\((R|TM)\)/gi, "").replace(/\s+\d+-Core Processor$/i, "")
  .replace(/\s+CPU\s+@.*$/i, "").replace(/\s+Processor$/i, "").replace(/\s+/g, " ").trim();

let apiKey = sessionStorage.getItem("basalt.key") || "";
let M = null;          // the last /metrics reading, with .slots from /slots

async function get(path) {
  const r = await fetch(path, { cache: "no-store", headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
  if (r.status === 401) throw Object.assign(new Error("locked"), { locked: true });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/* ---- series and charts ---- */
// the last HIST samples, padded at the start with null (no line) so the newest sample sits at the right edge
const tail = a => { const v = (a || []).slice(-HIST).map(x => num(x) ? x : null); while (v.length < HIST) v.unshift(null); return v; };
const stats = a => {
  const v = a.filter(num);
  return v.length ? { min: Math.min(...v), max: Math.max(...v), avg: v.reduce((t, x) => t + x, 0) / v.length, last: v[v.length - 1] } : null;
};
const busyStats = a => stats(a.map(x => x > 0 ? x : null));     // over the samples that ran
const STEPS = [10, 20, 50, 100, 120, 150, 160, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 3000, 4000, 5000,
  8000, 10000, 16000, 20000, 32000, 50000];
const nice = v => STEPS.find(s => s >= v * 1.15) || Math.ceil(v * 1.15 / 10000) * 10000;
// polyline points in a 0..W x 0..100 box; a null sample breaks the line
function lines(a, y, W = 100) {
  const out = []; let cur = [];
  a.forEach((v, i) => {
    if (v == null) { if (cur.length) out.push(cur); cur = []; return; }
    cur.push(`${(i / (a.length - 1) * W).toFixed(2)},${y(v).toFixed(2)}`);
  });
  if (cur.length) out.push(cur);
  return out.map(c => c.length === 1 ? [c[0], c[0].replace(/^[\d.]+/, x => String(+x + .01))] : c);
}
const poly = (pts, attrs) => `<polyline points="${pts.join(" ")}" fill="none" vector-effect="non-scaling-stroke" stroke-linejoin="round" ${attrs}/>`;
const area = (pts, fill) => `<polygon points="${pts[0].split(",")[0]},100 ${pts.join(" ")} ${pts[pts.length - 1].split(",")[0]},100" fill="${fill}"/>`;
const baseLine = `<line x1="0" y1="100" x2="100" y2="100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
let gid = 0;
// a sidebar chart: filled line between lo and hi; `limit` draws a dashed line at the top
function sbChart(a, lo, hi, accent, limit) {
  const id = `sbg${++gid}`, tint = accent ? "255,122,61" : "154,164,173";
  const segs = lines(a, v => clamp(100 - (v - lo) / (hi - lo) * 100, 1, 99));
  return `<div class="sb-ch"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">`
    + `<stop offset="0" stop-color="rgba(${tint},.13)"/><stop offset="1" stop-color="rgba(${tint},.02)"/></linearGradient></defs>${baseLine}`
    + (limit ? `<line x1="0" y1="1" x2="100" y2="1" stroke="var(--basalt-500)" stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>` : "")
    + segs.map(p => area(p, `url(#${id})`) + poly(p, `stroke="${accent ? "var(--accent)" : "var(--fg-2)"}" stroke-width="1.5"`)).join("")
    + `</svg></div>`;
}
function spark(a, max) {
  const segs = lines(a.map(v => v == null ? null : v), v => 100 - Math.min(v, max) / max * 96 - 2);
  return `<div class="spark"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${baseLine}`
    + segs.map(p => area(p, "rgba(154,164,173,.08)") + poly(p, `stroke="var(--fg-2)" stroke-width="1.4"`)).join("") + `</svg></div>`;
}
const gy = t => t ? 100 - t * 92 - 4 : 4;             // gridline height for a fraction of the axis
const grid = fr => fr.map(t => `<line x1="0" x2="100" y1="${gy(t)}" y2="${gy(t)}" stroke="var(--line-soft)" vector-effect="non-scaling-stroke"/>`).join("");
// small tick labels at the right edge, at the top and middle gridlines, skipped where an end label sits
const ticks = (max, labels) => [[4, max], [50, max / 2]].filter(t => !labels.some(l => Math.abs(l - t[0]) < 12))
  .map(t => `<span class="tk" style="top:${t[0]}%">${tick(t[1])}</span>`).join("");

/* ---- top bar ---- */
function setState(s, label) { $("#state").dataset.s = s; text($("#state"), label); }

/* ---- session figures ---- */
const big = n => !num(n) ? "-" : n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : kt(n);
const dur = ms => !num(ms) || ms <= 0 ? "-" : ms < 60000 ? `${Math.round(ms / 1000)} s` : ms < 3600000 ? `${Math.round(ms / 60000)} min` : `${(ms / 3600000).toFixed(1)} h`;
// the state word, annotated by the phase: exactly one running request (with several, one request's phase would stand for all)
function stateWord(live, name) {
  const p = live.phase;
  if (live.state !== "generating" || (live.slots || {}).active !== 1 || !p) return [name, ""];
  if (p === "thinking") return ["Thinking", ""];
  if (p === "answering") return ["Answering", ""];
  if (p.startsWith("writing a tool call") || p === "tool call complete") return ["Tool call", p.replace(/^writing a tool call: /, "Tool call: ")];
  return [name, ""];
}
// one chip in the Speed footer, only when something is cached; the rest is its tooltip
function cacheChip(conv, live) {
  if (["unloaded", "asleep", "loading"].includes(live.state) || !conv || !num(conv.parked) || conv.parked < 1) return "";
  const tip = [`${conv.parked} conversations cached, ${gib(conv.bytes)} GiB`, num(conv.evictions) ? `${conv.evictions} evicted to make room` : "",
    num(conv.restored) ? `last restore: ${fmt(conv.restored)} tokens${num(conv.restored_ms) ? ` in ${Math.round(conv.restored_ms)} ms` : ""}` : ""].filter(Boolean).join("\n");
  return `<span class="cc" title="${esc(tip)}">${conv.parked} chat${conv.parked === 1 ? "" : "s"} cached · ${gib(conv.bytes)} GiB</span>`;
}
function drawTotals(t) {
  const n = t && num(t.requests) && t.requests > 0 ? t : null;
  text($("#totals"), n ? `${fmt(n.requests)} requests served · ${fmt(n.written)} tokens written` : "");
  $("#totals").hidden = !n;
  $("#totals").title = n ? `Since the server started: ${fmt(n.prompt)} prompt tokens (${fmt(n.reused)} reused), ${dur(n.prompt_ms)} reading, ${dur(n.decode_ms)} generating` : "";
}

/* ---- sidebar ---- */
const sbRow = (k, v, u, st) => `<div class="sb-row"><span class="k">${k}</span>${st ? `<span class="st">${st}</span>` : ""}<span class="num">${v}${u ? `<span class="u">${u}</span>` : ""}</span></div>`;
const bar = f => `<div class="bar4"><i style="width:${(clamp(f || 0, 0, 1) * 100).toFixed(1)}%"></i></div>`;
const r0 = v => num(v) ? String(Math.round(v)) : "-";
function sbLoad(h, now, label = "Load") {
  const s = stats(h);
  return `<div class="sb-met grow">${sbRow(label, num(now) ? `${Math.round(now)}%` : "-", "", s ? `Min ${Math.round(s.min)} · Avg ${Math.round(s.avg)}%` : "")}${sbChart(h, 0, 100, true)}</div>`;
}
function sbPower(h, now, lim) {
  const s = stats(h), top = num(lim) && lim > 0 ? lim : Math.max(1, s ? s.max : 1);
  return `<div class="sb-met grow">${sbRow("Power", `${r0(now)}${num(lim) ? ` / ${Math.round(lim)}` : ""}`, "W", s ? `Avg ${Math.round(s.avg)} W` : "")}${sbChart(h, 0, top, false, num(lim))}</div>`;
}
function sbTemp(h, now) {
  const s = stats(h), lo = Math.min(40, s ? Math.floor(s.min) - 5 : 40), hi = Math.max(85, s ? Math.ceil(s.max) + 5 : 85);
  return `<div class="sb-met grow">${sbRow("Temp", r0(now), "°C", s ? `Min ${Math.round(s.min)} · Max ${Math.round(s.max)} °C` : "")}${sbChart(h, lo, hi, false)}</div>`;
}
const link = g => num(g.pcie_gen_max ?? g.pcie_gen) ? `Gen${g.pcie_gen_max ?? g.pcie_gen}${num(g.pcie_width) ? ` x${g.pcie_width}` : ""}` : "";
const vramRow = g => `<div class="sb-met">${sbRow("VRAM", num(g.mem_used) ? `${gib(g.mem_used)} / ${gib(g.mem_total)}` : "-", "GiB")}${bar(num(g.mem_total) ? g.mem_used / g.mem_total : 0)}</div>`;
function sbGpu(g, h, name, alone) {
  const right = alone ? `<span class="num">${esc(link(g))}</span>`
    : `<span class="num">${num(g.temp) ? `<b>${Math.round(g.temp)}</b><span class="u">°C</span>` : ""}${num(g.temp) && link(g) ? " · " : ""}${esc(link(g))}</span>`;
  return `<div class="sb-gpu"><div class="sb-gt"><span>${esc(name)}</span>${right}</div>${vramRow(g)}`
    + sbLoad(tail(h.util), g.util) + sbPower(tail(h.power), g.power, g.power_limit) + (alone ? sbTemp(tail(h.temp), g.temp) : "") + `</div>`;
}
function sbTable(G, H, names) {
  let h = `<div class="sb-tbl"><div class="h l">Load</div><div class="h">VRAM GiB</div><div class="h">Power W</div><div class="h">°C</div>`;
  G.forEach(g => {
    h += `<div class="t"><span>${esc(names(g))}</span><span class="num">${esc(link(g))}</span></div>`
      + `<div class="c l">${num(g.util) ? `${Math.round(g.util)}%` : "-"}</div>`
      + `<div class="c"><span>${gib(g.mem_used)}<span class="u">/${gib(g.mem_total)}</span></span></div>`
      + `<div class="c"><span>${r0(g.power)}<span class="u">/${r0(g.power_limit)}</span></span></div>`
      + `<div class="c">${r0(g.temp)}</div>`;
  });
  // the mean load and total power of all cards, sample by sample
  const series = k => tail(H[0] ? H[0][k] : []).map((_, i) => {
    const v = H.map(x => tail(x[k])[i]).filter(num);
    return v.length ? v.reduce((t, x) => t + x, 0) / (k === "util" ? v.length : 1) : null;
  });
  const sum = k => G.reduce((t, g) => t + (num(g[k]) ? g[k] : 0), 0);
  const load = G.filter(g => num(g.util)), mean = load.length ? load.reduce((t, g) => t + g.util, 0) / load.length : null;
  return h + `</div><div class="sb-agg"><div class="sb-gt"><span>All ${G.length} GPUs</span></div>`
    + sbLoad(series("util"), mean, "Mean load") + sbPower(series("power"), sum("power"), sum("power_limit") || null) + `</div>`;
}
function drawSide(hw, st, hist) {
  gid = 0;
  const G = hw.gpus || [], H = Object.fromEntries((hist.gpus || []).map(x => [x.index, x]));
  const named = Object.fromEntries((st.gpus || []).map(g => [g.index, shortGpu(g.name)]));
  const title = g => `GPU ${g.index} · ${named[g.index] || "GPU"}`;
  html($("#gpus"), !G.length ? `<div class="sb-none">No GPU readings.</div>`
    : G.length >= 3 ? sbTable(G, G.map(g => H[g.index] || {}), title)
    : G.map(g => sbGpu(g, H[g.index] || {}, G.length === 1 ? named[g.index] || title(g) : title(g), G.length === 1)).join(""));

  const cpu = shortCpu(st.cpu_name);
  text($("#cpu-name"), cpu); $("#cpu-name").title = st.cpu_name || "";
  html($("#sys"), sbRow("CPU", num(hw.cpu) ? `${Math.round(hw.cpu)}%` : "-")
    + `<div class="sb-met">${sbRow("RAM", num(hw.ram_total) ? `${gib(hw.ram_used)} / ${gib(hw.ram_total)}` : "-", "GiB")}${bar(num(hw.ram_total) ? hw.ram_used / hw.ram_total : 0)}</div>`
    + sbRow("Disk read", num(hw.disk_read_mb) ? hw.disk_read_mb.toFixed(1) : "-", "MB/s")
    + (st.driver ? `<div class="sb-foot">Driver ${esc(st.driver)}${st.cuda ? ` · CUDA ${esc(st.cuda)}` : ""}</div>` : ""));

}

function drawExperts(eng, r) {
  // the tier split of the last request's routed lookups; the peer tier shows only when the engine reports it, the SSD
  // tier and the warning only while experts are read from the SSD
  // every expert in VRAM: no lookup ledger exists to read, and the split is all VRAM
  const resident = !eng.spill && num(eng.expert_slots) && num(eng.expert_total) && eng.expert_total > 0 && eng.expert_slots >= eng.expert_total;
  const peer = r && num(r.pcie_share) ? r.pcie_share : null, hit = resident ? 1 : r && num(r.hit_rate) ? r.hit_rate : null;
  const spill = eng.spill, ssd = spill ? (r && num(r.ssd_share) ? r.ssd_share : hit == null ? null : 0) : 0;
  const tiers = [["VRAM", "var(--accent)", hit == null ? null : hit * (1 - (peer || 0))]];
  if (peer != null) tiers.push(["Peer", "#75b7d8", peer]);
  tiers.push(["CPU", "#81979d", hit == null ? null : Math.max(0, (1 - hit) * (1 - (peer || 0)) - (ssd || 0))]);
  if (spill) tiers.push(["SSD", "var(--warn)", ssd]);
  $(".expert-hud").classList.toggle("spill", !!spill);
  text($("#exp-note"), spill ? "SSD in use" : resident ? "All in VRAM" : hit == null ? "Awaiting data" : "Last request");
  const gb = spill ? `${(spill.bytes / 1024 ** 3).toFixed(1)} GiB` : "";
  html($("#experts"), `<div class="expert-stack" aria-hidden="true">${tiers.map(t => `<i style="width:${((t[2] || 0) * 100).toFixed(1)}%;background:${t[1]}"></i>`).join("")}</div>`
    + `<div class="expert-tiers">${tiers.map(t => `<div style="--c:${t[1]}"><span>${t[0]}</span><b>${t[2] == null ? "-" : `${(t[2] * 100).toFixed(1)}%`}</b></div>`).join("")}</div>`
    + (spill ? `<div class="expert-note"><span><b>${spill.experts} experts on SSD (${gb})</b> ·</span> <span>free ${gb} RAM</span></div>` : ""));
}

/* ---- Speed ---- */
const SC = ["#79F2D6", "#FFB66C", "#A0CEFF", "#D69FF3", "#E3CB76", "#82BFCC", "#ABD481", "#FF94A2"], SD = ["", "7 3", "1.5 3", "8 3 1.5 3"];
const off = new Set();       // decode lines switched off: "t" (total) and slot numbers
let offFor = 0;              // the slot count `off` was set up for
let chip = "";
function drawHero(live, hist, reqs, best, slots, total) {
  const multi = total > 1, generating = live.state === "generating", reading = live.state === "reading";
  // the live rate while generating, else the newest finished request's that has one
  const r = reqs[0], rd = reqs.find(x => num(x.decode_tok_s)), last = rd ? rd.decode_tok_s : null;
  const liveDec = generating && num(live.tok_s), dec = liveDec ? live.tok_s : last;
  text($("#dec-k"), liveDec ? (multi ? "Decode · All slots" : "Decode") : last != null ? "Decode · Last request" : "Decode");
  html($("#dec-v"), num(dec) ? `${dec.toFixed(1)}<small>tok/s</small>` : "-");
  $("#dec-v").classList.toggle("idle", !liveDec);

  if (offFor !== total) { offFor = total; off.clear(); if (total >= 4) off.add("t"); }
  const ss = multi ? Array.from({ length: total }, (_, k) => tail((hist.slot_tok_s || [])[k])) : [tail(hist.tok_s)];
  const agg = tail(hist.tok_s);
  $("#chips").hidden = !multi;
  if (multi) {
    const filtersLeft = $("#chips .cks")?.scrollLeft || 0;
    const items = [["t", "Total", "var(--fg)", ""]].concat(ss.map((_, k) => [String(k), `s${k + 1}`, SC[k % 8], SD[k % 4]]));
    html($("#chips"), `<div class="cks" role="group" aria-label="Decode lines">${items.map(x => `<button type="button" class="ck" data-k="${x[0]}" aria-pressed="${!off.has(x[0])}" style="color:${x[2]}"><i class="bx"></i>`
      + `<svg viewBox="0 0 12 6"><line x1="0" x2="12" y1="3" y2="3" stroke="${x[2]}" stroke-width="1.8" ${x[3] ? `stroke-dasharray="${x[3]}"` : ""}/></svg>${x[1]}</button>`).join("")}</div>`);
    $("#chips .cks").scrollLeft = filtersLeft;
  }
  const showT = multi && !off.has("t"), vis = ss.map((_, k) => !multi || !off.has(String(k)));
  const peak = a => Math.max(0, ...a.filter(num));
  const max = nice(Math.max(showT ? peak(agg) : 0, ...ss.map((a, k) => vis[k] ? peak(a) : 0), 40));
  const y = v => 100 - 4 - Math.min(v, max) / max * 92;
  let svg = grid([0, .25, .5, .75]);
  const labels = [];
  ss.forEach((a, k) => {
    if (!vis[k]) return;
    lines(a, y).forEach(p => { svg += poly(p, `stroke="${SC[k % 8]}" stroke-width="${multi ? 1.3 : 1.7}" ${multi && SD[k % 4] ? `stroke-dasharray="${SD[k % 4]}"` : ""}`); });
  });
  const anyS = vis.some(Boolean) && multi;
  if (showT) lines(agg, y).forEach(p => { svg += poly(p, `stroke="${anyS ? "var(--fg)" : "var(--accent)"}" stroke-width="1.8"`); });
  let lab = "";
  const end = a => { const s = stats(a); return s ? y(s.last) : null; };
  if (showT && end(agg) != null) { labels.push(end(agg)); lab += `<span class="cl" style="top:${end(agg).toFixed(1)}%;color:${anyS ? "var(--fg)" : "var(--accent)"}">All</span>`; }
  if (!multi && end(ss[0]) != null) { labels.push(end(ss[0])); lab += `<span class="cl" style="top:${end(ss[0]).toFixed(1)}%;color:${SC[0]}">s1</span>`; }
  html($("#dec-chart"), `<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Decode tokens per second, last 60 seconds">${svg}</svg>${lab}${ticks(max, labels)}`);

  // the live read rate while reading, else the newest finished request's that read part of its prompt
  const readOf = x => (x.prompt_tokens || 0) - (x.reused || 0);
  const rp = reqs.find(x => x.prompt_ms > 0 && readOf(x) > 0), lastPre = rp ? readOf(rp) / (rp.prompt_ms / 1000) : null;
  const livePre = reading && num(live.prefill_tok_s_mean) && live.prefill_tok_s_mean > 0;
  const pre = livePre ? live.prefill_tok_s_mean : lastPre;
  text($("#pre-k"), livePre ? "Prefill" : lastPre != null ? "Prefill · Last request" : "Prefill");
  html($("#pre-v"), num(pre) ? `${fmt(pre)}<small>tok/s</small>` : "-");
  const pa = tail(hist.prefill_tok_s_mean).map(v => v > 0 ? v : v === 0 ? 0 : null);
  const pmax = nice(Math.max(1000, ...pa.filter(num)));
  html($("#pre-chart"), `<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Prefill tokens per second, last 60 seconds">${grid([0, .5])}`
    + lines(pa, v => 100 - 4 - Math.min(v, pmax) / pmax * 92).map(p => poly(p, `stroke="var(--fg)" stroke-width="1.6"`)).join("") + `</svg>${ticks(pmax, [])}`);
  html($("#pre-last"), r && num(r.prompt_tokens) ? `<b>${fmt(r.prompt_tokens)}</b> tok${num(r.prompt_ms) ? ` in <b>${(r.prompt_ms / 1000).toFixed(2)}</b> s` : ""}` : "-");
  text($("#pre-hit"), r && r.prompt_tokens > 0 && num(r.reused) ? `${Math.round(r.reused / r.prompt_tokens * 100)}%` : "-");

  const bd = best.decode, bp = best.prefill;
  html($("#foot-dec"), `<span>Best <b>${bd ? fmt(bd.tok_s, 1) : "-"}</b> tok/s</span>${chip}`);
  html($("#foot-pre"), `Best <b>${bp ? fmt(bp.tok_s) : "-"}</b> tok/s`);
  $("#foot-dec").title = bd ? `Fastest whole request since the server started: ${bd.id}, ${fmt(bd.output_tokens)} tokens, ${clock(bd.time * 1000)}` : "";
  $("#foot-pre").title = bp ? `Fastest prompt read since the server started: ${bp.id}, ${fmt(bp.read_tokens)} tokens read, ${clock(bp.time * 1000)}` : "";
}
$("#chips").addEventListener("click", e => {
  const b = e.target.closest(".ck"); if (!b) return;
  off.has(b.dataset.k) ? off.delete(b.dataset.k) : off.add(b.dataset.k);
  draw();
});

/* ---- main card: Slots (Context for one slot) | Requests | Queue ---- */
let tab = sessionStorage.getItem("basalt.tab") || "slots";
$("#tabs").addEventListener("click", e => {
  const b = e.target.closest("button[data-t]"); if (!b) return;
  const pane = $("#pane");
  if (tab !== b.dataset.t) pane.scrollTo(0, 0);
  tab = b.dataset.t; try { sessionStorage.setItem("basalt.tab", tab); } catch (err) { /* private mode */ }
  draw();
});
const accept = s => s.drafts_offered > 20 ? Math.round(s.drafts_accepted / s.drafts_offered * 100) : null;
const prefixHit = s => num(s.reused) && s.prompt_tokens > 0 ? Math.round(s.reused / s.prompt_tokens * 100) : null;
const ACC_NOTE = "Accept: drafts accepted over offered, from the slot's last finished request";
const identityCell = r => `<span class="identity"><span class="identity-model" title="${esc(r.model || "Model unavailable")}">${esc(r.model || "—")}</span><span class="effort" title="${r.reasoning_effort == null ? "Thinking unavailable" : "Thinking effort"}">${esc(r.reasoning_effort ?? "—")}</span></span>`;
const slotPhase = s => s.state === "prefill" ? "Prefill" : s.is_processing ? "Decode" : "Idle";
function slotRows(rows, series) {
  if (!rows.length) return '<div class="empty">Waiting for slot telemetry.</div>';
  const max = nice(Math.max(20, ...rows.map(s => Math.max(0, ...(series[s.id] || []).filter(num)))));
  return `<div class="slot-head slot-grid"><span>Slot / Model / Thinking</span><span>Rate</span><span class="slot-trend">Last 60 s</span><span>Output</span><span>Context</span><span class="slot-accept" title="${ACC_NOTE}">Accept</span></div>`
    + rows.map(s => {
      const phase = slotPhase(s), a = series[s.id] || [], pre = phase === "Prefill";
      const rate = !s.is_processing ? "—" : pre ? (num(s.prefill_done) && s.prompt_tokens ? `${Math.min(100, Math.round(s.prefill_done / s.prompt_tokens * 100))}%` : "…") : fmt(s.tok_s, 1);
      const hit = prefixHit(s), acc = accept(s), recent = busyStats(a);
      return `<div class="slot-grid slot-row" data-phase="${phase.toLowerCase()}">
        <div class="slot-id"><div><b>Slot ${s.id + 1}</b><span class="slot-phase">${phase}</span></div>${s.is_processing ? identityCell(s) : ""}</div>
        <div><b>${rate}</b><small>${pre ? "Read" : s.is_processing ? "tok/s" : recent ? `Avg ${Math.round(recent.avg)}` : ""}</small></div>
        <div class="slot-trend">${spark(a, max)}</div>
        <div><b>${s.is_processing ? fmt(s.generated) : "—"}</b><small>${s.is_processing && num(s.max_tokens) ? `/ ${lim(s.max_tokens)}` : ""}</small>${num(s.ttft_s) && s.is_processing ? `<small>TTFT ${fmt(s.ttft_s, 1)} s</small>` : ""}</div>
        <div><b>${kt(s.ctx_used || 0)}</b><small>${hit != null ? `Hit ${hit}%` : `/ ${kt(s.n_ctx || 0)}`}</small>${bar(s.n_ctx ? s.ctx_used / s.n_ctx : 0)}</div>
        <div class="slot-accept"><b>${acc == null ? "—" : `${acc}%`}</b><small>${num(s.draft_len) ? `Draft ${s.draft_len.toFixed(1)}` : ""}</small></div>
      </div>`;
    }).join("");
}
function requestsPane(reqs, slots) {
  const now = Date.now() / 1000;
  const active = slots.filter(s => s.is_processing).map(s => ({
    ...s, time: s.started_at, duration_s: s.started_at ? now - s.started_at : null,
    id: s.request_id || "Running", slot: s.id, finish: slotPhase(s),
    output_tokens: s.generated, decode_tok_s: s.state === "prefill" ? null : s.tok_s, active: true
  }));
  const rows = active.concat(reqs);
  if (!rows.length) return '<div class="empty">No requests yet.</div>';
  const endpoint = r => r.path ? r.path.replace("/v1/", "").replace("chat/completions", "chat") : API[r.api] || "—";
  return `<table class="request-table"><thead><tr><th>Request</th><th>Slot</th><th>Model / Thinking</th><th>API</th><th class="r">Output</th><th class="r">Rate</th><th class="r">Duration</th></tr></thead><tbody>`
    + rows.map(r => {
      const state = r.active ? "live" : r.finish === "error" ? "err" : r.finish && r.finish !== "stop" ? "warn" : "dim";
      return `<tr${r.active ? ' class="live"' : ""}><td><span class="request-id" title="${esc(r.request_id || r.id)}">${esc(r.request_id || r.id)}</span><small class="${state}">${esc(r.finish || "—")}${num(r.time) ? ` · ${clock(r.time * 1000)}` : ""}</small></td>
        <td>${num(r.slot) ? `S${r.slot + 1}` : "—"}</td><td>${identityCell(r)}</td><td>${esc(endpoint(r))}</td>
        <td class="r">${fmt(r.output_tokens)}</td><td class="r">${fmt(r.decode_tok_s, 1)}</td>
        <td class="r">${num(r.duration_s) ? `${r.duration_s.toFixed(1)} s` : "—"}</td></tr>`;
    }).join("") + "</tbody></table>";
}
const mss = s => `${Math.floor(s / 60)}:${p2(Math.floor(s % 60))}`;
function queuePane(live) {
  const q = live.queue || [], n = live.queued || 0;
  if (!n && !q.length) return `<div class="empty">No request is waiting.</div>`;
  const toks = q.reduce((t, r) => t + (num(r.prompt_tokens) ? r.prompt_tokens : 0), 0);
  return `<div class="qsum"><span><b>${fmt(Math.max(n, q.length))}</b> prompt${Math.max(n, q.length) === 1 ? "" : "s"} waiting</span><span><b>${fmt(toks)}</b> tok to read</span></div>`
    + `<table class="queue-table"><thead><tr><th>API</th><th>Model / Thinking</th><th class="r">Prompt tok</th><th class="r">Waiting</th></tr></thead><tbody>`
    + q.map(r => `<tr><td>${esc(API[r.api] || r.api || "-")}</td><td>${identityCell(r)}</td><td class="r">${num(r.prompt_tokens) ? fmt(r.prompt_tokens) : "-"}</td><td class="r">${mss(r.waited_s || 0)}</td></tr>`).join("")
    + `</tbody></table>`;
}
/* ---- Context: one slot's window, used against free, and the tokens of each recent request ---- */
const kc = n => n >= 1048576 ? `${n / 1048576 >= 10 ? Math.round(n / 1048576) : +(n / 1048576).toFixed(1)}M` : kt(n).replace("k", "K");
const pctOf = (a, b) => b ? Math.round(a / b * 100) : 0;
// stacked columns, one per recent request, oldest first: prompt read (the uncached part) / generated, scaled to the largest
function ctxHistory(reqs) {
  const h = reqs.slice(0, 24).reverse().map(r => { return { p: (r.prompt_tokens || 0) - clamp(r.reused || 0, 0, r.prompt_tokens || 0), g: r.output_tokens || 0 }; });
  if (!h.length) return "";
  const W = 480, H = 100, step = W / h.length, bw = step * .68;
  const top = nice(Math.max(1, ...h.map(r => r.p + r.g)) / 1.15);
  let g = "";
  h.forEach((r, i) => {
    let y = H; const x = i * step + (step - bw) / 2;
    [["p", "rgba(154, 164, 173, .5)"], ["g", "var(--accent)"]].forEach(([k, col]) => {
      const hh = r[k] / top * H; y -= hh;
      g += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${hh.toFixed(1)}" fill="${col}"/>`;
    });
  });
  return `<div class="cx-hist"><div class="hh"><b>Tokens per request</b><span>${h.length === 1 ? "Last request" : `Last ${h.length} requests`}</span></div><div class="cx-chart">`
    + `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Tokens per request">`
    + `<line x1="0" x2="${W}" y1="${H}" y2="${H}" stroke="var(--line)"/><line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="var(--line-soft)" stroke-dasharray="2 4"/>${g}</svg>`
    + `<span class="tk" style="top:0">${kc(top)}</span><span class="tk" style="top:50%">${kc(top / 2)}</span><span class="tk" style="top:100%">0</span>`
    + `<div class="xl"><span>Oldest</span><span>Latest</span></div></div></div>`;
}
function contextPane(s, reqs, eng) {
  const n = s.n_ctx || eng.context || 0, used = s.ctx_used || 0;
  return `<div class="cx1"><div class="cx-head"><div class="v">${kc(used)}<small>/ ${kc(n)} · ${pctOf(used, n)}%</small></div></div>`
    + `<div class="cx-bar"><i style="width:${n ? clamp(used / n * 100, 0, 100).toFixed(2) : 0}%"></i></div>`
    + `<div class="cx-scale"><span>0</span><span>${kc(n / 2)}</span><span>${kc(n)}</span></div>`
    + ctxHistory(reqs) + `</div>`;
}
const SCHED_PHASE = { idle: "Idle", prefill: "Solo prefill", overlap: "Prefill + decode",
  solo_decode: "Solo decode", batch_decode: "Batch decode" };
const SCHED_DECISION = { none: "-", applied: "Trial applied", gap_guard: "Latency rollback", kept_incumbent: "Kept current budget",
  overlap_calibrating: "Measuring baseline", overlap_probe: "Testing chunk", overlap_confirming: "Confirming baseline",
  overlap_promoted: "Chunk applied", overlap_rolled_back: "Chunk rolled back", overlap_interrupted: "Trial interrupted" };
const LEARNING_OUTCOME = { provisional: "Provisional win", qualified: "Qualified", revalidated: "Revalidated",
  revoked: "Revoked", rejected: "Rejected" };
const LEARNING_STORE = { synced: "Synced", pending: "Write pending", error: "Write failed" };
const reading = (value, unit = "") => num(value) && value >= 0 ? `${fmt(value)}${unit}` : "-";
const memorySize = value => !num(value) || value < 0 ? "-" : value >= 1024 ** 3
  ? `${fmt(value / 1024 ** 3, 2)} GiB` : `${fmt(value / 1024 ** 2, 1)} MiB`;
function schedulerPane(m) {
  const s = m.scheduler;
  if (!s) return '<p class="empty">Scheduler telemetry is unavailable for this worker.</p>';
  const hw = m.hardware || {}, conv = m.conversation || {}, engine = m.engine || {};
  const stale = !num(s.sampled_at) || Date.now() / 1000 - s.sampled_at > 8;
  const queue = s.queue_fresh === 1 && !stale;
  const rows = items => items.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("");
  const box = (title, items) => `<section class="sched-box"><h3>${esc(title)}</h3><dl>${rows(items)}</dl></section>`;
  const learner = m.scheduler_learning;
  const last = learner?.last_outcome;
  const experts = [
    ["CUDA 0 warm lookups", reading(s.gpu0_cache_hits)],
    ["Peer resident experts", reading(s.peer_resident_experts)],
    ["Peer GPU routed entries", reading(s.peer_expert_entries)],
    ["CPU routed entries", reading(s.cpu_expert_entries)],
    ["Distinct PCIe experts", reading(s.pcie_distinct_experts)],
    ["Peer cache swaps", reading(s.peer_swaps)],
  ];
  const devices = (hw.gpus || []).map((g, i) => [
    `GPU ${g.index ?? i} device use`, `${memorySize(g.mem_used)} / ${memorySize(g.mem_total)}`,
  ]);
  const dense = Object.keys(engine).filter(k => /^dense_arena_cuda\d+_bytes$/.test(k)).sort().flatMap(key => {
    const dev = key.match(/cuda(\d+)/)[1];
    return [[`CUDA ${dev} dense arena`, memorySize(engine[key])],
      [`CUDA ${dev} native output head`, memorySize(engine[`native_head_cuda${dev}_bytes`])],
      [`CUDA ${dev} verifier workspace`, memorySize(engine[`verify_arena_cuda${dev}_bytes`])]];
  });
  const startup = new Map();
  for (const row of m.startup_memory?.kv_allocations || []) {
    const key = `${row.device} ${row.kind === "draft" ? "draft KV" : "target session and KV"}`;
    const sum = startup.get(key) || [0, 0];
    sum[0] += row.bytes;
    sum[1] += row.configured_bytes;
    startup.set(key, sum);
  }
  return `<div class="sched">
    <div class="sched-head"><strong>${esc(stale && m.live?.state === "idle" ? "Idle" : SCHED_PHASE[s.phase] || s.phase)}</strong>
    <span>${stale ? "Last sample" : "Sample"} ${num(s.sampled_at) ? clock(s.sampled_at * 1000) : "-"}</span></div>
    <div class="sched-grid">
      ${box("Active policy", [
        ["Prompt remaining", reading(s.prompt_remaining, " tok")], ["Prefill chunk", reading(s.chunk_tokens, " tok")],
        ["Decode time budget", reading(s.decode_quantum_ms, " ms")],
        ["Latest batch output gap", reading(s.batch_output_gap_ms, " ms")],
        ["Limited / joint batch windows", `${reading(s.limited_batch_windows)} / ${reading(s.batch_windows)}`],
        ["Time under a row limit", reading(s.limited_batch_window_ms, " ms")],
        ["Verify rows / limit", `${reading(s.verify_rows)} / ${reading(s.verify_limit)}`],
        ["Trial row budget", s.verify_trial_rows > 0
          ? `${reading(s.verify_trial_rows)}${s.verify_trial_remembered ? " (remembered)" : ""}` : "-"],
        ["Last decision", SCHED_DECISION[s.decision] || s.decision || "-"],
      ])}
      ${learner ? box("Overlap learning", [
        ["Qualified scopes / limit", `${reading(learner.qualified_scopes)} / ${reading(learner.scope_limit)}`],
        ["Provisional scopes", reading(learner.provisional_scopes)],
        ["Profile storage", LEARNING_STORE[learner.store_state] || "-"],
        ["Last outcome", LEARNING_OUTCOME[last?.outcome] || "-"],
        ["Last chunk", reading(last?.chunk_tokens, " tok")],
        ["Last scope", last?.scope || "-"],
        ["Last outcome time", num(last?.at) ? clock(last.at * 1000) : "-"],
      ]) : ""}
      ${box("Queue and KV", [
        ["Control waiters", queue ? reading(s.control_waiters) : "-"],
        ["Admission waiters", queue ? reading(s.admission_waiters) : "-"],
        ["Paused in admission", queue ? reading(s.parked_waiters) : "-"],
        ["Oldest waiter", queue ? reading(s.oldest_wait_ms, " ms") : "-"],
        ["Free shared KV pages", queue ? reading(s.free_kv_pages) : "-"],
        ["KV context capacity", reading(engine.max_context, " tok")],
      ])}
      ${box("Expert routing", experts)}
      ${box("VRAM allocations", [
        ...devices,
        ...dense,
        ["CUDA 0 adapter factors", memorySize(engine.adapter_cuda0_payload_bytes)],
        ["Peer adapter factors", memorySize(engine.adapter_peer_payload_bytes)],
        ["CUDA 0 expert pool, mapped", memorySize(s.gpu0_expert_mapped_bytes)],
        ["CUDA 0 expert pool, capacity", memorySize(s.gpu0_expert_capacity_bytes)],
        ["CUDA 0 prefill loan (within pool)", memorySize(s.gpu0_prefill_loan_bytes)],
        [s.cuda0_prefill_workspace_borrowed ? "CUDA 0 prefill workspace (inside loan)" : "CUDA 0 prefill workspace",
          memorySize(s.cuda0_prefill_workspace_bytes)],
        ["Peer expert pool", memorySize(s.peer_expert_cache_bytes)],
        ["Peer prefill scratch (within pool)", memorySize(s.peer_prefill_scratch_within_loan_bytes)],
        ["Peer vision loan (within pool)", memorySize(s.peer_vision_loan_bytes)],
        ["Vision device workspace and weights", memorySize(s.vision_device_bytes)],
        ["Shared indexer, mapped", memorySize(s.shared_index_mapped_bytes)],
        ["MTP tracked allocation", memorySize(s.mtp_tracked_bytes)],
      ])}
      ${startup.size ? box("KV at engine load (backed / configured)", Array.from(startup, ([k, values]) =>
        [k, `${memorySize(values[0])} / ${memorySize(values[1])}`])) : ""}
      ${box("Host memory", [
        ["System RAM use", `${memorySize(hw.ram_used)} / ${memorySize(hw.ram_total)}`],
        ["RAM available for capture", memorySize(s.capture_ram_available_bytes >= 0 ? s.capture_ram_available_bytes : null)],
        ["Expert arena and resident complement", memorySize(s.expert_arena_bytes)],
        ["Pinned KV, target + draft", memorySize(s.pinned_kv_bytes)],
        ["Pinned peer prefill weights", memorySize(s.pinned_peer_weights_bytes)],
        ["RAM prefix cache, all namespaces", memorySize(s.ram_prefix_cache_total_bytes)],
        ["RAM prefix cache, current namespace", memorySize(conv.bytes)],
        ["RAM prefix budget per namespace", memorySize(num(engine.conversation_cache_mib) ? engine.conversation_cache_mib * 1024 ** 2 : null)],
      ])}
      ${box("Snapshot storage", [
        ["RAM recovery files", memorySize(s.recovery_cache_bytes)],
        ["RAM recovery writer queue", memorySize(s.recovery_queued_bytes)],
        ["RAM recovery shared limit", memorySize(engine.recovery_budget_bytes)],
        ["NVMe cache files", memorySize(s.disk_cache_bytes)],
        ["NVMe shared budget", memorySize(engine.disk_cache_budget_bytes)],
      ])}
    </div>
  </div>`;
}
function drawMain(live, hist, reqs, slots, total) {
  const T = [["slots", "Slots"], ["req", "Requests"], ["queue", `Queue (${live.queued || 0})`]];
  if (total === 1) T.push(["ctx", "Context"]);
  if (M.engine?.scheduler_queue === 1 || M.scheduler) T.push(["scheduler", "Scheduler"]);
  const cur = T.some(t => t[0] === tab) ? tab : T[0][0];
  html($("#tabs"), T.map(t => `<button type="button" role="tab" data-t="${t[0]}" aria-selected="${t[0] === cur}">${t[1]}</button>`).join(""));
  let h;
  if (cur === "slots") {
    const series = Object.fromEntries(slots.map(s => [s.id, tail((hist.slot_tok_s || [])[s.id])]));
    h = slotRows(slots, series);
  } else if (cur === "ctx") h = contextPane(slots[0] || {}, reqs, M.engine || {});
  else if (cur === "scheduler") h = schedulerPane(M);
  else h = cur === "req" ? requestsPane(reqs, slots) : queuePane(live);
  html($("#pane"), h);
}

function drawModel(name, models, slots, active) {
  text($("#model"), name || "");
  $("#model").title = name ? `Base model · ${name}` : "";
  const root = $("#loras");
  const modes = { uncen: "Uncen", swift: "Swift" };
  const variants = ["uncen", "swift", "uncen-swift"].flatMap(mode => [mode, `${mode}-512k`]);
  const served = new Set(models.map(model => model.id));
  const running = slots.filter(slot => slot.is_processing);
  const known = num(active) ? running.length === active && running.every(slot => slot.model) : running.some(slot => slot.model);
  const badges = Object.entries(modes).filter(([mode]) =>
    variants.some(alias => alias.split("-").includes(mode) && served.has(`${name}-${alias}`)));
  root.hidden = !name || !badges.length;
  html(root, badges.map(([mode, label]) => {
    const count = running.filter(slot => variants.some(alias =>
      alias.split("-").includes(mode) && slot.model === `${name}-${alias}`)).length;
    const status = known ? (count ? `${count} active slot${count === 1 ? "" : "s"}` : "Idle") : "Usage unavailable";
    return `<span class="lora" data-active="${known && count > 0}" title="${label} LoRA · ${status}" aria-label="${label} LoRA. ${status}">${label}${known && count ? ` <b>${count}</b>` : ""}</span>`;
  }).join(""));
}

/* ---- the whole page ---- */
function draw() {
  if (!M) return;
  const live = M.live || {}, eng = M.engine || {}, hist = M.history || {}, reqs = M.requests || [];
  const total = (live.slots || {}).total || 1, slots = M.slots || [];
  const name = (live.state || "idle")[0].toUpperCase() + (live.state || "idle").slice(1);
  const ld = live.state === "loading" ? live.load || {} : null, weights = ld && ld.kind !== "wake";
  const models = M.models || [];
  const none = models.length > 0 && !models.some(m => m.status.value === "loaded" || m.status.value === "loading");
  const [sw, swTip] = stateWord(live, name);
  $("#state").title = swTip;
  setState(live.state || "idle", weights ? "Loading" : ld ? "Waking up" : live.queued > 0 ? `${name} · ${live.queued} queued` : none ? models.some(m => m.status.value === "failed") ? "Load failed" : "No model" : sw);
  const pct = weights && num(ld.percent) ? ld.percent : null;
  text($("#pct"), pct === null ? "" : `${pct}%`); $("#pct").hidden = pct === null;
  $("#pbar").hidden = !weights; $("#pbar").firstElementChild.style.width = `${pct || 0}%`;
  drawModel(eng.model, models, slots, live.slots?.active);
  chip = cacheChip(M.conversation, live);
  drawTotals(M.totals);
  drawSide(M.hardware || {}, M.hardware_static || {}, hist);
  drawExperts(eng, reqs[0]);
  drawHero(live, hist, reqs, M.best || {}, slots, total);
  drawMain(live, hist, reqs, slots, total);
  drawTuner(M.scheduler, live, slots, total);
}

/* ---- logs ---- */
// the engine's long conversation cache lines, shortened for the panel (the log and the API keep the raw line)
function shortLog(msg) {
  const k = n => n >= 1000 ? `${Math.round(n / 1000)}k` : String(n), g = b => `${(b / 1024 ** 3).toFixed(1)} GiB`;
  let m = /conversation cache: parked (\d+) tokens in [\d.]+ ms; parked=(\d+) bytes=(\d+)/.exec(msg);
  if (m) return `cache: parked ${k(+m[1])} tokens · ${m[2]} cached · ${g(+m[3])}`;
  m = /conversation cache: restored (\d+) tokens \(\w+\) in ([\d.]+) ms/.exec(msg);
  return m ? `cache: restored ${k(+m[1])} tokens in ${Math.round(+m[2])} ms` : msg;
}
const logEl = $("#log"), MAXLOG = 400;
let logCount = 0, follow = true;
function lineEl(e, arrival, i) {
  const row = document.createElement("div");
  row.className = "log-entry";
  row.dataset.level = e.lvl;
  row.title = e.msg;
  const time = document.createElement("time");
  time.textContent = clock(e.t * 1000);
  const message = document.createElement("span");
  message.textContent = (e.src === "srv" ? "[server] " : "") + shortLog(e.msg);
  row.append(time, message);
  if (i >= arrival) {
    row.classList.add("log-arrival");
    row.style.setProperty("--arrival-delay", `${(i - arrival) * 24}ms`);
  }
  return row;
}
function append(lines, animate) {
  logEl.querySelector(".empty")?.remove();
  const arrival = animate && follow && !document.hidden ? Math.max(0, lines.length - 8) : lines.length;
  logEl.append(...lines.map((line, i) => lineEl(line, arrival, i)));
  while (logEl.childElementCount > MAXLOG) logEl.firstChild.remove();
  logCount += lines.length;
  text($("#log-count"), `${fmt(logCount)} lines`);
  if (follow) logEl.scrollTop = logEl.scrollHeight;
}
logEl.addEventListener("scroll", () => { follow = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 12; });
function clearLog(note) {
  logCount = 0; follow = true; text($("#log-count"), "");
  const d = document.createElement("div"); d.className = "empty"; d.textContent = note; logEl.replaceChildren(d);
}

/* ---- polling ---- */
let lastLog = 0, logReady = false;
async function pollMetrics() {
  try {
    const m = await get("metrics");
    m.slots = await get("slots").catch(() => []);
    M = m; $("#auth").hidden = true; draw();
  } catch (e) {
    $("#auth").hidden = !e.locked; M = null;
    setState("offline", e.locked ? "Locked" : "Offline");
    drawModel(null, [], [], null);
    drawExperts({}, null);
    drawTuner(null, { state: "offline" }, [], 0);
  }
  setTimeout(pollMetrics, 1000);
}
async function pollLog() {
  try {
    const r = await get(`api/logs?after=${lastLog}`);
    if (r.last < lastLog) {
      lastLog = 0;
      logReady = false;
      clearLog("The server restarted.");
    } else {
      if (r.lines.length) { lastLog = r.last; append(r.lines, logReady); }
      else if (!logCount && !logEl.firstChild) clearLog("No log lines yet.");
      logReady = true;
    }
  } catch (e) { /* the metrics poll reports the connection */ }
  setTimeout(pollLog, 1000);
}
$("#auth").onsubmit = e => {
  e.preventDefault(); apiKey = $("#key").value.trim(); sessionStorage.setItem("basalt.key", apiKey);
};

$("#host").textContent = location.host;
fetch("health").then(r => r.json()).then(h => { $("#keyed").hidden = !h.api_key; }).catch(() => {});

/* ---- Tuner status and projected lattice ---- */
const tuner = $("#tuner"), canvas = $("#tuner-cube"), cube = canvas.getContext("2d");
const motion = $("#tuner-motion"), reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let angle = -.5, scan = 0, frame = 0, lastFrame = 0, paused = reducedMotion.matches, cubeVisible = false;
const lattice = Array.from({ length: 9 }, (_, i) => (i - 4) / 4);
const cells = Array.from({ length: 216 }, (_, i) => ({
  x: (i % 6 - 2.5) / 3, y: (Math.floor(i / 6) % 6 - 2.5) / 3, z: (Math.floor(i / 36) - 2.5) / 3,
  rank: (137 * i % 216) / 216,
}));
let expertSample = null, hitRate = null;
function sampleExpertHits(s, working) {
  const at = s?.sampled_at, hits = s?.gpu0_cache_hits;
  if (!working || !num(at) || !num(hits) || hits < 0) {
    expertSample = null;
    hitRate = null;
    return;
  }
  if (expertSample?.at === at) return;
  const dt = expertSample ? at - expertSample.at : 0;
  hitRate = dt > 0 && dt <= 8 && hits >= expertSample.hits ? (hits - expertSample.hits) / dt : null;
  expertSample = { at, hits };
}
function drawTuner(s, live, slots, total) {
  const slotCount = live.slots?.active;
  const workingSlots = num(slotCount) ? slotCount : slots.filter(slot => slot.is_processing).length;
  const work = workingSlots > 0 || live.state === "reading" || live.state === "generating" || slots.some(slot => slot.is_processing);
  const stale = !num(s?.sampled_at) || Date.now() / 1000 - s.sampled_at > 8;
  const phase = s && !work && stale ? "idle" : s?.phase;
  const pending = ["loading", "unloaded", "asleep"].includes(live.state);
  const status = live.state === "offline" ? "Disconnected" : pending ? "Standby" : !s ? "Unavailable" : work && stale ? "Awaiting sample" : SCHED_PHASE[phase] || (phase || "Idle");
  const decision = s && SCHED_DECISION[s.decision];
  tuner.dataset.mode = s && work && !stale && !pending
    ? (/rolled_back|gap_guard/.test(s.decision) ? "rollback" : phase) : "idle";
  text($("#tuner-phase"), status);
  text($("#tuner-decision"), decision && decision !== "-" ? (stale || pending ? "Last · " : "") + decision : "");
  sampleExpertHits(s, work && !stale && !pending);
  const slotLabel = total && (num(slotCount) || slots.length) ? `${workingSlots} / ${total}` : "—";
  text($("#tuner-slots"), slotLabel);
  text($("#tuner-hits"), num(hitRate) ? kt(Math.round(hitRate)) : "—");
  text($("#tuner-chunk"), num(s?.chunk_tokens) && s.chunk_tokens > 0 ? `${fmt(s.chunk_tokens)} tok` : "—");
  text($("#tuner-quantum"), num(s?.decode_quantum_ms) && s.decode_quantum_ms > 0 ? `${fmt(s.decode_quantum_ms)} ms` : "—");
  if (!frame) paintCube();
}
function paintCube() {
  if (!cube) return;
  const { clientWidth: w, clientHeight: h } = canvas;
  if (!w || !h) return;
  const dpr = Math.min(2, devicePixelRatio || 1);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  cube.setTransform(dpr, 0, 0, dpr, 0, 0);
  cube.clearRect(0, 0, w, h);
  const scale = Math.min(w * .237, h * .235), cx = w * .56, cy = h * .52;
  const c = Math.cos(angle), s = Math.sin(angle), tilt = .47, active = tuner.dataset.mode !== "idle";
  const rgb = ["prefill", "rollback"].includes(tuner.dataset.mode) ? "255,182,102" : "112,247,220";
  const point = ([x, y, z]) => {
    const u = x * c - z * s, v = x * s + z * c;
    const depth = y * Math.sin(tilt) + v * Math.cos(tilt), p = scale * 5 / (5 + depth);
    return [cx + u * p, cy + (y * Math.cos(tilt) - v * Math.sin(tilt)) * p];
  };
  const line = (vertices, alpha, width = 1) => {
    cube.beginPath();
    vertices.map(point).forEach(([x, y], i) => i ? cube.lineTo(x, y) : cube.moveTo(x, y));
    cube.strokeStyle = `rgba(${rgb},${alpha})`;
    cube.lineWidth = width;
    cube.stroke();
  };
  const square = y => [[-1,y,-1],[1,y,-1],[1,y,1],[-1,y,1],[-1,y,-1]];
  cube.save();
  cube.translate(cx, cy + scale * .59);
  cube.scale(1, .32);
  for (const r of [1.36, 1.59, 1.84]) {
    cube.beginPath();
    cube.arc(0, 0, scale * r, 0, Math.PI * 2);
    cube.strokeStyle = "rgba(105,218,205,.19)";
    cube.setLineDash(r === 1.59 ? [2, 5] : []);
    cube.stroke();
  }
  cube.restore();
  for (const y of lattice) {
    const rim = Math.abs(y) === 1;
    line(square(y), rim ? .65 : .20, rim ? 1.3 : .75);
    for (const k of lattice) {
      line([[k,y,-1], [k,y,1]], rim ? .24 : .065, .7);
      line([[-1,y,k], [1,y,k]], rim ? .24 : .065, .7);
    }
  }
  for (const x of lattice) for (const z of lattice) {
    const surface = Math.abs(x) === 1 || Math.abs(z) === 1;
    line([[x,-1,z], [x,1,z]], surface ? .23 : .05, .75);
  }
  if (!active) return;
  // Y increases downwards. Restart just above the cube after the sweep exits below it.
  const y = paused ? .18 : (scan % 1) * 2.24 - 1.12;
  const strength = clamp(Math.log1p((hitRate || 0) / 500) / Math.log1p(200), 0, 1);
  const trail = paused ? .83 : .50;
  const visible = cells.filter(cell => y - cell.y >= -.14 && y - cell.y < trail && cell.rank < strength)
    .sort((a, b) => (b.x - a.x) * s + (b.z - a.z) * c);
  const face = (vertices, alpha) => {
    cube.beginPath();
    vertices.map(point).forEach(([x, y], i) => i ? cube.lineTo(x, y) : cube.moveTo(x, y));
    cube.closePath();
    cube.fillStyle = `rgba(${rgb},${alpha})`;
    cube.fill();
  };
  for (const cell of visible) {
    const {x, y: level, z} = cell, r = .075;
    const a = Math.sign(s) * r, b = Math.sign(c) * r;
    const opacity = (.3 + .65 * strength) * (1 - Math.abs(y - level) / (trail + .14));
    const top = [[x-r,level-r,z-r], [x+r,level-r,z-r], [x+r,level-r,z+r], [x-r,level-r,z+r]];
    face([[x-a,level-r,z-r],[x-a,level-r,z+r],[x-a,level+r,z+r],[x-a,level+r,z-r]], opacity * .55);
    face([[x-r,level-r,z-b],[x+r,level-r,z-b],[x+r,level+r,z-b],[x-r,level+r,z-b]], opacity * .74);
    face(top, opacity);
  }
  if (Math.abs(y) <= 1) {
    cube.shadowColor = `rgba(${rgb},.85)`;
    cube.shadowBlur = 6 + strength * 13;
    line(square(y), .43 + strength * .50, 1.45);
    cube.shadowBlur = 0;
  }
}
function animateCube(now) {
  if (now - lastFrame >= 33) {
    const step = Math.min(80, now - lastFrame);
    lastFrame = now;
    angle += step * .00023;
    scan += step * .00048;
    paintCube();
  }
  frame = requestAnimationFrame(animateCube);
}
function syncCube() {
  motion.setAttribute("aria-pressed", String(paused));
  motion.setAttribute("aria-label", paused ? "Play cube motion" : "Pause cube motion");
  motion.title = motion.getAttribute("aria-label");
  text(motion, paused ? "▷" : "Ⅱ");
  const running = cube && cubeVisible && !document.hidden && !paused;
  if (running && !frame) frame = requestAnimationFrame(animateCube);
  if (!running && frame) { cancelAnimationFrame(frame); frame = 0; }
  if (!running) paintCube();
}
motion.onclick = () => { paused = !paused; syncCube(); };
reducedMotion.addEventListener("change", e => { paused = e.matches; syncCube(); });
document.addEventListener("visibilitychange", syncCube);
new ResizeObserver(paintCube).observe(canvas);
new IntersectionObserver(([entry]) => { cubeVisible = entry.isIntersecting; syncCube(); }).observe(tuner);
syncCube();

pollMetrics(); pollLog();
