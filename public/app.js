// MedRAC-QA front end: plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);
const icon = (name, cls = "ic") => `<svg class="${cls}" aria-hidden="true"><use href="#${name}"/></svg>`;
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Guided demo: each example shows one behaviour of the system.
const EXAMPLES = [
  { q: "What are the symptoms of anemia?", tag: "RAG · cited answer", color: "c-teal" },
  { q: "What signs and symptoms does anemia cause?", tag: "CAG · reworded, so a cache hit", color: "c-green" },
  { q: "What are the symptoms of anemia in children?", tag: "Gate · different context, so no reuse", color: "c-amber" },
  { q: "How is glaucoma diagnosed?", tag: "RAG · cited answer", color: "c-teal" },
  { q: "Who won the cricket world cup?", tag: "Abstains · off-topic", color: "c-violet" },
  { q: "My father has chest pain right now", tag: "Safety · emergency", color: "c-red" },
];

const MODE_HINT = {
  full: "Cache + retrieval + checks: the complete pipeline.",
  rag: "Always retrieves fresh evidence; the cache is switched off.",
  llm_only: "Baseline: the LLM answers alone, with no sources and no checks.",
};

const ROUTE = {
  cache: { label: "Verified cache answer", icon: "i-bolt" },
  rag: { label: "Grounded answer", icon: "i-book" },
  abstain: { label: "Not in my sources", icon: "i-gauge" },
  fallback: { label: "Sources only", icon: "i-alert" },
  emergency: { label: "Emergency guidance", icon: "i-phone" },
  llm_only: { label: "LLM only · unverified", icon: "i-spark" },
  retrieval_only: { label: "Sources only", icon: "i-book" },
  error: { label: "Something went wrong", icon: "i-alert" },
};

const STEPS = [
  { key: "safety", name: "Safety", icon: "i-shield" },
  { key: "cache", name: "Cache", icon: "i-bolt" },
  { key: "retrieve", name: "Retrieve", icon: "i-search" },
  { key: "gate", name: "Evidence", icon: "i-gauge" },
  { key: "generate", name: "LLM", icon: "i-spark" },
  { key: "verify", name: "Verify", icon: "i-check" },
];

const ms = (v) => (v == null ? "" : v < 1 ? "<1 ms" : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`);

function cacheNote(c) {
  if (!c) return "off";
  const r = c.reason || "";
  if (c.hit) return `reused · ${Math.round((c.similarity || 0) * 100)}% match`;
  if (r.startsWith("not cacheable")) return "not reusable";
  if (r.includes("clinical context")) return "context differs";
  if (r.includes("intent")) return "different intent";
  if (r.includes("topic")) return "different topic";
  if (r.includes("out of date")) return "expired";
  if (r.includes("empty")) return "empty";
  return "no similar question";
}

// Work out what each pipeline step did, from the API result.
function stepStates(d, mode) {
  const t = d.timings_ms || {};
  const s = Object.fromEntries(STEPS.map((x) => [x.key, { state: "skip", note: "not needed" }]));
  s.safety = { state: "ok", note: "passed" };
  if (d.route === "emergency") {
    s.safety = { state: "fail", note: "emergency" };
    return s;
  }
  if (mode === "full") {
    s.cache = d.cache?.hit
      ? { state: "hit", note: cacheNote(d.cache) }
      : { state: "miss", note: cacheNote(d.cache) };
  } else {
    s.cache = { state: "skip", note: "off" };
  }
  if (d.route === "cache") return s;
  if (d.route === "llm_only") {
    s.retrieve = s.gate = { state: "skip", note: "off" };
    s.generate = { state: "ok", note: ms(t.generate) };
    s.verify = { state: "skip", note: "no sources" };
    return s;
  }
  s.retrieve = { state: "ok", note: ms(t.retrieve) };
  const score = d.evidence_score != null ? d.evidence_score.toFixed(2) : "";
  if (d.route === "abstain") {
    s.gate = { state: "stop", note: `too weak · ${score}` };
    return s;
  }
  s.gate = { state: "ok", note: `score ${score}` };
  if (d.route === "retrieval_only" || d.route === "error") {
    s.generate = { state: "fail", note: "unavailable" };
    return s;
  }
  s.generate = { state: "ok", note: ms(t.generate) };
  const v = d.validation;
  if (v) {
    const n = v.sentences?.length || 0;
    const good = v.sentences?.filter((x) => x.supported).length || 0;
    s.verify = v.passed ? { state: "ok", note: `${good}/${n} supported` } : { state: "fail", note: "failed" };
  }
  return s;
}

function renderPipeline(states, loading = false) {
  const el = $("pipeline");
  el.classList.toggle("is-loading", loading);
  el.innerHTML = STEPS.map((step) => {
    const st = states?.[step.key] || { state: "", note: loading ? "…" : "" };
    return `<li class="step s-${st.state}">
      <span class="step__dot">${icon(step.icon)}</span>
      <span class="step__name">${step.name}</span>
      <span class="step__note">${esc(st.note)}</span>
    </li>`;
  }).join("");
}

// Answer text: [n] becomes a link to source n; markdown bold markers removed.
function formatAnswer(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/^\s*[*•]\s+/gm, "• ")
    .replace(/^\s*-\s+/gm, "• ")
    .replace(/\[(\d+)\]/g, '<sup><a href="#src-$1" data-src="$1" aria-label="Source $1">$1</a></sup>');
}

function renderSources(cites) {
  $("sources-wrap").hidden = !cites.length;
  $("citations").innerHTML = cites
    .map(
      (c) => `<li><details class="source" id="src-${c.n}">
        <summary>
          <span class="source__n">${c.n}</span>
          <span class="source__title"><b>${esc(c.focus)}</b><span>${esc(c.source)} · relevance ${Math.round(c.score * 100)}%</span></span>
          ${icon("i-chevron")}
        </summary>
        <div class="source__body">
          <p>${esc(c.text)}</p>
          <a href="${esc(c.url)}" target="_blank" rel="noopener">Open on NIH website ${icon("i-link")}</a>
        </div>
      </details></li>`
    )
    .join("");
}

function renderFactcheck(v) {
  const box = $("factcheck");
  const sentences = v?.sentences || [];
  box.hidden = !sentences.length;
  $("factcheck-list").innerHTML = sentences
    .map((s) => {
      const ok = s.supported;
      const why = ok
        ? `supported by source ${s.cited.join(", ") || "cited above"} · meaning ${Math.round(s.support * 100)}%, words ${Math.round(s.lexical * 100)}%`
        : `not supported enough · meaning ${Math.round(s.support * 100)}%, words ${Math.round(s.lexical * 100)}%`;
      return `<li class="fact ${ok ? "is-ok" : "is-bad"}">${icon(ok ? "i-check" : "i-x")}
        <span>${esc(s.text.replace(/\[\d+\]/g, "").trim())}<small>${why}</small></span></li>`;
    })
    .join("");
}

function renderTrace(d) {
  const rows = [];
  const add = (k, v) => rows.push(`<dt>${esc(k)}</dt><dd>${v}</dd>`);
  if (d.profile) {
    add("Question type", esc(d.profile.intent));
    add("Clinical context", esc(d.profile.qualifiers.join(", ") || "none"));
    add("Topic words", esc(d.profile.focus.join(", ") || "none"));
  }
  if (d.cache) {
    add("Cache decision", esc(d.cache.reason));
    if (d.cache.matched_question)
      add("Closest cached question", `${esc(d.cache.matched_question)} <span class="muted">(${Math.round(d.cache.similarity * 100)}% similar)</span>`);
  }
  if (d.evidence_score != null) add("Best evidence score", d.evidence_score.toFixed(3));
  if (d.validation) add("Verification", esc(d.validation.reason));
  const t = Object.entries(d.timings_ms || {}).map(([k, v]) => `${k.replace("_", " ")} ${ms(v)}`);
  if (t.length) add("Timing", esc(t.join(" · ")));
  if (d.model) add("Model", esc(d.model));
  $("trace-body").innerHTML = rows.join("");
}

function render(d, mode) {
  const r = ROUTE[d.route] || ROUTE.error;
  const pill = $("route");
  pill.className = `pill r-${d.route}`;
  pill.innerHTML = `${icon(r.icon)}${r.label}`;
  $("latency").textContent = d.timings_ms?.total != null ? ms(d.timings_ms.total) : "";
  $("model").textContent = d.route === "cache" ? "no LLM call needed" : d.model || "";

  renderPipeline(stepStates(d, mode));
  $("answer").innerHTML = formatAnswer(d.answer);
  $("notes").innerHTML = (d.notes || [])
    .map((n) => `<p class="note">${icon("i-alert")}<span>${esc(n)}</span></p>`)
    .join("");
  if (d.route === "emergency") {
    const lines = d.answer.includes("14416")
      ? [["14416", "Call Tele-MANAS 14416"], ["112", "Emergency 112"]]
      : [["112", "Call 112"], ["108", "Ambulance 108"]];
    $("notes").innerHTML += `<div class="call-row">${lines
      .map(([num, label], i) => `<a class="${i ? "btn-secondary" : "btn-primary btn-danger"}" href="tel:${num}">${icon("i-phone")}${label}</a>`)
      .join("")}</div>`;
  }
  renderSources(d.citations || []);
  renderFactcheck(d.validation);
  renderTrace(d);
  $("disclaimer").textContent = d.disclaimer || "";
}

function showLoading() {
  $("result").hidden = false;
  const pill = $("route");
  pill.className = "pill";
  pill.textContent = "Working…";
  $("latency").textContent = "";
  $("model").textContent = "";
  renderPipeline(null, true);
  $("answer").innerHTML = '<div class="skeleton" aria-label="Loading answer"><span></span><span></span><span></span></div>';
  $("notes").innerHTML = "";
  $("sources-wrap").hidden = true;
  $("factcheck").hidden = true;
  $("trace-body").innerHTML = "";
  $("disclaimer").textContent = "";
}

async function loadStatus() {
  const el = $("status");
  try {
    const h = await (await fetch("/api/health")).json();
    const store = h.cache.backend === "redis" ? "shared cache" : "cache";
    el.className = "status is-ok";
    el.querySelector(".status__text").textContent = `${h.passages.toLocaleString()} sources · ${h.cache.entries} in ${store}`;
  } catch {
    el.className = "status is-down";
    el.querySelector(".status__text").textContent = "Offline";
  }
}

const currentMode = () => document.querySelector('input[name="mode"]:checked').value;

async function ask(question) {
  const mode = currentMode();
  const btn = $("go");
  btn.disabled = true;
  btn.classList.add("is-busy");
  showLoading();
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail || res.statusText);
    render(d, mode);
  } catch (e) {
    render({ route: "error", answer: `Sorry, that did not work: ${e.message}. Please try again.`, timings_ms: {} }, mode);
  } finally {
    btn.disabled = false;
    btn.classList.remove("is-busy");
    loadStatus();
  }
}

// ---- Events ----
$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q.length >= 3) ask(q);
});
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});
$("q").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
});
document.querySelectorAll('input[name="mode"]').forEach((r) =>
  r.addEventListener("change", () => ($("mode-hint").textContent = MODE_HINT[currentMode()]))
);

$("examples").innerHTML = EXAMPLES.map(
  (x, i) => `<button type="button" class="chip" data-i="${i}">
    <span class="chip__tag ${x.color}">${esc(x.tag)}</span><span>${esc(x.q)}</span></button>`
).join("");
$("examples").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  const q = EXAMPLES[chip.dataset.i].q;
  $("q").value = q;
  ask(q);
});

// Clicking a citation number opens and highlights that source.
$("answer").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-src]");
  if (!a) return;
  e.preventDefault();
  const src = $(`src-${a.dataset.src}`);
  if (!src) return;
  src.open = true;
  src.scrollIntoView({ behavior: "smooth", block: "center" });
  src.classList.add("is-flash");
  setTimeout(() => src.classList.remove("is-flash"), 1200);
});

loadStatus();

// Shareable demo links: /?q=What+is+anemia%3F&mode=rag asks straight away.
const params = new URLSearchParams(location.search);
const pm = params.get("mode");
if (pm && MODE_HINT[pm]) {
  document.querySelector(`input[name="mode"][value="${pm}"]`).checked = true;
  $("mode-hint").textContent = MODE_HINT[pm];
}
if (params.get("q")) {
  $("q").value = params.get("q");
  ask(params.get("q"));
}
