// MedRAC-QA front end: plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  "What are the symptoms of anemia?",
  "What signs does anemia cause?",            // paraphrase -> cache hit
  "What are the symptoms of anemia in children?", // different context -> no reuse
  "How is glaucoma diagnosed?",
  "What causes high blood pressure?",
  "Who won the cricket world cup?",           // out of scope -> abstain
];

const ROUTE_LABEL = {
  cache: "CAG · validated cache",
  rag: "RAG · grounded answer",
  abstain: "Abstained · no evidence",
  fallback: "Fallback · sources only",
  emergency: "Emergency guidance",
  llm_only: "LLM only (no evidence)",
  retrieval_only: "Retrieval only",
  error: "Error",
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Show [1] as a superscript and strip markdown bold markers.
function formatAnswer(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g, "$1").replace(/\[(\d+)\]/g, "<sup>[$1]</sup>");
}

async function loadStatus() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    $("status").textContent = `${h.passages.toLocaleString()} passages · ${h.llm_model || "no LLM"} · cache ${h.cache.entries}`;
  } catch {
    $("status").textContent = "server offline";
  }
}

function renderTrace(d) {
  const rows = [];
  const add = (k, v) => rows.push(`<dt>${esc(k)}</dt><dd>${v}</dd>`);
  if (d.profile) {
    add("Intent", esc(d.profile.intent));
    add("Clinical context", esc(d.profile.qualifiers.join(", ") || "none"));
    add("Topic words", esc(d.profile.focus.join(", ")));
  }
  if (d.cache) {
    const c = d.cache;
    add("Cache", `${c.hit ? '<span class="ok">hit</span>' : "miss"} — ${esc(c.reason)}`);
    if (c.matched_question) add("Closest cached question", `${esc(c.matched_question)} (similarity ${c.similarity})`);
  }
  if (d.evidence_score != null) add("Best evidence score", d.evidence_score.toFixed(3));
  if (d.validation) {
    const v = d.validation;
    add("Validation", `${v.passed ? '<span class="ok">passed</span>' : '<span class="bad">failed</span>'} — ${esc(v.reason)}`);
    add("Unsupported sentences", `${Math.round(v.unsupported_rate * 100)}%`);
  }
  const t = d.timings_ms || {};
  add("Time (ms)", esc(Object.entries(t).map(([k, v]) => `${k} ${v}`).join(" · ")));
  $("trace-body").innerHTML = `<dl class="kv">${rows.join("")}</dl>`;
}

function render(d) {
  $("result").hidden = false;
  const badge = $("route");
  badge.className = `badge ${d.route}`;
  badge.textContent = ROUTE_LABEL[d.route] || d.route;
  const total = d.timings_ms?.total;
  $("latency").textContent = total != null ? `${total < 1000 ? total + " ms" : (total / 1000).toFixed(1) + " s"}` : "";
  $("model").textContent = d.route === "cache" ? "no LLM call" : d.model ? d.model : "";
  $("answer").innerHTML = formatAnswer(d.answer);
  $("notes").innerHTML = (d.notes || []).map((n) => `<p class="note">${esc(n)}</p>`).join("");

  const cites = d.citations || [];
  $("citations").innerHTML = cites.length
    ? `<ul class="cites">${cites
        .map(
          (c) => `<li><a href="${esc(c.url)}" target="_blank" rel="noopener">[${c.n}] ${esc(c.focus)}</a>
          <span class="muted"> · ${esc(c.source)} · score ${c.score}</span>
          <details><summary>Show passage</summary><p>${esc(c.text)}</p></details></li>`
        )
        .join("")}</ul>`
    : "";
  renderTrace(d);
  $("disclaimer").textContent = d.disclaimer || "";
}

async function ask(question) {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  $("go").disabled = true;
  $("go").textContent = "Thinking…";
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail?.[0]?.msg || d.detail || r.statusText);
    render(d);
  } catch (e) {
    render({ route: "error", answer: `Something went wrong: ${e.message}`, timings_ms: {} });
  } finally {
    $("go").disabled = false;
    $("go").textContent = "Ask";
    loadStatus();
  }
}

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q.length >= 3) ask(q);
});
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask").requestSubmit();
  }
});
$("examples").innerHTML = EXAMPLES.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
$("examples").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    $("q").value = e.target.textContent;
    ask(e.target.textContent);
  }
});
loadStatus();

// Shareable demo links: /?q=What+is+anemia%3F&mode=rag asks straight away.
const params = new URLSearchParams(location.search);
if (params.get("mode")) {
  const radio = document.querySelector(`input[name="mode"][value="${CSS.escape(params.get("mode"))}"]`);
  if (radio) radio.checked = true;
}
if (params.get("q")) {
  $("q").value = params.get("q");
  ask(params.get("q"));
}
