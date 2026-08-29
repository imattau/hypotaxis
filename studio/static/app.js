import { SimplePool } from "nostr-tools";
import { verifyEvent } from "nostr-tools/pure";

const main = document.getElementById("main");
const navLibrary = document.getElementById("nav-library");
const navNew = document.getElementById("nav-new");
const navDataset = document.getElementById("nav-dataset");
const navAdapters = document.getElementById("nav-adapters");
const navSettings = document.getElementById("nav-settings");
const nostrPool = new SimplePool();
const localAdapterVersions = new Map();

function setNav(active) {
  navLibrary.classList.toggle("active", active === "library");
  navNew.classList.toggle("active", active === "new");
  navDataset.classList.toggle("active", active === "dataset");
  navAdapters.classList.toggle("active", active === "adapters");
  navSettings.classList.toggle("active", active === "settings");
}

navLibrary.addEventListener("click", showLibrary);
navNew.addEventListener("click", showNewStory);
navDataset.addEventListener("click", showDatasetCuration);
navAdapters.addEventListener("click", showAdapters);
navSettings.addEventListener("click", showSettings);

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function el(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

// ---------- Settings ----------

async function showSettings() {
  setNav("settings");
  main.innerHTML = "<p class='empty-state'>Loading settings...</p>";
  const settings = await api("/api/settings");
  const cards = settings.models.map((model) => `
    <div class="card model-card">
      <h3>${escapeHtml(model.name)}</h3>
      <div class="meta">${escapeHtml(model.repo_id)}</div>
      <p>${escapeHtml(model.purpose)}</p>
      <span class="badge model-status">${model.downloaded ? "Downloaded" : "Not downloaded"}</span>
      <button class="btn secondary model-download" data-model-key="${escapeHtml(model.key)}" type="button" ${model.downloaded ? "disabled" : ""}>
        ${model.downloaded ? "Ready" : "Download model"}
      </button>
      <div class="status-line model-message"></div>
    </div>
  `).join("");
  main.innerHTML = `
    <section class="section-heading"><h2>Settings</h2><p>Desktop data is stored in your per-user application folder. Models are downloaded once and reused locally.</p></section>
    <div class="card path-card">
      <h3>Storage</h3>
      <div class="meta">Data: ${escapeHtml(settings.data_dir)}</div>
      <div class="meta">Output: ${escapeHtml(settings.output_dir)}</div>
      <div class="meta">Models: ${escapeHtml(settings.models_dir)}</div>
    </div>
    <h2>Models</h2><div class="grid settings-grid">${cards}</div>
  `;
  main.querySelectorAll(".model-download").forEach((button) => {
    button.addEventListener("click", () => downloadModel(button, button.dataset.modelKey));
  });
}

async function downloadModel(button, modelKey) {
  const card = button.closest(".model-card");
  const status = card.querySelector(".model-message");
  button.disabled = true;
  status.textContent = "Starting download...";
  try {
    const { job_id } = await api("/api/settings/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_key: modelKey }),
    });
    let job;
    do {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      job = await api(`/api/jobs/${job_id}`);
      status.textContent = job.message || job.status;
    } while (job.status === "queued" || job.status === "running");
    if (job.status === "done") {
      status.textContent = "Ready for use.";
      card.querySelector(".model-status").textContent = "Downloaded";
      button.textContent = "Ready";
    } else {
      throw new Error(job.message || "download failed");
    }
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
  }
}

// ---------- Library ----------

async function showLibrary() {
  setNav("library");
  main.innerHTML = "<p class='empty-state'>Loading library...</p>";
  const { stories } = await api("/api/stories");
  if (stories.length === 0) {
    main.innerHTML = "<p class='empty-state'>No stories yet. Start one from the New Story tab.</p>";
    return;
  }
  const grid = el("<div class='grid'></div>");
  for (const s of stories) {
    const card = el(`
      <div class="card">
        <button class="card-delete" title="Delete story">&times;</button>
        <h3>${escapeHtml(s.title)}</h3>
        <div class="meta">${s.page_count} pages &middot; ${s.panel_count} panels</div>
        ${s.has_output ? "<span class='badge'>Generated</span>" : ""}
      </div>
    `);
    card.addEventListener("click", () => showStoryDetail(s.id));
    card.querySelector(".card-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${s.title}"? This removes its script, generated pages, and reference images permanently.`)) {
        return;
      }
      await api(`/api/stories/${s.id}`, { method: "DELETE" });
      showLibrary();
    });
    grid.appendChild(card);
  }
  main.innerHTML = "";
  main.appendChild(grid);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map((item) => stableJson(item)).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function compareAdapterVersions(left, right) {
  const parse = (value) => String(value).split(/[.+-]/).map((part) => /^\d+$/.test(part) ? Number(part) : part);
  const a = parse(left);
  const b = parse(right);
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const av = a[index] ?? 0;
    const bv = b[index] ?? 0;
    if (av === bv) continue;
    if (typeof av === "number" && typeof bv === "number") return av - bv;
    return String(av).localeCompare(String(bv));
  }
  return 0;
}

function formatTrainingMetadata(training) {
  if (!training || typeof training !== "object") return "";
  const parts = [];
  if (training.method) parts.push(escapeHtml(training.method));
  if (training.rank) parts.push(`rank ${escapeHtml(String(training.rank))}`);
  if (training.steps) parts.push(`${escapeHtml(String(training.steps))} steps`);
  if (training.learning_rate) parts.push(`lr ${escapeHtml(String(training.learning_rate))}`);
  if (training.seed !== undefined) parts.push(`seed ${escapeHtml(String(training.seed))}`);
  if (training.dataset) parts.push(`dataset ${escapeHtml(training.dataset)}`);
  if (training.dataset_sha256) parts.push(`dataset sha256 ${escapeHtml(training.dataset_sha256.slice(0, 16))}...`);
  return parts.length ? `<div class="meta">Training: ${parts.join(" · ")}</div>` : "";
}

function formatCompositionComponents(components) {
  if (!Array.isArray(components)) return "";
  const entries = components.map((component) => {
    const identity = `${component.name || "?"}@${component.version || "?"}`;
    const digest = typeof component.manifest_sha256 === "string" ? ` · ${component.manifest_sha256.slice(0, 12)}...` : "";
    return `${escapeHtml(identity)} × ${escapeHtml(String(component.weight))}${digest}`;
  });
  return entries.length ? `<div class="meta">Components: ${entries.join(" · ")}</div>` : "";
}

function formatEvaluations(evaluations) {
  if (!Array.isArray(evaluations) || !evaluations.length) return "";
  const entries = evaluations.map((evaluation) => {
    const digest = typeof evaluation.dataset_sha256 === "string" ? ` · ${evaluation.dataset_sha256.slice(0, 12)}...` : " · unpinned dataset";
    return `${escapeHtml(evaluation.name)} on ${escapeHtml(evaluation.dataset)}: ${escapeHtml(String(evaluation.score))}${digest}`;
  });
  return `<div class="meta">Evaluations: ${entries.join(" · ")}</div>`;
}

function compositionWeightTagValue(weight) {
  const value = String(weight);
  return value.includes(".") ? value : `${value}.0`;
}

function base64Url(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function loadReportSummaries(relays, targetIds) {
  const summaries = new Map();
  if (!targetIds.length) return summaries;
  const reportEvents = (await nostrPool.querySync(relays, { kinds: [1985], "#e": targetIds, limit: 500 }))
    .filter((report) => verifyEvent(report));
  const latestReports = new Map();
  for (const report of reportEvents) {
    const label = report.tags.find((tag) => tag[0] === "l" && tag[2] === "hypotaxis.adapter.report");
    const target = report.tags.find((tag) => tag[0] === "e")?.[1];
    if (!target || !targetIds.includes(target) || !label?.[1]) continue;
    const key = `${target}:${report.pubkey}:${label[1]}`;
    const previous = latestReports.get(key);
    if (!previous || report.created_at > previous.created_at ||
        (report.created_at === previous.created_at && report.id > previous.id)) {
      latestReports.set(key, { target, reason: label[1], created_at: report.created_at, id: report.id });
    }
  }
  for (const report of latestReports.values()) {
    const summary = summaries.get(report.target) || { count: 0, reasons: new Map() };
    summary.count += 1;
    summary.reasons.set(report.reason, (summary.reasons.get(report.reason) || 0) + 1);
    summaries.set(report.target, summary);
  }
  return summaries;
}

// ---------- Dataset Curation ----------

async function showDatasetCuration() {
  setNav("dataset");
  main.innerHTML = "<p class='empty-state'>Loading candidates...</p>";
  const resp = await api("/api/dataset/candidates?limit=1");
  main.innerHTML = "";
  main.appendChild(el(`<div class="section-title">Dataset Curation</div>`));
  main.appendChild(
    el(`
    <div class="status-line">
      Reviewing teacher-generated caption candidates for the Phase 3 LoRA captioner
      (see curate_dataset.py). Accept a candidate as-is, edit it first, or reject it -
      accepted captions go to data/caption_pairs_curated.jsonl, the clean dataset the
      LoRA trainer should eventually use instead of the noisy auto-harvested one.
    </div>
  `)
  );
  const countsHolder = el(`<div class="status-line" id="dataset-counts"></div>`);
  main.appendChild(countsHolder);
  const queueHolder = el(`<div id="dataset-queue"></div>`);
  main.appendChild(queueHolder);
  renderDatasetCounts(countsHolder, resp);
  renderCandidateQueue(queueHolder, resp);
}

function showAdapters() {
  setNav("adapters");
  main.innerHTML = "";
  main.appendChild(el(`<div class="section-title">Package Adapter</div>`));
  main.appendChild(el(`<div class="status-line">Create a verified local bundle for a trained LoRA. Blossom upload and mirror APIs are available; Nostr signing remains in the browser signer.</div>`));
  const discovery = el(`
    <form class="card" id="adapter-discovery-form">
      <div class="section-title">Community Discovery</div>
      <div class="status-line">Query Nostr relays for Hypotaxis adapter release metadata. Verified Blossom downloads are available for releases that advertise mirrors.</div>
      <label>Nostr relay URLs (one per line)</label>
      <textarea id="adapter-relays" class="compact-textarea" placeholder="wss://relay.example"></textarea>
      <button class="btn secondary" id="load-nostr-relays" type="button">Load My Nostr Relays</button>
      <button class="btn secondary" id="load-blossom-servers" type="button">Load My Blossom Servers</button>
      <button class="btn secondary" type="submit">Discover Adapters</button>
      <div class="status-line" id="adapter-discovery-status"></div>
      <div id="adapter-discovery-results"></div>
    </form>
  `);
  discovery.addEventListener("submit", discoverAdapters);
  discovery.querySelector("#load-nostr-relays").addEventListener("click", loadNostrRelays);
  discovery.querySelector("#load-blossom-servers").addEventListener("click", loadBlossomServers);
  main.appendChild(discovery);
  const registry = el(`<div id="local-adapter-registry"><div class="section-title">Local Adapter Registry</div><p class="empty-state">Loading local bundles...</p></div>`);
  main.appendChild(registry);
  const compositions = el(`<div id="adapter-compositions"><div class="section-title">Adapter Compositions</div><p class="empty-state">Loading compositions...</p></div>`);
  main.appendChild(compositions);
  const form = el(`
    <form class="card" id="adapter-package-form">
      <label>Source directory</label>
      <input type="text" id="adapter-source" placeholder="models/captioner/adapter" required />
      <label>Name</label>
      <input type="text" id="adapter-name" value="community-adapter" required />
      <label>Version</label>
      <input type="text" id="adapter-version" value="1.0.0" required />
      <label>Base model</label>
      <input type="text" id="adapter-base" value="Qwen/Qwen2.5-7B-Instruct" required />
      <label>License</label>
      <input type="text" id="adapter-license" value="CC-BY-4.0" required />
      <label>Files (one relative path per line; blank uses supported files in the directory)</label>
      <textarea id="adapter-files" class="compact-textarea" placeholder="adapter_model.safetensors&#10;adapter_config.json"></textarea>
      <label>Blossom mirror URLs (one per line, optional)</label>
      <textarea id="adapter-blossom" class="compact-textarea"></textarea>
      <label>BitTorrent magnet (optional)</label>
      <input type="text" id="adapter-magnet" />
      <label>Nostr public key (optional; emits an unsigned event template)</label>
      <input type="text" id="adapter-pubkey" />
      <label>Training metadata (optional JSON object)</label>
      <textarea id="adapter-training" class="compact-textarea" placeholder='{"method":"lora","rank":16,"dataset":"curated-v1"}'></textarea>
      <label>Evaluation records (optional JSON array)</label>
      <textarea id="adapter-evaluations" class="compact-textarea" placeholder='[{"name":"heldout","dataset":"corpus-v1","score":0.82}]'></textarea>
      <button class="btn" type="submit">Package Adapter</button>
      <div class="status-line" id="adapter-package-status"></div>
    </form>
  `);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button");
    const status = form.querySelector("#adapter-package-status");
    button.disabled = true;
    status.classList.remove("error");
    status.textContent = "Packaging and verifying...";
    const lines = (id) => document.getElementById(id).value.split("\n").map((line) => line.trim()).filter(Boolean);
    const optionalJson = (id, fallback) => {
      const raw = document.getElementById(id).value.trim();
      return raw ? JSON.parse(raw) : fallback;
    };
    try {
      const training = optionalJson("adapter-training", null);
      const evaluations = optionalJson("adapter-evaluations", null);
      const result = await api("/api/adapters/package", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: document.getElementById("adapter-source").value.trim(),
          name: document.getElementById("adapter-name").value.trim(),
          version: document.getElementById("adapter-version").value.trim(),
          base_model: document.getElementById("adapter-base").value.trim(),
          license: document.getElementById("adapter-license").value.trim(),
          files: lines("adapter-files"),
          blossom: lines("adapter-blossom"),
          magnet: document.getElementById("adapter-magnet").value.trim() || null,
          nostr_pubkey: document.getElementById("adapter-pubkey").value.trim() || null,
          training,
          evaluations,
        }),
      });
      status.textContent = `Created ${result.manifest_path} (${result.manifest.files.length} file(s) verified).`;
    } catch (e) {
      status.textContent = "Error: " + e.message;
      status.classList.add("error");
    } finally {
      button.disabled = false;
    }
  });
  main.appendChild(form);
  const compositionForm = el(`
    <form class="card" id="adapter-composition-form">
      <div class="section-title">Compose Adapter Bank</div>
      <div class="status-line">Combine compatible local adapters at runtime. Enter one <code>name@version=weight</code> per line.</div>
      <label>Composition name</label>
      <input type="text" id="composition-name" value="community-bank" required />
      <label>Version</label>
      <input type="text" id="composition-version" value="1.0.0" required />
      <label>Base model</label>
      <input type="text" id="composition-base" required />
      <label>Components</label>
      <textarea id="composition-components" class="compact-textarea" placeholder="grounding@1.0.0=0.7&#10;style@1.0.0=1.0" required></textarea>
      <label>Evaluation records (optional JSON array)</label>
      <textarea id="composition-evaluations" class="compact-textarea" placeholder='[{"name":"heldout-set","dataset":"corpus-v1","dataset_sha256":"...64 lowercase hex...","score":0.82}]'></textarea>
      <label><input type="checkbox" id="composition-community-merge" /> Mark as community merge (requires evaluations with dataset_sha256)</label>
      <button class="btn secondary" type="submit">Create Composition</button>
      <div class="status-line" id="composition-status"></div>
    </form>
  `);
  compositionForm.addEventListener("submit", createComposition);
  main.appendChild(compositionForm);
  loadLocalAdapters(registry);
  loadCompositions(compositions);
}

async function loadNostrRelays() {
  const button = document.getElementById("load-nostr-relays");
  const textarea = document.getElementById("adapter-relays");
  const status = document.getElementById("adapter-discovery-status");
  const bootstrapRelays = textarea.value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!window.nostr || typeof window.nostr.getPublicKey !== "function") {
    status.textContent = "A NIP-07 browser signer is required to load your relay list.";
    status.classList.add("error");
    return;
  }
  if (!bootstrapRelays.length) {
    status.textContent = "Enter at least one bootstrap relay first.";
    status.classList.add("error");
    return;
  }
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Loading your signed relay list...";
  try {
    const pubkey = await window.nostr.getPublicKey();
    const events = await nostrPool.querySync(bootstrapRelays, { kinds: [10002], authors: [pubkey], limit: 10 });
    const latest = events.filter((event) => verifyEvent(event)).sort((a, b) => b.created_at - a.created_at)[0];
    if (!latest) throw new Error("no verified relay-list event found");
    const relays = latest.tags
      .filter((tag) => tag[0] === "r" && /^wss?:\/\//.test(tag[1]) && (!tag[2] || tag[2] === "read"))
      .map((tag) => tag[1]);
    if (!relays.length) throw new Error("relay list contains no read relays");
    textarea.value = [...new Set(relays)].join("\n");
    status.textContent = `Loaded ${new Set(relays).size} read relay(s) from your signed relay list.`;
  } catch (error) {
    status.textContent = "Could not load relay list: " + error.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

async function loadBlossomServers() {
  const button = document.getElementById("load-blossom-servers");
  const relayTextarea = document.getElementById("adapter-relays");
  const serverTextarea = document.getElementById("adapter-blossom");
  const status = document.getElementById("adapter-discovery-status");
  const bootstrapRelays = relayTextarea.value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!window.nostr || typeof window.nostr.getPublicKey !== "function") {
    status.textContent = "A NIP-07 browser signer is required to load your Blossom server list.";
    status.classList.add("error");
    return;
  }
  if (!bootstrapRelays.length) {
    status.textContent = "Enter at least one bootstrap relay first.";
    status.classList.add("error");
    return;
  }
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Loading your signed Blossom server list...";
  try {
    const pubkey = await window.nostr.getPublicKey();
    const events = await nostrPool.querySync(bootstrapRelays, { kinds: [10063], authors: [pubkey], limit: 10 });
    const latest = events.filter((event) => verifyEvent(event)).sort((a, b) => b.created_at - a.created_at)[0];
    if (!latest) throw new Error("no verified Blossom server-list event found");
    const servers = latest.tags
      .filter((tag) => tag[0] === "server" && /^https?:\/\//.test(tag[1]))
      .map((tag) => tag[1].replace(/\/$/, ""));
    if (!servers.length) throw new Error("server-list event contains no HTTP(S) servers");
    serverTextarea.value = [...new Set(servers)].join("\n");
    status.textContent = `Loaded ${new Set(servers).size} Blossom server(s) into the packaging form.`;
  } catch (error) {
    status.textContent = "Could not load Blossom server list: " + error.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

async function createComposition(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button");
  const status = form.querySelector("#composition-status");
  let evaluations;
  try {
    const rawEvaluations = form.querySelector("#composition-evaluations").value.trim();
    evaluations = rawEvaluations ? JSON.parse(rawEvaluations) : undefined;
  } catch (_) {
    status.textContent = "Error: evaluations must be valid JSON.";
    status.classList.add("error");
    return;
  }
  if (form.querySelector("#composition-community-merge").checked &&
      (!Array.isArray(evaluations) || !evaluations.length ||
       evaluations.some((evaluation) => !evaluation || typeof evaluation !== "object" ||
         typeof evaluation.dataset_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(evaluation.dataset_sha256)))) {
    status.textContent = "Error: community merges require at least one evaluation with a lowercase dataset_sha256 digest.";
    status.classList.add("error");
    return;
  }
  const components = form.querySelector("#composition-components").value.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
    const [adapter, rawWeight = "1"] = line.split("=");
    const at = adapter.lastIndexOf("@");
    if (at < 1) throw new Error(`Invalid component: ${line}`);
    return { name: adapter.slice(0, at), version: adapter.slice(at + 1), weight: Number(rawWeight) };
  });
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Validating compatibility and writing composition...";
  try {
    const result = await api("/api/adapters/composition", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: form.querySelector("#composition-name").value.trim(),
        version: form.querySelector("#composition-version").value.trim(),
        base_model: form.querySelector("#composition-base").value.trim(),
        components,
        evaluations,
        community_merge: form.querySelector("#composition-community-merge").checked,
      }),
    });
    status.textContent = `Created ${result.path}`;
    loadCompositions(document.getElementById("adapter-compositions"));
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

async function loadCompositions(holder) {
  try {
    const { compositions } = await api("/api/adapters/compositions");
    holder.innerHTML = `<div class="section-title">Adapter Compositions</div>`;
    if (!compositions.length) {
      holder.appendChild(el(`<p class="empty-state">No compositions yet.</p>`));
      return;
    }
    const grid = el(`<div class="grid"></div>`);
    for (const composition of compositions) {
      const card = el(`<div class="card"><h3>${escapeHtml(composition.name)} <span class="badge">${escapeHtml(composition.version)}</span></h3><div class="meta">${escapeHtml(composition.base_model)} · ${composition.component_count} component(s)</div>${formatCompositionComponents(composition.composition.components)}<div class="meta">${composition.community_merge ? "Community merge" : "Local composition"} · ${composition.evaluation_count ? `${composition.evaluation_count} evaluation record(s)` : "No evaluation records"}</div>${formatEvaluations(composition.composition.evaluations)}<div class="meta">${escapeHtml(composition.path)}</div><button class="btn secondary publish-composition" type="button">Publish to Nostr</button><button class="btn secondary remove-composition" type="button">Remove</button><div class="status-line composition-status"></div></div>`);
      card.querySelector(".publish-composition").addEventListener("click", () => publishComposition(composition, card));
      card.querySelector(".remove-composition").addEventListener("click", () => removeLocalComposition(composition, card));
      grid.appendChild(card);
    }
    holder.appendChild(grid);
  } catch (e) {
    holder.innerHTML = `<div class="section-title">Adapter Compositions</div><p class="status-line error">Could not load compositions: ${escapeHtml(e.message)}</p>`;
  }
}

async function removeLocalComposition(composition, card) {
  if (!confirm(`Remove composition ${composition.name} ${composition.version}?`)) return;
  const button = card.querySelector(".remove-composition");
  const status = card.querySelector(".composition-status");
  button.disabled = true;
  status.classList.remove("error");
  try {
    await api(`/api/adapters/compositions/${encodeURIComponent(composition.name)}/${encodeURIComponent(composition.version)}`, { method: "DELETE" });
    await loadCompositions(document.getElementById("adapter-compositions"));
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function publishComposition(entry, card) {
  const button = card.querySelector(".publish-composition");
  const status = card.querySelector(".composition-status");
  const relays = document.getElementById("adapter-relays").value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!window.nostr || typeof window.nostr.signEvent !== "function") {
    status.textContent = "A NIP-07 browser signer is required to publish compositions.";
    status.classList.add("error");
    return;
  }
  if (!relays.length) {
    status.textContent = "Enter or load at least one relay first.";
    status.classList.add("error");
    return;
  }
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Signing composition...";
  try {
    const composition = entry.composition;
    const componentTags = composition.components.map((component) => ["component", `${component.name}@${component.version}`, component.manifest_sha256, compositionWeightTagValue(component.weight)]);
    const signed = await window.nostr.signEvent({
      kind: 30079,
      created_at: Math.floor(Date.now() / 1000),
      tags: [["d", `composition:${composition.name}`], ["version", composition.version], ["base-model", composition.base_model], ["t", "hypotaxis-adapter-composition"], ...componentTags],
      content: stableJson(composition),
    });
    if (!verifyEvent(signed)) throw new Error("signer returned an invalid event");
    await Promise.any(nostrPool.publish(relays, signed));
    status.textContent = "Published to Nostr.";
    button.textContent = "Published";
  } catch (error) {
    status.textContent = "Publish failed: " + error.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function discoverAdapters(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button");
  const status = form.querySelector("#adapter-discovery-status");
  const results = form.querySelector("#adapter-discovery-results");
  const relays = form.querySelector("#adapter-relays").value.split("\n").map((line) => line.trim()).filter(Boolean);
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Querying relays...";
  results.innerHTML = "";
  try {
    const events = await nostrPool.querySync(relays, { kinds: [30078], limit: 20 });
    const compositionEvents = await nostrPool.querySync(relays, { kinds: [30079], limit: 20 });
    const deletionTargetsEvent = (deletion, artifact) => {
      if (deletion.pubkey !== artifact.event.pubkey) return false;
      if (deletion.tags.some((tag) => tag[0] === "e" && tag[1] === artifact.event.id)) return true;
      const dTag = artifact.event.tags.find((tag) => tag[0] === "d")?.[1];
      const address = dTag === undefined ? null : `${artifact.event.kind}:${artifact.event.pubkey}:${dTag}`;
      return address !== null && deletion.tags.some((tag) => tag[0] === "a" && tag[1] === address);
    };
    let releases = events
      .filter((release) => verifyEvent(release))
      .map((release) => {
        try {
          const manifest = JSON.parse(release.content);
          if (manifest.schema !== "hypotaxis.adapter.v1" || !Array.isArray(manifest.files) || !manifest.files.length) return null;
          return { event_id: release.id, creator_pubkey: release.pubkey, signature_verified: true, event: release, manifest };
        } catch (_) {
          return null;
        }
      })
      .filter(Boolean);
    const newestReleases = new Map();
    for (const release of releases) {
      const dTag = release.event.tags.find((tag) => tag[0] === "d")?.[1] || `adapter:${release.manifest.name}`;
      const key = `${release.creator_pubkey}:${dTag}`;
      const previous = newestReleases.get(key);
      if (!previous || release.event.created_at > previous.event.created_at ||
          (release.event.created_at === previous.event.created_at && release.event.id > previous.event.id)) {
        newestReleases.set(key, release);
      }
    }
    releases = [...newestReleases.values()];
    if (releases.length) {
      const revocationResults = await Promise.all([
        nostrPool.querySync(relays, { kinds: [5], "#e": releases.map((release) => release.event_id), limit: 100 }),
        nostrPool.querySync(relays, { kinds: [5], "#a": releases.map((release) => {
          const dTag = release.event.tags.find((tag) => tag[0] === "d")?.[1];
          return dTag === undefined ? null : `${release.event.kind}:${release.event.pubkey}:${dTag}`;
        }).filter(Boolean), limit: 100 }),
      ]);
      const deletionEvents = [...new Map(revocationResults.flat().map((deletion) => [deletion.id, deletion])).values()]
        .filter((deletion) => verifyEvent(deletion));
      releases = releases.filter((release) => !deletionEvents.some((deletion) => deletionTargetsEvent(deletion, release)));
      if (!releases.length) {
        status.textContent = "No active verified releases found; checking compositions...";
        results.innerHTML = "<p class='empty-state'>No active Hypotaxis releases found.</p>";
      }
    }
    const releaseReports = await loadReportSummaries(relays, releases.map((release) => release.event_id));
    if (releases.length) {
      const ratingEvents = (await nostrPool.querySync(relays, { kinds: [1985], "#e": releases.map((release) => release.event_id), limit: 500 }))
        .filter((rating) => verifyEvent(rating));
      const latestRatings = new Map();
      for (const rating of ratingEvents) {
        const label = rating.tags.find((tag) => tag[0] === "l" && tag[2] === "hypotaxis.adapter.rating");
        const target = rating.tags.find((tag) => tag[0] === "e")?.[1];
        const value = Number(label?.[1]?.split("/")[0]);
        if (!target || !Number.isInteger(value) || value < 1 || value > 5) continue;
        const key = `${target}:${rating.pubkey}`;
        const previous = latestRatings.get(key);
        if (!previous || rating.created_at > previous.created_at ||
            (rating.created_at === previous.created_at && rating.id > previous.id)) {
          latestRatings.set(key, { target, value, created_at: rating.created_at, id: rating.id });
        }
      }
      const aggregates = new Map();
      for (const item of latestRatings.values()) {
        const aggregate = aggregates.get(item.target) || { total: 0, count: 0 };
        aggregate.total += item.value;
        aggregate.count += 1;
        aggregates.set(item.target, aggregate);
      }
      releases = releases.map((release) => {
        const aggregate = aggregates.get(release.event_id) || { total: 0, count: 0 };
        const reports = releaseReports.get(release.event_id);
        return { ...release, rating_average: aggregate.count ? aggregate.total / aggregate.count : null, rating_count: aggregate.count, report_count: reports?.count || 0, report_reasons: reports ? [...reports.reasons.entries()] : [] };
      });
    }
    status.textContent = `${releases.length} verified release(s) found.`;
    if (releases.length === 0) results.innerHTML = "<p class='empty-state'>No Hypotaxis releases found.</p>";
    for (const release of releases) {
      const manifest = release.manifest;
      const localVersion = localAdapterVersions.get(manifest.name);
      const versionComparison = localVersion ? compareAdapterVersions(manifest.version, localVersion) : 1;
      const installable = !localVersion || versionComparison > 0;
      const installAction = !localVersion || versionComparison > 0
        ? `<button class="btn secondary install-adapter" type="button">${localVersion ? "Update" : "Install from Blossom"}</button>`
        : "";
      const card = el(`
        <div class="card">
          <h3>${escapeHtml(manifest.name)} <span class="badge">${escapeHtml(manifest.version)}</span></h3>
          <div class="meta">${escapeHtml(manifest.base_model)} &middot; ${manifest.files.length} file(s)</div>
          <div class="meta">License: ${escapeHtml(manifest.license || "unspecified")}</div>
          ${formatTrainingMetadata(manifest.training)}
          ${manifest.training?.examples !== undefined ? `<div class="meta">Training examples: ${escapeHtml(String(manifest.training.examples))}</div>` : ""}
          ${formatEvaluations(manifest.evaluations)}
          <div class="meta">Creator: ${escapeHtml(release.creator_pubkey.slice(0, 16))}...</div>
          <div class="meta">Trust: Nostr signature verified</div>
          <div class="meta">Community rating: ${release.rating_average === null ? "none yet" : `${release.rating_average.toFixed(1)}/5 (${release.rating_count})`}</div>
          ${release.report_count ? `<div class="meta">Community reports: ${release.report_count} (${release.report_reasons.map(([reason, count]) => `${escapeHtml(reason)} ${count}`).join(" · ")})</div>` : ""}
          <div class="meta">Event: ${escapeHtml(release.event_id.slice(0, 16))}...</div>
          ${Array.isArray(manifest.distribution?.blossom) && manifest.distribution.blossom.length ? installAction : ""}
          ${manifest.distribution?.torrent?.magnet && installable ? `<button class="btn secondary torrent-download" type="button">${localVersion ? "Update via BitTorrent" : "Download via BitTorrent"}</button>` : ""}
          <button class="btn secondary report-release" type="button">Report</button>
          <button class="btn secondary rate-release" type="button">Rate</button>
          <div class="status-line install-status"></div>
        </div>
      `);
      const installButton = card.querySelector(".install-adapter");
      if (installButton) installButton.addEventListener("click", () => installAdapter(release, installButton));
      const torrentButton = card.querySelector(".torrent-download");
      if (torrentButton) torrentButton.addEventListener("click", () => downloadAdapterTorrent(release, torrentButton));
      card.querySelector(".report-release").addEventListener("click", () => reportRelease(release, card.querySelector(".report-release")));
      card.querySelector(".rate-release").addEventListener("click", () => rateRelease(release, card.querySelector(".rate-release")));
      results.appendChild(card);
    }
    const compositions = new Map();
    const releasesByComponent = new Map(releases.map((release) => [`${release.manifest.name}@${release.manifest.version}`, release]));
    for (const event of compositionEvents.filter((candidate) => verifyEvent(candidate))) {
      try {
        const composition = JSON.parse(event.content);
        if (composition.schema !== "hypotaxis.adapter-composition.v1" || !Array.isArray(composition.components) || !composition.components.length) continue;
        const taggedComponents = event.tags.filter((tag) => Array.isArray(tag) && tag[0] === "component");
        if (taggedComponents.length) {
          const expectedComponents = composition.components.map((component) => ["component", `${component.name}@${component.version}`, component.manifest_sha256, compositionWeightTagValue(component.weight)]);
          if (JSON.stringify(taggedComponents) !== JSON.stringify(expectedComponents)) continue;
        }
        const dTag = event.tags.find((tag) => tag[0] === "d")?.[1] || `composition:${composition.name}`;
        const key = `${event.pubkey}:${dTag}`;
        const previous = compositions.get(key);
        if (!previous || event.created_at > previous.event.created_at ||
            (event.created_at === previous.event.created_at && event.id > previous.event.id)) {
          compositions.set(key, { event, composition });
        }
      } catch (_) {
        // Ignore malformed composition events without interrupting release discovery.
      }
    }
    if (compositions.size) {
      const compositionItems = [...compositions.values()];
      const compositionRevocationResults = await Promise.all([
        nostrPool.querySync(relays, { kinds: [5], "#e": compositionItems.map((item) => item.event.id), limit: 100 }),
        nostrPool.querySync(relays, { kinds: [5], "#a": compositionItems.map((item) => {
          const dTag = item.event.tags.find((tag) => tag[0] === "d")?.[1];
          return dTag === undefined ? null : `${item.event.kind}:${item.event.pubkey}:${dTag}`;
        }).filter(Boolean), limit: 100 }),
      ]);
      const compositionDeletionEvents = [...new Map(compositionRevocationResults.flat().map((deletion) => [deletion.id, deletion])).values()]
        .filter((deletion) => verifyEvent(deletion));
      for (const [key, item] of compositions) {
        if (compositionDeletionEvents.some((deletion) => deletionTargetsEvent(deletion, { event: item.event }))) compositions.delete(key);
      }
    }
    if (compositions.size) {
      const compositionReports = await loadReportSummaries(relays, [...compositions.values()].map((item) => item.event.id));
      const compositionRatings = new Map();
      const ratingEvents = (await nostrPool.querySync(relays, { kinds: [1985], "#e": [...compositions.values()].map((item) => item.event.id), limit: 500 })).filter((rating) => verifyEvent(rating));
      for (const rating of ratingEvents) {
        const label = rating.tags.find((tag) => tag[0] === "l" && tag[2] === "hypotaxis.adapter.rating");
        const target = rating.tags.find((tag) => tag[0] === "e")?.[1];
        const value = Number(label?.[1]?.split("/")[0]);
        if (!target || !Number.isInteger(value) || value < 1 || value > 5) continue;
        const key = `${target}:${rating.pubkey}`;
        const previous = compositionRatings.get(key);
        if (!previous || rating.created_at > previous.created_at ||
            (rating.created_at === previous.created_at && rating.id > previous.id)) {
          compositionRatings.set(key, { target, value, created_at: rating.created_at, id: rating.id });
        }
      }
      const compositionAggregates = new Map();
      for (const item of compositionRatings.values()) {
        const aggregate = compositionAggregates.get(item.target) || { total: 0, count: 0 };
        aggregate.total += item.value;
        aggregate.count += 1;
        compositionAggregates.set(item.target, aggregate);
      }
      results.appendChild(el("<div class='section-title'>Community Compositions</div>"));
      for (const { event, composition } of compositions.values()) {
        const componentReleases = composition.components.map((component) => releasesByComponent.get(`${component.name}@${component.version}`));
        const installable = componentReleases.length === composition.components.length && componentReleases.every((release) => {
          const distribution = release?.manifest.distribution || {};
          return (Array.isArray(distribution.blossom) && distribution.blossom.length) || distribution.torrent?.magnet;
        });
        const aggregate = compositionAggregates.get(event.id) || { total: 0, count: 0 };
        const ratingSummary = aggregate.count ? `${(aggregate.total / aggregate.count).toFixed(1)}/5 (${aggregate.count})` : "none yet";
        const reports = compositionReports.get(event.id);
        const reportSummary = reports ? `Community reports: ${reports.count} (${[...reports.reasons.entries()].map(([reason, count]) => `${escapeHtml(reason)} ${count}`).join(" · ")})` : "";
        const card = el(`<div class="card"><h3>${escapeHtml(composition.name)} <span class="badge">${escapeHtml(composition.version)}</span></h3><div class="meta">${escapeHtml(composition.base_model)} · ${composition.components.length} component(s)</div>${formatCompositionComponents(composition.components)}<div class="meta">${composition.community_merge ? "Community merge" : "Composition"} · ${Array.isArray(composition.evaluations) ? composition.evaluations.length : 0} evaluation record(s)</div>${formatEvaluations(composition.evaluations)}<div class="meta">Community rating: ${ratingSummary}</div>${reportSummary ? `<div class="meta">${reportSummary}</div>` : ""}<div class="meta">Creator: ${escapeHtml(event.pubkey.slice(0, 16))}... · Event: ${escapeHtml(event.id.slice(0, 16))}...</div>${installable ? `<button class="btn secondary install-composition" type="button">Install Components</button>` : ""}<button class="btn secondary report-composition" type="button">Report</button><button class="btn secondary rate-composition" type="button">Rate</button><div class="status-line composition-install-status"></div></div>`);
        if (installable) card.querySelector(".install-composition").addEventListener("click", () => installRemoteComposition(composition, event, componentReleases, card));
        card.querySelector(".report-composition").addEventListener("click", () => reportArtifact(event, 30079, card.querySelector(".report-composition")));
        card.querySelector(".rate-composition").addEventListener("click", () => rateArtifact(event, 30079, card.querySelector(".rate-composition")));
        results.appendChild(card);
      }
    }
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

async function installRemoteComposition(composition, compositionEvent, releases, card) {
  const button = card.querySelector(".install-composition");
  const status = card.querySelector(".composition-install-status");
  const licenses = [...new Set(releases.map((release) => release.manifest.license || "unspecified"))];
  if (!confirm(`Install this composition's ${releases.length} adapter component(s)?\n\nLicenses: ${licenses.join(", ")}\n\nReview the license terms before continuing.`)) return;
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Verifying and installing components...";
  try {
    const result = await api("/api/adapters/composition/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ composition, composition_event: compositionEvent, release_events: releases.map((release) => release.event), license_acknowledged: true }),
    });
    status.textContent = `Installed ${result.installed.length} component(s).`;
    button.remove();
  } catch (error) {
    status.textContent = "Install failed: " + error.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function reportRelease(release, button) {
  return reportArtifact(release.event, 30078, button);
}

async function reportArtifact(artifactEvent, artifactKind, button) {
  const rawReason = prompt("Report label (lowercase, for example: license.mismatch or malware)", "license.mismatch");
  if (!rawReason) return;
  const reason = rawReason.trim().toLowerCase();
  if (!/^[a-z0-9][a-z0-9._:-]{1,63}$/.test(reason)) {
    alert("Report labels must be 2–64 lowercase letters, numbers, dots, underscores, colons, or hyphens.");
    return;
  }
  if (!window.nostr || typeof window.nostr.signEvent !== "function") {
    alert("A NIP-07 browser signer is required to submit reports.");
    return;
  }
  const relays = document.getElementById("adapter-relays").value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!relays.length) {
    alert("Enter or load at least one Nostr relay before submitting a report.");
    return;
  }
  const details = prompt("Optional report details", "") || "";
  button.disabled = true;
  try {
    const signed = await window.nostr.signEvent({
      kind: 1985,
      created_at: Math.floor(Date.now() / 1000),
      tags: [["L", "hypotaxis.adapter.report"], ["l", reason, "hypotaxis.adapter.report"], ["e", artifactEvent.id], ["k", String(artifactKind)]],
      content: details,
    });
    if (!verifyEvent(signed)) throw new Error("signer returned an invalid event");
    await Promise.any(nostrPool.publish(relays, signed));
    button.textContent = "Reported";
  } catch (e) {
    button.textContent = "Report failed";
    button.disabled = false;
  }
}

async function rateRelease(release, button) {
  return rateArtifact(release.event, 30078, button);
}

async function rateArtifact(artifactEvent, artifactKind, button) {
  const rawRating = prompt("Rating from 1 to 5", "5");
  const rating = Number(rawRating);
  if (!Number.isInteger(rating) || rating < 1 || rating > 5) return;
  if (!window.nostr || typeof window.nostr.signEvent !== "function") {
    alert("A NIP-07 browser signer is required to submit ratings.");
    return;
  }
  const details = prompt("Optional rating details", "") || "";
  button.disabled = true;
  try {
    const signed = await window.nostr.signEvent({
      kind: 1985,
      created_at: Math.floor(Date.now() / 1000),
      tags: [["L", "hypotaxis.adapter.rating"], ["l", `${rating}/5`, "hypotaxis.adapter.rating"], ["e", artifactEvent.id], ["k", String(artifactKind)]],
      content: details,
    });
    if (!verifyEvent(signed)) throw new Error("signer returned an invalid event");
    await Promise.any(nostrPool.publish(document.getElementById("adapter-relays").value.split("\n").map((line) => line.trim()).filter(Boolean), signed));
    button.textContent = `Rated ${rating}/5`;
  } catch (e) {
    button.textContent = "Rating failed";
    button.disabled = false;
  }
}

async function downloadAdapterTorrent(release, button) {
  const card = button.closest(".card");
  const status = card.querySelector(".install-status");
  if (!confirmAdapterLicense(release.manifest)) return;
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting torrent...";
  try {
    const { job_id } = await api("/api/adapters/torrent/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ magnet: release.manifest.distribution.torrent.magnet, manifest: release.manifest, release_event: release.event, license_acknowledged: true }),
    });
    const poll = async () => {
      const job = await api(`/api/jobs/${job_id}`);
      const progress = Math.round((job.progress || 0) * 100);
      const rate = job.download_rate ? ` · ${Math.round(job.download_rate / 1024)} KiB/s` : "";
      status.textContent = `${job.message || "downloading"} ${progress}% · ${job.peers || 0} peer(s)${rate}`;
      if (job.status === "done") {
        status.textContent = `Installed at ${job.bundle_dir}`;
        button.remove();
      } else if (job.status === "error") {
        throw new Error(job.message);
      } else {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        return poll();
      }
    };
    await poll();
  } catch (e) {
    const blossomButton = card.querySelector(".install-adapter");
    if (blossomButton && Array.isArray(release.manifest.distribution?.blossom) && release.manifest.distribution.blossom.length) {
      status.textContent = "BitTorrent failed; trying verified Blossom mirrors...";
      await installAdapter(release, blossomButton, true);
    } else {
      status.textContent = "Error: " + e.message;
      status.classList.add("error");
      button.disabled = false;
    }
  }
}

function confirmAdapterLicense(manifest) {
  const license = manifest.license || "unspecified";
  return confirm(`Install ${manifest.name}@${manifest.version}?\n\nDeclared license: ${license}\n\nReview the license terms before continuing.`);
}

async function installAdapter(release, button, licenseConfirmed = false) {
  const card = button.closest(".card");
  const status = card.querySelector(".install-status");
  if (!licenseConfirmed && !confirmAdapterLicense(release.manifest)) return;
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Downloading and verifying...";
  try {
    const result = await api("/api/adapters/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ manifest: release.manifest, release_event: release.event, license_acknowledged: true }),
    });
    status.textContent = `Installed at ${result.bundle_dir}`;
    button.remove();
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function loadLocalAdapters(holder) {
  try {
    const { adapters } = await api("/api/adapters/local");
    localAdapterVersions.clear();
    for (const adapter of adapters) {
      const current = localAdapterVersions.get(adapter.name);
      if (!current || compareAdapterVersions(adapter.version, current) > 0) {
        localAdapterVersions.set(adapter.name, adapter.version);
      }
    }
    holder.innerHTML = `<div class="section-title">Local Adapter Registry</div>`;
    if (adapters.length === 0) {
      holder.appendChild(el(`<p class="empty-state">No packaged adapters yet.</p>`));
      return;
    }
    const grid = el(`<div class="grid"></div>`);
    for (const adapter of adapters) {
      grid.appendChild(el(`
      <div class="card">
        <h3>${escapeHtml(adapter.name)} <span class="badge">${escapeHtml(adapter.version)}</span></h3>
        <div class="meta">${escapeHtml(adapter.base_model)} &middot; ${adapter.file_count} file(s)</div>
        <div class="meta">manifest ${escapeHtml(adapter.manifest_sha256.slice(0, 16))}...</div>
        <div class="meta">${escapeHtml(adapter.bundle_dir)}</div>
        ${formatTrainingMetadata(adapter.manifest.training)}
        ${formatEvaluations(adapter.manifest.evaluations)}
        <div class="meta torrent-state">BitTorrent: ${adapter.torrent_exists ? `ready (${escapeHtml(adapter.torrent_path)})` : adapter.torrent_available ? "not created" : "libtorrent unavailable"}</div>
        ${adapter.torrent_available && !adapter.torrent_exists ? `<button class="btn secondary create-torrent" type="button">Create Torrent</button>` : ""}
        ${adapter.torrent_exists ? `<button class="btn secondary seed-torrent" type="button">Start Seeding</button>` : ""}
        <button class="btn secondary upload-blossom" type="button">Upload to Blossom</button>
        <button class="btn secondary publish-adapter" type="button">Publish to Nostr</button>
        <button class="btn secondary remove-adapter" type="button">Remove</button>
        <div class="status-line torrent-status"></div>
        <div class="status-line seed-status"></div>
        <div class="status-line blossom-status"></div>
        <div class="status-line publish-status"></div>
      </div>
      `));
      const torrentButton = grid.lastElementChild.querySelector(".create-torrent");
      if (torrentButton) {
        torrentButton.addEventListener("click", () => createTorrent(adapter, torrentButton));
      }
      const seedButton = grid.lastElementChild.querySelector(".seed-torrent");
      if (seedButton) seedButton.addEventListener("click", () => toggleSeeding(adapter, seedButton));
      const publishButton = grid.lastElementChild.querySelector(".publish-adapter");
      publishButton.addEventListener("click", () => publishAdapter(adapter, publishButton));
      const blossomButton = grid.lastElementChild.querySelector(".upload-blossom");
      blossomButton.addEventListener("click", () => uploadAdapterToBlossom(adapter, blossomButton));
      const removeButton = grid.lastElementChild.querySelector(".remove-adapter");
      removeButton.addEventListener("click", () => removeLocalAdapter(adapter, removeButton, holder));
    }
    holder.appendChild(grid);
  } catch (e) {
    holder.innerHTML = `<div class="section-title">Local Adapter Registry</div><p class="status-line error">Could not load local bundles: ${escapeHtml(e.message)}</p>`;
  }
}

async function removeLocalAdapter(adapter, button, holder) {
  if (!confirm(`Remove ${adapter.name} ${adapter.version}? This deletes the local bundle and torrent metadata.`)) return;
  button.disabled = true;
  const status = button.closest(".card").querySelector(".blossom-status");
  try {
    await api(`/api/adapters/${encodeURIComponent(adapter.name)}/${encodeURIComponent(adapter.version)}`, { method: "DELETE" });
    await loadLocalAdapters(holder);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function uploadAdapterToBlossom(adapter, button) {
  const card = button.closest(".card");
  const status = card.querySelector(".blossom-status");
  const serverText = prompt("Blossom server URLs (comma-separated)", "https://blossom.example");
  if (!serverText) return;
  const serverUrls = serverText.split(",").map((value) => value.trim()).filter(Boolean);
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Checking Blossom servers...";
  try {
    const health = await api("/api/adapters/blossom/health", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ server_urls: serverUrls }),
    });
    const healthy = health.servers.filter((server) => server.healthy).map((server) => server.server);
    if (!healthy.length) throw new Error("no Blossom servers are reachable");
    let authorizations = null;
    let commonAuthorization = null;
    if (window.nostr && typeof window.nostr.signEvent === "function") {
      authorizations = {};
      for (const file of adapter.manifest.files) {
        const event = await window.nostr.signEvent({
          kind: 24242,
          created_at: Math.floor(Date.now() / 1000),
          tags: [["t", "upload"], ["expiration", String(Math.floor(Date.now() / 1000) + 900)], ["x", file.sha256]],
          content: "Upload Hypotaxis adapter blob",
        });
        authorizations[file.sha256] = `Nostr ${base64Url(event)}`;
      }
    } else {
      commonAuthorization = prompt("Optional BUD-11 Authorization header (Nostr ...)", "") || null;
      status.textContent = "Uploading with the supplied authorization...";
    }
    status.textContent = `Uploading to ${healthy.length} server(s)...`;
    const result = await api("/api/adapters/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: adapter.name, version: adapter.version, server_urls: healthy, authorization: commonAuthorization, authorizations }),
    });
    const failed = result.servers._failures?.length || 0;
    status.textContent = `Uploaded to ${healthy.length - failed} server(s)${failed ? `; ${failed} failed` : ""}.`;
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

async function publishAdapter(adapter, button) {
  const card = button.closest(".card");
  const status = card.querySelector(".publish-status");
  const relayText = prompt("Nostr relay URLs (comma-separated)", "wss://relay.damus.io,wss://nos.lol");
  if (!relayText) return;
  const relays = relayText.split(",").map((value) => value.trim()).filter(Boolean);
  if (!window.nostr || typeof window.nostr.signEvent !== "function") {
    status.textContent = "Error: a NIP-07 browser signer is required.";
    status.classList.add("error");
    return;
  }
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Requesting signature...";
  try {
    const manifest = adapter.manifest;
    const unsigned = {
      kind: 30078,
      created_at: Math.floor(Date.now() / 1000),
      tags: [
        ["d", `adapter:${manifest.name}`],
        ["version", manifest.version],
        ["base-model", manifest.base_model],
        ["t", "hypotaxis-adapter"],
      ],
      content: stableJson(manifest) + "\n",
    };
    const signed = await window.nostr.signEvent(unsigned);
    if (!verifyEvent(signed)) throw new Error("signer returned an invalid event");
    status.textContent = "Publishing to relays...";
    await Promise.any(nostrPool.publish(relays, signed));
    status.textContent = `Published event ${signed.id.slice(0, 16)}...`;
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

async function createTorrent(adapter, button) {
  const card = button.closest(".card");
  const status = card.querySelector(".torrent-status");
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Creating torrent metadata...";
  try {
    const result = await api("/api/adapters/torrent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: adapter.name, version: adapter.version, trackers: [] }),
    });
    card.querySelector(".torrent-state").textContent = `BitTorrent: ready (${result.torrent_path})`;
    button.remove();
    status.textContent = "Torrent metadata created.";
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function toggleSeeding(adapter, button) {
  const card = button.closest(".card");
  const status = card.querySelector(".seed-status");
  button.disabled = true;
  try {
    const current = await api(`/api/adapters/torrent/seed/${encodeURIComponent(adapter.name)}/${encodeURIComponent(adapter.version)}`);
    if (current.seeding) {
      await api(`/api/adapters/torrent/seed/${encodeURIComponent(adapter.name)}/${encodeURIComponent(adapter.version)}/stop`, { method: "POST" });
      button.textContent = "Start Seeding";
      status.textContent = "Seeding stopped.";
      return;
    }
    const result = await api("/api/adapters/torrent/seed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: adapter.name, version: adapter.version }),
    });
    button.textContent = "Stop Seeding";
    status.textContent = `Seeding with ${result.peers || 0} peer(s). Upload: ${Math.round((result.upload_rate || 0) / 1024)} KiB/s`;
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

function renderDatasetCounts(holder, resp) {
  holder.textContent = `${resp.pending_count} pending review · ${resp.curated_count} curated so far`;
}

function renderCandidateQueue(holder, resp) {
  holder.innerHTML = "";
  if (resp.candidates.length === 0) {
    holder.appendChild(
      el(
        `<p class="empty-state">No pending candidates. Run curate_dataset.py against some story text files to generate more.</p>`
      )
    );
    return;
  }
  const candidate = resp.candidates[0];
  // mirrors CAMERA_HINTS in manga_pipeline/train_captioner.py
  const CAMERA_HINTS = [
    "extreme close-up",
    "close-up",
    "medium shot",
    "wide two-shot",
    "wide establishing shot",
    "over-the-shoulder",
    "bird's-eye view",
  ];
  const candidateCamera = CAMERA_HINTS.includes(candidate.camera) ? candidate.camera : CAMERA_HINTS[2];
  const card = el(`
    <div class="card">
      <label>Source passage${candidate.characters && candidate.characters.length ? ` (characters: ${escapeHtml(candidate.characters.join(", "))})` : ""}</label>
      <div class="desc">${escapeHtml(candidate.input)}</div>
      <label>Candidate caption</label>
      <textarea id="candidate-target">${escapeHtml(candidate.target)}</textarea>
      <label>Camera</label>
      <select id="candidate-camera">
        ${CAMERA_HINTS.map((h) => `<option value="${escapeHtml(h)}"${h === candidateCamera ? " selected" : ""}>${escapeHtml(h)}</option>`).join("")}
      </select>
      <div class="row panel-edit-actions">
        <button class="btn" id="candidate-accept">Accept</button>
        <button class="btn secondary danger" id="candidate-reject">Reject</button>
      </div>
      <div class="status-line" id="candidate-status"></div>
    </div>
  `);

  const status = card.querySelector("#candidate-status");

  card.querySelector("#candidate-accept").addEventListener("click", async () => {
    status.classList.remove("error");
    status.textContent = "Saving...";
    try {
      await api(`/api/dataset/candidates/${candidate.index}/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target: card.querySelector("#candidate-target").value,
          camera: card.querySelector("#candidate-camera").value,
        }),
      });
      showDatasetCuration();
    } catch (e) {
      status.textContent = "Error: " + e.message;
      status.classList.add("error");
    }
  });

  card.querySelector("#candidate-reject").addEventListener("click", async () => {
    status.textContent = "Rejecting...";
    try {
      await api(`/api/dataset/candidates/${candidate.index}/reject`, { method: "POST" });
      showDatasetCuration();
    } catch (e) {
      status.textContent = "Error: " + e.message;
      status.classList.add("error");
    }
  });

  holder.appendChild(card);
}

// ---------- New Story ----------

const STYLE_PRESETS = [
  {
    label: "Monochrome Manga (screentone)",
    value: "monochrome manga, screentone shading, dynamic ink linework",
  },
  {
    label: "Shonen Action",
    value: "dynamic shonen manga style, bold high-contrast linework, dense screentone, speed lines, dramatic angles",
  },
  {
    label: "Shojo Romance",
    value: "soft shojo manga style, delicate thin linework, light screentone, sparkles, large expressive eyes",
  },
  {
    label: "Seinen Gritty",
    value: "gritty seinen manga style, heavy ink linework, realistic proportions, dense crosshatching, high contrast",
  },
  {
    label: "Chibi / Comedy",
    value: "cute chibi manga style, simple rounded linework, exaggerated expressions, light screentone",
  },
  {
    label: "Custom (edit below)",
    value: null,
  },
];

function showNewStory() {
  setNav("new");
  main.innerHTML = "";
  const presetOptions = STYLE_PRESETS.map((p, i) => `<option value="${i}">${escapeHtml(p.label)}</option>`).join("");
  main.appendChild(
    el(`
    <div>
      <label>Story ID (letters/numbers/underscore, used for file names)</label>
      <input type="text" id="f-id" placeholder="rain_letter_auto" />
      <label>Title</label>
      <input type="text" id="f-title" placeholder="The Letter in the Rain" />
      <label>Art style preset</label>
      <select id="f-style-preset">${presetOptions}</select>
      <label>Style prompt (fed to the image generator - edit freely)</label>
      <input type="text" id="f-style" value="${escapeHtml(STYLE_PRESETS[0].value)}" />
      <label>Chapter file (.txt, .md, .docx) &mdash; or paste below</label>
      <input type="file" id="f-file" accept=".txt,.md,.docx" />
      <div class="status-line" id="f-file-status"></div>
      <label>Prose</label>
      <textarea id="f-prose" placeholder="Paste your story text here, or upload a chapter file above..."></textarea>
      <label>Character profiles (optional) &mdash; one "Name: description" per line</label>
      <input type="file" id="f-profiles-file" accept=".txt,.md,.docx" />
      <div class="status-line" id="f-profiles-file-status"></div>
      <textarea id="f-profiles" placeholder="Aiko: young woman, shoulder-length black hair, tan raincoat, red satchel&#10;Ren: young man, short dark hair, casual jacket" style="min-height:100px"></textarea>
      <div class="status-line">List as many characters as your cast needs, one per line. Guarantees each is recognized and uses your description instead of a guessed one.</div>
      <label>Location profiles (optional) &mdash; same format</label>
      <input type="file" id="f-location-profiles-file" accept=".txt,.md,.docx" />
      <div class="status-line" id="f-location-profiles-file-status"></div>
      <textarea id="f-location-profiles" placeholder="Mill: old wooden watermill beside a river, moss-covered walls&#10;Vault: heavy steel bank vault door, dim warehouse lighting" style="min-height:100px"></textarea>
      <div class="status-line">Locations have no automatic detection at all, so list any recurring setting you want to stay visually consistent across pages.</div>
      <label>Prop profiles (optional) &mdash; same format</label>
      <input type="file" id="f-prop-profiles-file" accept=".txt,.md,.docx" />
      <div class="status-line" id="f-prop-profiles-file-status"></div>
      <textarea id="f-prop-profiles" placeholder="Letter: a folded handwritten letter with a wax seal&#10;Key: a small tarnished brass key on a frayed ribbon" style="min-height:100px"></textarea>
      <div class="status-line">Small portable objects, not whole settings. Kept separate from locations — a prop's description is woven into the panel's prompt wherever it's mentioned, not used as an image reference like locations/characters.</div>
      <label><input type="checkbox" id="f-use-captioner" style="width:auto;display:inline-block;margin-right:6px;" />Use trained captioner instead of the bridge LLM for panel captions</label>
      <div class="status-line">Faster and lighter on VRAM, but only available once you've trained one (see README's "Curating a clean caption dataset"). A captioner trained on the camera-aware dataset predicts its own shot framing directly; an older adapter falls back to the built-in heuristic.</div>
      <button class="btn" id="f-submit">Adapt Story (Stage A)</button>
      <div class="status-line" id="f-status"></div>
    </div>
  `)
  );
  wireFileUpload("f-file", "f-file-status", "f-prose");
  wireFileUpload("f-profiles-file", "f-profiles-file-status", "f-profiles");
  wireFileUpload("f-location-profiles-file", "f-location-profiles-file-status", "f-location-profiles");
  wireFileUpload("f-prop-profiles-file", "f-prop-profiles-file-status", "f-prop-profiles");
  document.getElementById("f-style-preset").addEventListener("change", (e) => {
    const preset = STYLE_PRESETS[parseInt(e.target.value, 10)];
    if (preset.value !== null) {
      document.getElementById("f-style").value = preset.value;
    }
  });
  document.getElementById("f-submit").addEventListener("click", submitNewStory);
}

function wireFileUpload(fileInputId, statusId, targetTextareaId) {
  document.getElementById(fileInputId).addEventListener("change", (e) => handleFileUpload(e, statusId, targetTextareaId));
}

async function handleFileUpload(e, statusId, targetTextareaId) {
  const file = e.target.files[0];
  if (!file) return;
  const status = document.getElementById(statusId);
  const target = document.getElementById(targetTextareaId);
  const ext = file.name.toLowerCase().split(".").pop();

  status.classList.remove("error");

  if (ext === "txt" || ext === "md") {
    status.textContent = `Loaded ${file.name}.`;
    target.value = await file.text();
    return;
  }

  status.textContent = `Extracting text from ${file.name}...`;
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/extract-text", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    const { text } = await res.json();
    target.value = text;
    status.textContent = `Loaded ${file.name}.`;
  } catch (err) {
    status.textContent = "Error: " + err.message;
    status.classList.add("error");
  }
}

async function submitNewStory() {
  const id = document.getElementById("f-id").value.trim();
  const title = document.getElementById("f-title").value.trim();
  const style_prompt = document.getElementById("f-style").value.trim();
  const prose = document.getElementById("f-prose").value.trim();
  const character_profiles = document.getElementById("f-profiles").value.trim();
  const location_profiles = document.getElementById("f-location-profiles").value.trim();
  const prop_profiles = document.getElementById("f-prop-profiles").value.trim();
  const use_trained_captioner = document.getElementById("f-use-captioner").checked;
  const status = document.getElementById("f-status");
  const button = document.getElementById("f-submit");

  if (!id || !title || !prose) {
    status.textContent = "id, title, and prose are all required.";
    status.classList.add("error");
    return;
  }

  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting...";

  try {
    const { job_id } = await api("/api/stories/adapt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id,
        title,
        prose,
        style_prompt,
        character_profiles,
        location_profiles,
        prop_profiles,
        use_trained_captioner,
      }),
    });
    pollAdaptJob(job_id, id, status, button);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function pollAdaptJob(jobId, storyId, status, button) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    status.textContent = job.message;
    if (job.status === "done") {
      button.disabled = false;
      showStoryDetail(storyId);
      return;
    }
    if (job.status === "error") {
      status.classList.add("error");
      button.disabled = false;
      return;
    }
    setTimeout(() => pollAdaptJob(jobId, storyId, status, button), 1200);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

// ---------- Story Detail ----------

async function showStoryDetail(id) {
  setNav(null);
  main.innerHTML = "<p class='empty-state'>Loading script...</p>";

  const [story, registryResp, locationsResp, propsResp, pagesResp] = await Promise.all([
    api(`/api/stories/${id}`),
    api(`/api/stories/${id}/registry`),
    api(`/api/stories/${id}/locations`),
    api(`/api/stories/${id}/props`),
    api(`/api/stories/${id}/pages`),
  ]);

  main.innerHTML = "";
  const titleRow = el(`
    <div class="detail-header">
      <div class="section-title">${escapeHtml(story.title)}</div>
      <button class="btn secondary danger" id="delete-story">Delete Story</button>
    </div>
  `);
  titleRow.querySelector("#delete-story").addEventListener("click", async () => {
    if (!confirm(`Delete "${story.title}"? This removes its script, generated pages, and reference images permanently.`)) {
      return;
    }
    await api(`/api/stories/${id}`, { method: "DELETE" });
    showLibrary();
  });
  main.appendChild(titleRow);

  main.appendChild(renderScript(story));
  main.appendChild(el(`<div class="section-title">Character Sheets</div>`));
  main.appendChild(renderCharacters(registryResp.characters, null, id, "registry"));
  main.appendChild(el(`<div class="section-title">Location Sheets</div>`));
  main.appendChild(renderCharacters(locationsResp.locations, "No location profiles for this story.", id, "locations"));
  main.appendChild(el(`<div class="section-title">Prop Sheets</div>`));
  main.appendChild(renderCharacters(propsResp.props, "No prop profiles for this story.", id, "props"));
  main.appendChild(el(`<div class="section-title">Generate Cast</div>`));
  main.appendChild(renderCastPanel(id));
  main.appendChild(el(`<div class="section-title">Generate Pages</div>`));
  main.appendChild(renderGeneratePanel(id));
  const pagesHeader = el(`
    <div class="detail-header">
      <div class="section-title">Pages</div>
      <button class="btn secondary" id="clear-pages">Clear Pages</button>
    </div>
  `);
  const galleryHolder = el(`<div id="gallery-holder"></div>`);
  pagesHeader.querySelector("#clear-pages").addEventListener("click", async () => {
    if (!confirm("Clear all generated pages and the PDF for this story? The script and cast are kept, so you can re-run Generate Pages afterward.")) {
      return;
    }
    await api(`/api/stories/${id}/pages`, { method: "DELETE" });
    renderGallery(galleryHolder, { pages: [], pdf_url: null });
  });
  main.appendChild(pagesHeader);
  main.appendChild(galleryHolder);
  renderGallery(galleryHolder, pagesResp);
}

function renderScript(story) {
  const wrap = el("<div></div>");
  story.pages.forEach((page, pageIndex) => {
    const pageEl = el(`
      <div class="script-page">
        <div class="page-head"><span>Page ${pageIndex + 1}</span><span>${page.layout}</span></div>
      </div>
    `);
    page.panels.forEach((panel, panelIndex) => {
      pageEl.appendChild(renderPanelView(story.id, pageIndex, panelIndex, panel));
    });
    wrap.appendChild(pageEl);
  });
  return wrap;
}

function renderPanelView(storyId, pageIndex, panelIndex, panel) {
  const dialogueHtml = panel.dialogue
    .map(
      (line) =>
        `<div class="bubble ${line.kind === "narration" ? "narration" : ""}"><span class="speaker">${escapeHtml(
          line.speaker
        )}:</span>${escapeHtml(line.text)}</div>`
    )
    .join("");
  const chips = panel.characters.map((c) => `<span class="chip">${escapeHtml(c)}</span>`).join("");
  const locationChips = (panel.locations || [])
    .map((l) => `<span class="chip location-chip">${escapeHtml(l)}</span>`)
    .join("");
  const propChips = (panel.props || [])
    .map((p) => `<span class="chip prop-chip">${escapeHtml(p)}</span>`)
    .join("");
  const view = el(`
    <div class="panel-card">
      <div class="panel-num">Panel ${panelIndex + 1}<br/><span class="camera">${escapeHtml(
    panel.camera_hint
  )}</span></div>
      <div>
        <div class="desc">${escapeHtml(panel.scene_description)}</div>
        <div>${chips}${locationChips}${propChips}</div>
        <div>${dialogueHtml}</div>
        <button class="btn secondary panel-edit-btn" type="button">Edit</button>
      </div>
    </div>
  `);
  view.querySelector(".panel-edit-btn").addEventListener("click", () => {
    view.replaceWith(renderPanelEditForm(storyId, pageIndex, panelIndex, panel));
  });
  return view;
}

function renderPanelEditForm(storyId, pageIndex, panelIndex, panel) {
  const form = el(`
    <div class="panel-card panel-edit">
      <div class="panel-num">Panel ${panelIndex + 1}</div>
      <div>
        <label>Scene description</label>
        <textarea class="p-desc">${escapeHtml(panel.scene_description)}</textarea>
        <div class="row">
          <div>
            <label>Camera hint</label>
            <input type="text" class="p-camera" value="${escapeHtml(panel.camera_hint)}" />
          </div>
          <div>
            <label>Characters (comma-separated)</label>
            <input type="text" class="p-characters" value="${escapeHtml(panel.characters.join(", "))}" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>Locations (comma-separated)</label>
            <input type="text" class="p-locations" value="${escapeHtml((panel.locations || []).join(", "))}" />
          </div>
          <div>
            <label>Props (comma-separated)</label>
            <input type="text" class="p-props" value="${escapeHtml((panel.props || []).join(", "))}" />
          </div>
        </div>
        <label>Dialogue</label>
        <div class="dialogue-rows"></div>
        <button class="btn secondary p-add-line" type="button">+ Add line</button>
        <div class="row panel-edit-actions">
          <button class="btn p-save" type="button">Save</button>
          <button class="btn secondary p-cancel" type="button">Cancel</button>
        </div>
        <div class="status-line p-status"></div>
      </div>
    </div>
  `);

  const rowsHolder = form.querySelector(".dialogue-rows");

  function addDialogueRow(line) {
    const row = el(`
      <div class="dialogue-row row">
        <div><input type="text" class="dl-speaker" placeholder="Speaker" value="${escapeHtml(line.speaker)}" /></div>
        <div class="dl-kind-wrap">
          <select class="dl-kind">
            <option value="speech">speech</option>
            <option value="thought">thought</option>
            <option value="narration">narration</option>
          </select>
        </div>
        <div class="dl-text-wrap"><input type="text" class="dl-text" placeholder="Line text" value="${escapeHtml(
          line.text
        )}" /></div>
        <button class="dl-remove" type="button" title="Remove line">&times;</button>
      </div>
    `);
    row.querySelector(".dl-kind").value = line.kind;
    row.querySelector(".dl-remove").addEventListener("click", () => row.remove());
    rowsHolder.appendChild(row);
  }
  panel.dialogue.forEach(addDialogueRow);

  form.querySelector(".p-add-line").addEventListener("click", () => addDialogueRow({ speaker: "", text: "", kind: "speech" }));

  form.querySelector(".p-cancel").addEventListener("click", () => {
    form.replaceWith(renderPanelView(storyId, pageIndex, panelIndex, panel));
  });

  form.querySelector(".p-save").addEventListener("click", async () => {
    const saveBtn = form.querySelector(".p-save");
    const status = form.querySelector(".p-status");
    saveBtn.disabled = true;
    status.classList.remove("error");
    status.textContent = "Saving...";

    const splitList = (v) =>
      v
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    const updated = {
      scene_description: form.querySelector(".p-desc").value.trim(),
      camera_hint: form.querySelector(".p-camera").value.trim() || "medium shot",
      characters: splitList(form.querySelector(".p-characters").value),
      locations: splitList(form.querySelector(".p-locations").value),
      props: splitList(form.querySelector(".p-props").value),
      dialogue: Array.from(rowsHolder.querySelectorAll(".dialogue-row"))
        .map((row) => ({
          speaker: row.querySelector(".dl-speaker").value.trim(),
          text: row.querySelector(".dl-text").value.trim(),
          kind: row.querySelector(".dl-kind").value,
        }))
        .filter((line) => line.text),
    };

    try {
      await api(`/api/stories/${storyId}/pages/${pageIndex}/panels/${panelIndex}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updated),
      });
      Object.assign(panel, updated);
      form.replaceWith(renderPanelView(storyId, pageIndex, panelIndex, panel));
    } catch (e) {
      status.textContent = "Error: " + e.message;
      status.classList.add("error");
      saveBtn.disabled = false;
    }
  });

  return form;
}

function renderCharacters(characters, emptyMessage, storyId, kind) {
  const names = Object.keys(characters);
  if (names.length === 0) {
    return el(`<p class='empty-state'>${escapeHtml(emptyMessage || "No characters detected in this story.")}</p>`);
  }
  const grid = el("<div class='grid'></div>");
  for (const name of names) {
    const c = characters[name];
    const imgHtml = c.reference_image_url
      ? `<img src="${c.reference_image_url}" alt="${escapeHtml(name)}" />`
      : `<div class="placeholder">Not designed yet</div>`;
    const loraBadge = c.has_lora ? `<div class="meta">LoRA trained</div>` : "";
    const loraButton = kind === "registry" ? `<button class="btn secondary lora-train-btn">Train LoRA</button>` : "";
    const card = el(`
      <div class="card char-card">
        <button class="card-delete" title="Delete ${escapeHtml(name)}">&times;</button>
        ${imgHtml}
        <h3>${escapeHtml(name)}</h3>
        <div class="meta">${escapeHtml(c.description || "")}</div>
        ${loraBadge}
        ${loraButton}
        <div class="status-line lora-status"></div>
      </div>
    `);
    if (storyId && kind) {
      card.querySelector(".card-delete").addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete "${name}"? This removes it and any reference image permanently.`)) {
          return;
        }
        await api(`/api/stories/${storyId}/${kind}/${encodeURIComponent(name)}`, { method: "DELETE" });
        await refreshIdentityGrids(storyId);
      });
    }
    const loraBtn = card.querySelector(".lora-train-btn");
    if (loraBtn) {
      loraBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        startCharacterLoraTraining(storyId, name, loraBtn, card.querySelector(".lora-status"));
      });
    }
    grid.appendChild(card);
  }
  return grid;
}

async function startCharacterLoraTraining(storyId, name, button, status) {
  if (
    !confirm(
      `Train a LoRA for "${name}"? This generates a handful of extra portrait images and trains for several ` +
        "hundred steps directly on this machine's GPU - it can take a while (minutes, not seconds) and will " +
        "hold the GPU lock for the duration, blocking other adapt/cast/generate jobs."
    )
  ) {
    return;
  }
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting...";
  try {
    const { job_id } = await api(`/api/stories/${storyId}/characters/${encodeURIComponent(name)}/train-lora`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    pollLoraJob(job_id, storyId, button, status);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function pollLoraJob(jobId, storyId, button, status) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    status.textContent = job.message;
    if (job.status === "done") {
      await refreshIdentityGrids(storyId);
      return;
    }
    if (job.status === "error") {
      status.classList.add("error");
      button.disabled = false;
      return;
    }
    setTimeout(() => pollLoraJob(jobId, storyId, button, status), 2000);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function refreshIdentityGrids(storyId) {
  const registryResp = await api(`/api/stories/${storyId}/registry`);
  const locationsResp = await api(`/api/stories/${storyId}/locations`);
  const propsResp = await api(`/api/stories/${storyId}/props`);
  const grids = document.querySelectorAll(".grid, .empty-state");
  if (grids[0]) grids[0].replaceWith(renderCharacters(registryResp.characters, null, storyId, "registry"));
  if (grids[1])
    grids[1].replaceWith(
      renderCharacters(locationsResp.locations, "No location profiles for this story.", storyId, "locations")
    );
  if (grids[2])
    grids[2].replaceWith(renderCharacters(propsResp.props, "No prop profiles for this story.", storyId, "props"));
}

function renderCastPanel(storyId) {
  const wrap = el(`
    <div>
      <div class="status-line">
        Generates a reference portrait for every character, location, and prop up front, so you
        can review and approve them before spending GPU time on full page generation. Optional -
        "Generate Pages" below will design any missing ones automatically if you skip this.
      </div>
      <div class="row">
        <div>
          <label>Backend</label>
          <select id="c-backend">
            <option value="mock">mock (instant, no GPU)</option>
            <option value="diffusers">diffusers (real generation)</option>
          </select>
        </div>
        <div>
          <label>Steps</label>
          <input type="number" id="c-steps" value="20" min="1" max="100" />
        </div>
        <div>
          <label>Identity adapter</label>
          <select id="c-adapter">
            <option value="true">on</option>
            <option value="false">off</option>
          </select>
        </div>
        <div>
          <label>Adapter scale</label>
          <input type="number" id="c-scale" value="0.6" min="0" max="1" step="0.05" />
        </div>
      </div>
      <label><input type="checkbox" id="c-force" style="width:auto;display:inline-block;margin-right:6px;" />Regenerate existing references too</label>
      <button class="btn secondary" id="c-submit">Generate Cast</button>
      <div class="status-line" id="c-status"></div>
    </div>
  `);
  wrap.querySelector("#c-submit").addEventListener("click", () => startCastGeneration(storyId));
  return wrap;
}

async function startCastGeneration(storyId) {
  const backend = document.getElementById("c-backend").value;
  const steps = parseInt(document.getElementById("c-steps").value, 10);
  const use_identity_adapter = document.getElementById("c-adapter").value === "true";
  const identity_adapter_scale = parseFloat(document.getElementById("c-scale").value);
  const force = document.getElementById("c-force").checked;
  const status = document.getElementById("c-status");
  const button = document.getElementById("c-submit");

  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting...";

  try {
    const { job_id } = await api(`/api/stories/${storyId}/prepare-cast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, steps, use_identity_adapter, identity_adapter_scale, force }),
    });
    pollCastJob(job_id, storyId, status, button);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function pollCastJob(jobId, storyId, status, button) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    status.textContent = job.message;
    if (job.status === "done") {
      button.disabled = false;
      await refreshIdentityGrids(storyId);
      return;
    }
    if (job.status === "error") {
      status.classList.add("error");
      button.disabled = false;
      return;
    }
    setTimeout(() => pollCastJob(jobId, storyId, status, button), 1500);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

function renderGeneratePanel(storyId) {
  const wrap = el(`
    <div>
      <div class="row">
        <div>
          <label>Backend</label>
          <select id="g-backend">
            <option value="mock">mock (instant, no GPU)</option>
            <option value="diffusers">diffusers (real generation)</option>
          </select>
        </div>
        <div>
          <label>Steps</label>
          <input type="number" id="g-steps" value="20" min="1" max="100" />
        </div>
        <div>
          <label>Identity adapter</label>
          <select id="g-adapter">
            <option value="true">on</option>
            <option value="false">off</option>
          </select>
        </div>
        <div>
          <label>Adapter scale</label>
          <input type="number" id="g-scale" value="0.6" min="0" max="1" step="0.05" />
        </div>
        <div>
          <label>Character LoRA</label>
          <select id="g-char-lora">
            <option value="true">on</option>
            <option value="false">off</option>
          </select>
        </div>
        <div>
          <label>LoRA scale</label>
          <input type="number" id="g-char-lora-scale" value="0.8" min="0" max="2" step="0.05" />
        </div>
        <div>
          <label>Adapter composition</label>
          <select id="g-composition"><option value="">none</option></select>
        </div>
        <div>
          <label>Pose ControlNet</label>
          <select id="g-pose-controlnet">
            <option value="true">on</option>
            <option value="false">off</option>
          </select>
        </div>
        <div>
          <label>Pose scale</label>
          <input type="number" id="g-pose-controlnet-scale" value="0.5" min="0" max="2" step="0.05" />
        </div>
        <div>
          <label>Quality review</label>
          <select id="g-quality-review">
            <option value="false">off</option>
            <option value="true">on</option>
          </select>
        </div>
        <div>
          <label>Quality review retries</label>
          <input type="number" id="g-quality-review-retries" value="2" min="0" max="5" />
        </div>
      </div>
      <div class="status-line">
        Character LoRA only affects characters you've trained one for (see the character cards
        above) - anyone else still falls back to identity adapter conditioning.
      </div>
      <div class="status-line">
        Pose ControlNet only affects panels tagged with 2+ real (non-abstract) characters on a
        camera hint other than close-up/extreme close-up (identity adapter and character LoRA
        already skip such panels, since blending multiple identities isn't solved) - it fixes
        SDXL dropping/duplicating figures in such a panel, at the cost of a second full SDXL
        pipeline loaded on first use (one-time load/VRAM overhead per story, not a per-panel
        cost). Doesn't control which figure looks like which character.
      </div>
      <div class="status-line">
        Quality review loads a third model (Qwen2.5-VL, ~11GB) to check panels that resolved to
        exactly one real character against that character's reference portrait, regenerating up
        to the retry limit if the count comes back wrong (e.g. a spurious duplicate face). Off
        by default: real extra VRAM/time cost on top of an already resource-heavy pipeline.
      </div>
      <label><input type="checkbox" id="g-force" style="width:auto;display:inline-block;margin-right:6px;" />Regenerate existing pages too</label>
      <label>Seed <input type="number" id="g-seed" value="0" step="1" /></label>
      <div class="status-line">
        Off by default: a page whose image already exists on disk is reused rather than
        redrawn, so a job that stopped partway (a crash, an out-of-memory error) can be
        resumed by clicking Generate Pages again instead of starting over from page one.
      </div>
      <button class="btn" id="g-submit">Generate Pages</button>
      <div class="status-line" id="g-status"></div>
    </div>
  `);
  wrap.querySelector("#g-submit").addEventListener("click", () => startGeneration(storyId));
  loadGenerationCompositions(wrap.querySelector("#g-composition"));
  return wrap;
}

async function loadGenerationCompositions(select) {
  try {
    const { compositions } = await api("/api/adapters/compositions");
    for (const composition of compositions) {
      const option = document.createElement("option");
      option.value = composition.path;
      option.textContent = `${composition.name} ${composition.version} (${composition.component_count} adapters)`;
      select.appendChild(option);
    }
  } catch (_) {
    // Composition selection is optional; generation remains usable if the registry is unavailable.
  }
}

async function startGeneration(storyId) {
  const backend = document.getElementById("g-backend").value;
  const steps = parseInt(document.getElementById("g-steps").value, 10);
  const use_identity_adapter = document.getElementById("g-adapter").value === "true";
  const identity_adapter_scale = parseFloat(document.getElementById("g-scale").value);
  const use_character_lora = document.getElementById("g-char-lora").value === "true";
  const character_lora_scale = parseFloat(document.getElementById("g-char-lora-scale").value);
  const adapter_composition_path = document.getElementById("g-composition").value;
  const use_pose_controlnet = document.getElementById("g-pose-controlnet").value === "true";
  const pose_controlnet_scale = parseFloat(document.getElementById("g-pose-controlnet-scale").value);
  const use_quality_review = document.getElementById("g-quality-review").value === "true";
  const quality_review_max_retries = parseInt(document.getElementById("g-quality-review-retries").value, 10);
  const force = document.getElementById("g-force").checked;
  const seed = parseInt(document.getElementById("g-seed").value, 10) || 0;
  const status = document.getElementById("g-status");
  const button = document.getElementById("g-submit");

  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting...";

  try {
    const { job_id } = await api(`/api/stories/${storyId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend,
        steps,
        use_identity_adapter,
        identity_adapter_scale,
        use_character_lora,
        character_lora_scale,
        adapter_composition_path,
        use_pose_controlnet,
        pose_controlnet_scale,
        use_quality_review,
        quality_review_max_retries,
        seed,
        force,
      }),
    });
    pollJob(job_id, storyId, status, button);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

async function pollJob(jobId, storyId, status, button) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    status.textContent = job.message;
    if (job.status === "done") {
      button.disabled = false;
      const galleryHolder = document.getElementById("gallery-holder");
      const pagesResp = await api(`/api/stories/${storyId}/pages`);
      renderGallery(galleryHolder, pagesResp);
      await refreshIdentityGrids(storyId);
      return;
    }
    if (job.status === "error") {
      status.classList.add("error");
      button.disabled = false;
      return;
    }
    setTimeout(() => pollJob(jobId, storyId, status, button), 1500);
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.classList.add("error");
    button.disabled = false;
  }
}

function renderGallery(holder, pagesResp) {
  holder.innerHTML = "";
  if (pagesResp.pages.length === 0) {
    holder.appendChild(el("<p class='empty-state'>No pages generated yet.</p>"));
    return;
  }
  if (pagesResp.pdf_url) {
    const link = el(`<a class="btn secondary" href="${pagesResp.pdf_url}" target="_blank">Download PDF</a>`);
    holder.appendChild(link);
  }
  const gallery = el("<div class='pages-gallery'></div>");
  for (const url of pagesResp.pages) {
    const img = el(`<img src="${url}" />`);
    img.addEventListener("click", () => showLightbox(url));
    gallery.appendChild(img);
  }
  holder.appendChild(gallery);
}

function showLightbox(url) {
  const box = el(`<div class="lightbox"><img src="${url}" /></div>`);
  box.addEventListener("click", () => box.remove());
  document.body.appendChild(box);
}

showLibrary();
