const main = document.getElementById("main");
const navLibrary = document.getElementById("nav-library");
const navNew = document.getElementById("nav-new");

function setNav(active) {
  navLibrary.classList.toggle("active", active === "library");
  navNew.classList.toggle("active", active === "new");
}

navLibrary.addEventListener("click", showLibrary);
navNew.addEventListener("click", showNewStory);

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
      body: JSON.stringify({ id, title, prose, style_prompt, character_profiles, location_profiles, prop_profiles }),
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
    const card = el(`
      <div class="card char-card">
        <button class="card-delete" title="Delete ${escapeHtml(name)}">&times;</button>
        ${imgHtml}
        <h3>${escapeHtml(name)}</h3>
        <div class="meta">${escapeHtml(c.description || "")}</div>
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
    grid.appendChild(card);
  }
  return grid;
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
          <input type="number" id="c-steps" value="4" min="1" max="50" />
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
          <input type="number" id="g-steps" value="4" min="1" max="50" />
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
      </div>
      <label><input type="checkbox" id="g-force" style="width:auto;display:inline-block;margin-right:6px;" />Regenerate existing pages too</label>
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
  return wrap;
}

async function startGeneration(storyId) {
  const backend = document.getElementById("g-backend").value;
  const steps = parseInt(document.getElementById("g-steps").value, 10);
  const use_identity_adapter = document.getElementById("g-adapter").value === "true";
  const identity_adapter_scale = parseFloat(document.getElementById("g-scale").value);
  const force = document.getElementById("g-force").checked;
  const status = document.getElementById("g-status");
  const button = document.getElementById("g-submit");

  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting...";

  try {
    const { job_id } = await api(`/api/stories/${storyId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, steps, use_identity_adapter, identity_adapter_scale, force }),
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
