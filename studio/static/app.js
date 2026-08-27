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
        <h3>${escapeHtml(s.title)}</h3>
        <div class="meta">${s.page_count} pages &middot; ${s.panel_count} panels</div>
        ${s.has_output ? "<span class='badge'>Generated</span>" : ""}
      </div>
    `);
    card.addEventListener("click", () => showStoryDetail(s.id));
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
      <textarea id="f-profiles" placeholder="Aiko: young woman, shoulder-length black hair, tan raincoat, red satchel&#10;Ren: young man, short dark hair, casual jacket" style="min-height:100px"></textarea>
      <div class="status-line">List as many characters as your cast needs, one per line. Guarantees each is recognized and uses your description instead of a guessed one.</div>
      <button class="btn" id="f-submit">Adapt Story (Stage A)</button>
      <div class="status-line" id="f-status"></div>
    </div>
  `)
  );
  document.getElementById("f-file").addEventListener("change", handleFileUpload);
  document.getElementById("f-style-preset").addEventListener("change", (e) => {
    const preset = STYLE_PRESETS[parseInt(e.target.value, 10)];
    if (preset.value !== null) {
      document.getElementById("f-style").value = preset.value;
    }
  });
  document.getElementById("f-submit").addEventListener("click", submitNewStory);
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const status = document.getElementById("f-file-status");
  const prose = document.getElementById("f-prose");
  const ext = file.name.toLowerCase().split(".").pop();

  status.classList.remove("error");

  if (ext === "txt" || ext === "md") {
    status.textContent = `Loaded ${file.name}.`;
    prose.value = await file.text();
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
    prose.value = text;
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
      body: JSON.stringify({ id, title, prose, style_prompt, character_profiles }),
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

  const [story, registryResp, pagesResp] = await Promise.all([
    api(`/api/stories/${id}`),
    api(`/api/stories/${id}/registry`),
    api(`/api/stories/${id}/pages`),
  ]);

  main.innerHTML = "";
  main.appendChild(el(`<div class="section-title">${escapeHtml(story.title)}</div>`));

  main.appendChild(renderScript(story));
  main.appendChild(el(`<div class="section-title">Character Sheets</div>`));
  main.appendChild(renderCharacters(registryResp.characters));
  main.appendChild(el(`<div class="section-title">Generate Pages</div>`));
  main.appendChild(renderGeneratePanel(id));
  main.appendChild(el(`<div class="section-title">Pages</div>`));
  const galleryHolder = el(`<div id="gallery-holder"></div>`);
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
      const dialogueHtml = panel.dialogue
        .map(
          (line) =>
            `<div class="bubble ${line.kind === "narration" ? "narration" : ""}"><span class="speaker">${escapeHtml(
              line.speaker
            )}:</span>${escapeHtml(line.text)}</div>`
        )
        .join("");
      const chips = panel.characters.map((c) => `<span class="chip">${escapeHtml(c)}</span>`).join("");
      pageEl.appendChild(
        el(`
        <div class="panel-card">
          <div class="panel-num">Panel ${panelIndex + 1}<br/><span class="camera">${escapeHtml(
          panel.camera_hint
        )}</span></div>
          <div>
            <div class="desc">${escapeHtml(panel.scene_description)}</div>
            <div>${chips}</div>
            <div>${dialogueHtml}</div>
          </div>
        </div>
      `)
      );
    });
    wrap.appendChild(pageEl);
  });
  return wrap;
}

function renderCharacters(characters) {
  const names = Object.keys(characters);
  if (names.length === 0) {
    return el("<p class='empty-state'>No characters detected in this story.</p>");
  }
  const grid = el("<div class='grid'></div>");
  for (const name of names) {
    const c = characters[name];
    const imgHtml = c.reference_image_url
      ? `<img src="${c.reference_image_url}" alt="${escapeHtml(name)}" />`
      : `<div class="placeholder">Not designed yet</div>`;
    grid.appendChild(
      el(`
      <div class="card char-card">
        ${imgHtml}
        <h3>${escapeHtml(name)}</h3>
        <div class="meta">${escapeHtml(c.description || "")}</div>
      </div>
    `)
    );
  }
  return grid;
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
  const status = document.getElementById("g-status");
  const button = document.getElementById("g-submit");

  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Starting...";

  try {
    const { job_id } = await api(`/api/stories/${storyId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, steps, use_identity_adapter, identity_adapter_scale }),
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
      const registryResp = await api(`/api/stories/${storyId}/registry`);
      const charSection = document.querySelectorAll(".grid")[0];
      if (charSection) charSection.replaceWith(renderCharacters(registryResp.characters));
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
