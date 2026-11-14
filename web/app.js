/**
 * Autonomous Video Studio - Web Frontend Application (Vanilla ES Modules)
 * Strict KISS / YAGNI / Minimalist engineering: Zero build step.
 */

// -----------------------------------------------------------------------------
// Application State
// -----------------------------------------------------------------------------

let STAGE_ORDER = [
  "ingest", "script", "edl", "base_cut", "bgm_broll", "graphics", "composite", "deliverables"
];

const state = {
  currentView: "new",          // "new" | "job"
  currentJobId: null,
  currentJob: null,
  activeEventSource: null,
  jobTimerInterval: null,

  // New Run Creator State
  folderPath: "/home/sandbox/ContentCreation/raw_clips",
  availableClips: [],         // [{ name, path, duration_s, width, height, thumb_url }]
  selectedClips: [],          // [{ name, path, duration_s, width, height, thumb_url }]
  selectedPreset: "story_vertical",
  whisperAccuracy: "fast",
  aiBroll: "off",

  // Delivery & Gate State
  selectedScriptOption: "A",
  selectedBgm: null,
  activeDescTab: "hook",
  descriptionsData: null,

  // AI Models Routing State
  agyModels: [
    { id: "gemini-3.8-flash-high", label: "Gemini 3.8 Flash High", tier: "fast", note: "Sub-second speed, unlimited quota" },
    { id: "gemini-4-argon-medium", label: "Gemini 4 Argon Medium", tier: "balanced", note: "Deep reasoning & sharp EDL joins" },
    { id: "claude-sonnet-5-5-medium", label: "Claude Sonnet 5.5", tier: "creative_boost", note: "Premium nuance (uses quota quickly)" },
    { id: "claude-opus-5-5-medium", label: "Claude Opus 5.5", tier: "creative_boost", note: "Maximum creative depth (heavy quota)" }
  ],
  defaultModels: {
    creative_model: "gemini-3.8-flash-high",
    mechanical_model: "gemini-3.8-flash-high",
    vision_model: "gemini-3.8-flash-high",
    copilot_model: "gemini-3.8-flash-high"
  },

  // Antigravity Co-Pilot Workspace State
  copilotSessionId: null,
  copilotBusy: false,
  copilotStartTs: null,
  copilotElapsedTimer: null,
  copilotMessages: [],
  copilotCollapsed: false
};

// -----------------------------------------------------------------------------
// Toast Notifications
// -----------------------------------------------------------------------------

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${type === "error" ? "⚠️" : (type === "success" ? "✓" : "ℹ️")}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

// -----------------------------------------------------------------------------
// Format Utilities
// -----------------------------------------------------------------------------

function formatDuration(seconds) {
  if (!seconds || isNaN(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatBytes(bytes) {
  if (!bytes || isNaN(bytes)) return "";
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// -----------------------------------------------------------------------------
// Initialization & Health
// -----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initToolReadiness();
  initNavigation();
  initClipSelector();
  initUploadDropzone();
  initPresetsAndOptions();
  initModelRouting();
  initJobControls();
  initCopilot();
  initDescriptionsTabs();
  initConsoleControls();

  // Load initial presets, jobs & folder clips
  loadPresetsConfig();
  loadJobHistory();
  loadFolderClips(state.folderPath);
});

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    updateToolPills(data.tools || {});
    if (data.default_roots && data.default_roots.length) {
      renderQuickRoots(data.default_roots);
    }
  } catch (err) {
    console.error("Health check error:", err);
  }
}

function initToolReadiness() {
  checkHealth();
  const btnRefresh = document.getElementById("btn-refresh-health");
  if (btnRefresh) {
    btnRefresh.addEventListener("click", () => {
      btnRefresh.classList.add("spinning");
      checkHealth().finally(() => setTimeout(() => btnRefresh.classList.remove("spinning"), 500));
    });
  }
}

function updateToolPills(tools) {
  const container = document.getElementById("tool-readiness");
  if (!container) return;

  for (const [toolName, info] of Object.entries(tools)) {
    const pill = container.querySelector(`[data-tool="${toolName}"]`);
    if (pill) {
      const isReady = typeof info === "boolean" ? info : (info && info.available === true);
      pill.classList.remove("ready", "missing");
      pill.classList.add(isReady ? "ready" : "missing");
      const pathInfo = typeof info === "object" && info?.path ? ` (${info.path})` : "";
      pill.title = `${toolName}: ${isReady ? "Available" : "Not Found"}${pathInfo}`;
    }
  }
}

function renderQuickRoots(roots) {
  const container = document.getElementById("quick-root-buttons");
  if (!container) return;
  container.innerHTML = "";

  roots.forEach((rootPath, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `btn-chip ${idx === 0 ? "active" : ""}`;
    btn.textContent = rootPath.split("/").pop() || rootPath;
    btn.dataset.path = rootPath;
    btn.addEventListener("click", () => {
      container.querySelectorAll(".btn-chip").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.folderPath = rootPath;
      document.getElementById("input-folder-path").value = rootPath;
      loadFolderClips(rootPath);
    });
    container.appendChild(btn);
  });
}

// -----------------------------------------------------------------------------
// Navigation & Views
// -----------------------------------------------------------------------------

function initNavigation() {
  const btnNew = document.getElementById("btn-nav-new");
  const jobSelect = document.getElementById("job-history-select");

  if (btnNew) {
    btnNew.addEventListener("click", () => switchView("new"));
  }

  if (jobSelect) {
    jobSelect.addEventListener("change", (e) => {
      const val = e.target.value;
      if (val === "new") {
        switchView("new");
      } else if (val) {
        openJob(val);
      }
    });
  }
}

function switchView(viewName, jobId = null) {
  state.currentView = viewName;
  const viewNew = document.getElementById("view-new-run");
  const viewJob = document.getElementById("view-job-monitor");

  if (viewName === "new") {
    viewNew.classList.add("active");
    viewJob.classList.remove("active");
    document.getElementById("job-history-select").value = "new";
    if (state.activeEventSource) {
      state.activeEventSource.close();
      state.activeEventSource = null;
    }
    clearInterval(state.jobTimerInterval);
  } else {
    viewNew.classList.remove("active");
    viewJob.classList.add("active");
    if (jobId) {
      document.getElementById("job-history-select").value = jobId;
    }
  }
}

async function loadJobHistory() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    const select = document.getElementById("job-history-select");
    if (!select) return;

    // Preserve the first option ("+ New Run")
    select.innerHTML = '<option value="new">+ New Video Run</option>';

    jobs.forEach(j => {
      const opt = document.createElement("option");
      opt.value = j.id;
      const statusIcon = j.status === "completed" ? "✓" : (j.status === "failed" ? "✗" : "▶");
      opt.textContent = `${statusIcon} ${j.id} (${j.status || "pending"})`;
      select.appendChild(opt);
    });

    if (state.currentJobId) {
      select.value = state.currentJobId;
    }
  } catch (err) {
    console.error("Failed to load jobs list:", err);
  }
}

// -----------------------------------------------------------------------------
// Clip Selection & Management
// -----------------------------------------------------------------------------

function initClipSelector() {
  const btnLoad = document.getElementById("btn-load-folder");
  const inputFolder = document.getElementById("input-folder-path");
  const btnSelectAll = document.getElementById("btn-select-all");
  const btnDeselectAll = document.getElementById("btn-deselect-all");

  if (btnLoad && inputFolder) {
    btnLoad.addEventListener("click", () => {
      const p = inputFolder.value.trim();
      if (p) {
        state.folderPath = p;
        loadFolderClips(p);
      }
    });
    inputFolder.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        btnLoad.click();
      }
    });
  }

  if (btnSelectAll) {
    btnSelectAll.addEventListener("click", () => {
      state.selectedClips = [...state.availableClips];
      renderClipsGrid();
      renderSelectedSequence();
      updateSelectionSummary();
    });
  }

  if (btnDeselectAll) {
    btnDeselectAll.addEventListener("click", () => {
      state.selectedClips = [];
      renderClipsGrid();
      renderSelectedSequence();
      updateSelectionSummary();
    });
  }
}

async function loadFolderClips(folderPath) {
  const grid = document.getElementById("clips-grid");
  if (!grid) return;
  grid.innerHTML = '<div class="empty-state">Scanning directory for video clips...</div>';

  try {
    const res = await fetch(`/api/clips?folder=${encodeURIComponent(folderPath)}`);
    if (!res.ok) {
      const err = await res.json();
      grid.innerHTML = `<div class="empty-state text-danger">${escapeHtml(err.detail || "Failed to load clips")}</div>`;
      return;
    }
    const data = await res.json();
    state.availableClips = data.clips || [];

    if (state.availableClips.length === 0) {
      grid.innerHTML = '<div class="empty-state">No compatible video files (.mp4, .mov, .webm) found in this folder.</div>';
    } else {
      renderClipsGrid();
    }
  } catch (err) {
    grid.innerHTML = `<div class="empty-state text-danger">Error: ${escapeHtml(err.message)}</div>`;
  }
}

function renderClipsGrid() {
  const grid = document.getElementById("clips-grid");
  if (!grid) return;
  grid.innerHTML = "";

  state.availableClips.forEach(clip => {
    const isSelected = state.selectedClips.some(c => c.path === clip.path);
    const orderIndex = state.selectedClips.findIndex(c => c.path === clip.path) + 1;

    const card = document.createElement("div");
    card.className = `clip-card ${isSelected ? "selected" : ""}`;
    card.dataset.path = clip.path;

    card.innerHTML = `
      <div class="clip-thumb-wrap">
        <img class="clip-thumb-img" src="${clip.thumb_url}" alt="${escapeHtml(clip.name)}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%2260%22><rect width=%22100%25%22 height=%22100%25%22 fill=%22%23222%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%23666%22 font-size=%2212%22 text-anchor=%22middle%22 dy=%22.3em%22>No Preview</text></svg>'">
        <span class="clip-order-badge">${orderIndex > 0 ? '#' + orderIndex : ''}</span>
        <span class="clip-res-badge">${clip.width && clip.height ? `${clip.width}x${clip.height}` : 'Video'}</span>
        <span class="clip-duration-badge">${formatDuration(clip.duration_s)}</span>
      </div>
      <div class="clip-info">
        <div class="clip-title" title="${escapeHtml(clip.name)}">${escapeHtml(clip.name)}</div>
      </div>
    `;

    card.addEventListener("click", () => toggleClipSelection(clip));
    grid.appendChild(card);
  });

  updateSelectionSummary();
}

function toggleClipSelection(clip) {
  const existingIdx = state.selectedClips.findIndex(c => c.path === clip.path);
  if (existingIdx >= 0) {
    state.selectedClips.splice(existingIdx, 1);
  } else {
    state.selectedClips.push(clip);
  }
  renderClipsGrid();
  renderSelectedSequence();
  updateSelectionSummary();
}

function renderSelectedSequence() {
  const container = document.getElementById("sequence-list");
  if (!container) return;
  container.innerHTML = "";

  if (state.selectedClips.length === 0) {
    container.innerHTML = '<span class="text-muted text-sm">No clips selected yet. Click clips above to sequence.</span>';
    return;
  }

  state.selectedClips.forEach((clip, idx) => {
    const item = document.createElement("div");
    item.className = "sequence-item";
    item.innerHTML = `
      <span class="seq-idx">#${idx + 1}</span>
      <span class="seq-name" title="${escapeHtml(clip.name)}">${escapeHtml(clip.name.slice(0, 16))}...</span>
      <span class="text-muted text-xs">(${formatDuration(clip.duration_s)})</span>
      <button class="seq-btn seq-up" title="Move earlier" ${idx === 0 ? 'disabled' : ''}>▲</button>
      <button class="seq-btn seq-down" title="Move later" ${idx === state.selectedClips.length - 1 ? 'disabled' : ''}>▼</button>
      <button class="seq-btn seq-remove" title="Remove">✕</button>
    `;

    item.querySelector(".seq-up")?.addEventListener("click", (e) => {
      e.stopPropagation();
      if (idx > 0) {
        const temp = state.selectedClips[idx];
        state.selectedClips[idx] = state.selectedClips[idx - 1];
        state.selectedClips[idx - 1] = temp;
        renderClipsGrid();
        renderSelectedSequence();
        updateSelectionSummary();
      }
    });

    item.querySelector(".seq-down")?.addEventListener("click", (e) => {
      e.stopPropagation();
      if (idx < state.selectedClips.length - 1) {
        const temp = state.selectedClips[idx];
        state.selectedClips[idx] = state.selectedClips[idx + 1];
        state.selectedClips[idx + 1] = temp;
        renderClipsGrid();
        renderSelectedSequence();
        updateSelectionSummary();
      }
    });

    item.querySelector(".seq-remove")?.addEventListener("click", (e) => {
      e.stopPropagation();
      state.selectedClips.splice(idx, 1);
      renderClipsGrid();
      renderSelectedSequence();
      updateSelectionSummary();
    });

    container.appendChild(item);
  });
}

function updateSelectionSummary() {
  const count = state.selectedClips.length;
  const badge = document.getElementById("selected-clips-badge");
  const btnGen = document.getElementById("btn-generate-video");
  const totalDurElem = document.getElementById("total-selection-duration");

  const totalDur = state.selectedClips.reduce((acc, c) => acc + (c.duration_s || 0), 0);
  if (totalDurElem) totalDurElem.textContent = `Total: ${formatDuration(totalDur)}`;

  if (badge) badge.textContent = `${count} clip${count === 1 ? '' : 's'} selected`;

  if (btnGen) {
    btnGen.disabled = count === 0;
    btnGen.querySelector(".btn-text").textContent = `Generate Video (${count} clip${count === 1 ? '' : 's'} selected)`;
  }
}

// -----------------------------------------------------------------------------
// Drag & Drop Upload Zone
// -----------------------------------------------------------------------------

function initUploadDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-upload-input");
  const btnBrowse = document.getElementById("btn-browse-files");

  if (!dropzone || !fileInput) return;

  btnBrowse?.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  ["dragleave", "dragend"].forEach(ev => {
    dropzone.addEventListener(ev, () => dropzone.classList.remove("drag-over"));
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) {
      handleFilesUpload(Array.from(e.dataTransfer.files));
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      handleFilesUpload(Array.from(e.target.files));
      fileInput.value = "";
    }
  });
}

async function handleFilesUpload(files) {
  const validFiles = files.filter(f => {
    const ext = f.name.toLowerCase();
    return ext.endsWith(".mp4") || ext.endsWith(".mov") || ext.endsWith(".webm") || ext.endsWith(".mkv");
  });

  if (validFiles.length === 0) {
    showToast("Please upload video files (.mp4, .mov, .webm, .mkv)", "error");
    return;
  }

  const progressWrap = document.getElementById("upload-progress-container");
  const progressBar = document.getElementById("upload-progress-bar");
  const progressText = document.getElementById("upload-progress-text");

  progressWrap?.classList.remove("hidden");
  if (progressBar) progressBar.style.width = "10%";
  if (progressText) progressText.textContent = `Uploading ${validFiles.length} file(s)...`;

  const formData = new FormData();
  validFiles.forEach(f => formData.append("files", f));

  try {
    if (progressBar) progressBar.style.width = "50%";
    const res = await fetch("/api/uploads", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Upload failed on server");
    const uploadedClips = await res.json();

    if (progressBar) progressBar.style.width = "100%";
    if (progressText) progressText.textContent = `Uploaded ${uploadedClips.length} clips!`;

    // Add to available and selected list
    uploadedClips.forEach(c => {
      state.availableClips.unshift(c);
      state.selectedClips.push(c);
    });

    renderClipsGrid();
    renderSelectedSequence();
    updateSelectionSummary();
    showToast(`Successfully uploaded ${uploadedClips.length} clip(s)!`, "success");

    setTimeout(() => {
      progressWrap?.classList.add("hidden");
    }, 2000);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, "error");
    if (progressText) progressText.textContent = `Upload failed: ${err.message}`;
  }
}

// -----------------------------------------------------------------------------
// Presets, AI Model Routing & Job Creation
// -----------------------------------------------------------------------------

function isQuotaHeavyModel(modelId, tier = "") {
  if (!modelId) return false;
  const m = modelId.toLowerCase();
  return m.includes("sonnet") || m.includes("opus") || tier === "creative_boost";
}

function populateModelDropdown(selectElem, models, selectedVal, groupTiers = true) {
  if (!selectElem) return;
  selectElem.innerHTML = "";

  if (groupTiers) {
    const fastGroup = models.filter(m => m.tier === "fast" || m.tier === "balanced");
    const boostGroup = models.filter(m => m.tier === "creative_boost");

    if (fastGroup.length > 0) {
      const optGroup1 = document.createElement("optgroup");
      optGroup1.label = "Fast & Balanced (Unlimited Quota)";
      fastGroup.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.label} — ${m.note || m.tier}`;
        opt.dataset.tier = m.tier;
        optGroup1.appendChild(opt);
      });
      selectElem.appendChild(optGroup1);
    }

    if (boostGroup.length > 0) {
      const optGroup2 = document.createElement("optgroup");
      optGroup2.label = "Creative Boost (Uses Quota Quickly)";
      boostGroup.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.label} — ${m.note || 'Quota Intensive'}`;
        opt.dataset.tier = m.tier;
        optGroup2.appendChild(opt);
      });
      selectElem.appendChild(optGroup2);
    }
  } else {
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      opt.dataset.tier = m.tier;
      selectElem.appendChild(opt);
    });
  }

  if (selectedVal && models.some(m => m.id === selectedVal)) {
    selectElem.value = selectedVal;
  } else if (models.length > 0) {
    selectElem.value = models[0].id;
  }
}

function updateQuotaBadge(selectElem, badgeElem) {
  if (!selectElem || !badgeElem) return;
  const selectedOpt = selectElem.selectedOptions[0];
  const isHeavy = isQuotaHeavyModel(selectElem.value, selectedOpt?.dataset?.tier);
  badgeElem.classList.toggle("hidden", !isHeavy);
}

function initModelRouting() {
  const creativeSel = document.getElementById("select-creative-model");
  const mechSel = document.getElementById("select-mechanical-model");
  const monCreativeSel = document.getElementById("mon-select-creative-model");
  const monMechSel = document.getElementById("mon-select-mechanical-model");
  const copilotSel = document.getElementById("copilot-model-select");

  const badgeCreative = document.getElementById("badge-creative-quota");
  const badgeMech = document.getElementById("badge-mechanical-quota");
  const monBadgeCreative = document.getElementById("mon-badge-creative-quota");
  const monBadgeMech = document.getElementById("mon-badge-mechanical-quota");
  const copilotQuotaBadge = document.getElementById("copilot-quota-badge");

  // Populate New Run Model selectors
  populateModelDropdown(creativeSel, state.agyModels, state.defaultModels.creative_model, true);
  populateModelDropdown(mechSel, state.agyModels, state.defaultModels.mechanical_model, true);

  // Populate Job Monitor inline selectors
  populateModelDropdown(monCreativeSel, state.agyModels, state.defaultModels.creative_model, false);
  populateModelDropdown(monMechSel, state.agyModels, state.defaultModels.mechanical_model, false);

  // Populate Co-Pilot selector
  populateModelDropdown(copilotSel, state.agyModels, state.defaultModels.copilot_model, true);

  // Event listeners for New Run selectors
  creativeSel?.addEventListener("change", () => {
    updateQuotaBadge(creativeSel, badgeCreative);
  });
  mechSel?.addEventListener("change", () => {
    updateQuotaBadge(mechSel, badgeMech);
  });

  // Event listeners for Job Monitor inline selectors (live patch)
  monCreativeSel?.addEventListener("change", () => {
    updateQuotaBadge(monCreativeSel, monBadgeCreative);
    patchJobModels();
  });
  monMechSel?.addEventListener("change", () => {
    updateQuotaBadge(monMechSel, monBadgeMech);
    patchJobModels();
  });

  // Event listener for Co-Pilot selector
  copilotSel?.addEventListener("change", () => {
    updateQuotaBadge(copilotSel, copilotQuotaBadge);
  });

  // Initial badge check
  updateQuotaBadge(creativeSel, badgeCreative);
  updateQuotaBadge(mechSel, badgeMech);
  updateQuotaBadge(monCreativeSel, monBadgeCreative);
  updateQuotaBadge(monMechSel, monBadgeMech);
  updateQuotaBadge(copilotSel, copilotQuotaBadge);
}

async function patchJobModels() {
  if (!state.currentJobId) return;
  const creative = document.getElementById("mon-select-creative-model")?.value;
  const mechanical = document.getElementById("mon-select-mechanical-model")?.value;
  if (!creative || !mechanical) return;

  try {
    const res = await fetch(`/api/jobs/${state.currentJobId}/models`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        creative_model: creative,
        mechanical_model: mechanical,
        vision_model: mechanical
      })
    });
    if (res.ok) {
      showToast("Updated AI models routing for job", "success");
      const payload = await res.json().catch(() => null);
      if (payload?.job?.config) {
        state.currentJob = payload.job;
      } else if (payload?.config) {
        state.currentJob = payload;
      }
    } else {
      const err = await res.json().catch(() => ({}));
      showToast(`Model update failed: ${err.detail || "Error"}`, "error");
    }
  } catch (err) {
    showToast(`Model update error: ${err.message}`, "error");
  }
}

async function loadPresetsConfig() {
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) return;
    const data = await res.json();
    if (data.stages && data.stages.length) {
      STAGE_ORDER = data.stages;
      const retrySelect = document.getElementById("select-retry-stage");
      if (retrySelect) {
        retrySelect.innerHTML = "";
        data.stages.forEach(s => {
          const opt = document.createElement("option");
          opt.value = s;
          opt.textContent = `Retry: ${s.replace(/_/g, " ").toUpperCase()}`;
          retrySelect.appendChild(opt);
        });
      }
    }
    if (data.agy_models && Array.isArray(data.agy_models) && data.agy_models.length > 0) {
      state.agyModels = data.agy_models;
    }
    if (data.default_models && typeof data.default_models === "object") {
      state.defaultModels = { ...state.defaultModels, ...data.default_models };
    }
    initModelRouting();
  } catch (err) {
    console.error("Failed to load presets:", err);
    initModelRouting();
  }
}

function initPresetsAndOptions() {
  // Preset card selection
  const presetCards = document.querySelectorAll(".preset-card");
  presetCards.forEach(card => {
    card.addEventListener("click", () => {
      presetCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      state.selectedPreset = card.dataset.preset;
    });
  });

  // Whisper Accuracy segmented control
  const whisperBtns = document.querySelectorAll("#whisper-control .segment-btn");
  whisperBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      whisperBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.whisperAccuracy = btn.dataset.val;
    });
  });

  // AI B-Roll selector
  const brollSelect = document.getElementById("select-ai-broll");
  if (brollSelect) {
    brollSelect.addEventListener("change", (e) => {
      state.aiBroll = e.target.value;
    });
  }

  // Primary Generate CTA
  const btnGen = document.getElementById("btn-generate-video");
  if (btnGen) {
    btnGen.addEventListener("click", triggerGenerateVideo);
  }
}

async function triggerGenerateVideo() {
  if (state.selectedClips.length === 0) {
    showToast("Please select at least 1 video clip to generate.", "error");
    return;
  }

  const promptText = document.getElementById("input-story-prompt")?.value.trim() || "";
  const gateA = document.getElementById("toggle-gate-a")?.checked || false;
  const gateB = document.getElementById("toggle-gate-b")?.checked || false;

  const creativeModel = document.getElementById("select-creative-model")?.value || state.defaultModels.creative_model;
  const mechanicalModel = document.getElementById("select-mechanical-model")?.value || state.defaultModels.mechanical_model;
  const copilotModel = document.getElementById("copilot-model-select")?.value || state.defaultModels.copilot_model;
  const visionModel = mechanicalModel;

  const payload = {
    config: {
      preset_id: state.selectedPreset,
      story_prompt: promptText,
      gate_a: gateA,
      gate_b: gateB,
      whisper_model: state.whisperAccuracy === "high" ? "medium.en" : "small.en",
      ai_broll: state.aiBroll,
      models: {
        creative_model: creativeModel,
        mechanical_model: mechanicalModel,
        vision_model: visionModel,
        copilot_model: copilotModel
      }
    },
    clip_paths: state.selectedClips.map(c => c.path),
    start_immediately: true
  };

  const btnGen = document.getElementById("btn-generate-video");
  if (btnGen) {
    btnGen.disabled = true;
    btnGen.querySelector(".btn-text").textContent = "Creating Studio Job...";
  }

  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to create job");
    }

    const job = await res.json();
    showToast(`Job ${job.id} created! Starting pipeline...`, "success");
    await loadJobHistory();
    openJob(job.id);
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  } finally {
    updateSelectionSummary();
  }
}

// -----------------------------------------------------------------------------
// Job Monitor & Progress Rail
// -----------------------------------------------------------------------------

async function openJob(jobId) {
  state.currentJobId = jobId;
  switchView("job", jobId);
  renderStageRail({});
  clearConsoleLogs();
  appendConsoleLog(`Connecting live events for ${jobId}...`, "log-system");

  // Fetch initial enriched job state immediately
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (res.ok) {
      const job = await res.json();
      updateJobUI(job);
    }
  } catch (err) {
    console.warn("Could not fetch initial job state:", err);
  }

  // Connect to SSE
  connectJobEvents(jobId);
}

function connectJobEvents(jobId) {
  if (state.activeEventSource) {
    state.activeEventSource.close();
    state.activeEventSource = null;
  }

  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.activeEventSource = es;

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleJobEvent(data);
    } catch (err) {
      console.error("Error parsing SSE event:", err, e.data);
    }
  };

  es.onerror = (err) => {
    appendConsoleLog("SSE connection disconnected. Reconnecting in background...", "log-warn");
  };
}

function handleJobEvent(event) {
  if (!event || typeof event !== "object") return;
  const evType = event.type || event.event || "update";

  // Apply job update immediately whenever event carries job object
  if (event.job) {
    updateJobUI(event.job);
  }

  const msg = event.message || event.data?.message;

  if (evType === "log") {
    if (msg) {
      appendConsoleLog(msg, event.level ? `log-${event.level}` : "log-info");
    }
  } else if (evType === "stage_started") {
    const stage = event.stage || event.data?.stage || "";
    appendConsoleLog(`Stage started: ${stage}`, "log-info");
  } else if (evType === "stage_completed") {
    const stage = event.stage || event.data?.stage || "";
    appendConsoleLog(`Stage completed: ${stage}`, "log-info");
  } else if (evType === "stage_failed") {
    const err = event.error || event.data?.error || "";
    appendConsoleLog(`Stage failed: ${err}`, "log-error");
  } else if (evType === "awaiting_approval") {
    const gate = event.gate || event.data?.gate || "";
    appendConsoleLog(`Awaiting Gate ${gate} approval`, "log-warn");
  } else if (evType === "job_completed") {
    appendConsoleLog("All pipeline stages completed successfully!", "log-info");
  } else if (evType === "recomposite_completed") {
    appendConsoleLog("Recomposite completed successfully!", "log-info");
  } else if (evType === "copilot_progress") {
    updateCopilotProgress(event);
  } else if (evType === "copilot_reply") {
    if (event.job) {
      updateJobUI(event.job);
    } else if (event.copilot) {
      renderCopilotUI(event.copilot);
    }
    setCopilotBusy(false);
  } else if (evType === "stage_change") {
    const stage = event.stage || event.data?.stage;
    const status = event.status || event.data?.status;
    appendConsoleLog(`Stage transition: ${stage} -> ${status}`, "log-info");
    if (state.currentJob) {
      state.currentJob.stages = state.currentJob.stages || {};
      state.currentJob.stages[stage] = { status: status };
      renderStageRail(state.currentJob.stages);
    }
  } else if (msg) {
    appendConsoleLog(msg, "log-info");
  }
}

function updateJobUI(job) {
  state.currentJob = job;

  // Header summary
  const idElem = document.getElementById("mon-job-id");
  const statusElem = document.getElementById("mon-job-status");
  const presetElem = document.getElementById("mon-job-preset");

  if (idElem) idElem.textContent = job.id;
  if (statusElem) {
    statusElem.textContent = (job.status || "running").replace("_", " ");
    statusElem.className = `status-pill status-${job.status || "running"}`;
  }
  if (presetElem) {
    presetElem.textContent = job.config?.preset_id?.replace("_", " ").toUpperCase() || "STORY VERTICAL";
  }

  // Mid-job model routing dropdowns in header
  if (job.config?.models) {
    const monCreative = document.getElementById("mon-select-creative-model");
    const monMech = document.getElementById("mon-select-mechanical-model");
    if (monCreative && job.config.models.creative_model) {
      monCreative.value = job.config.models.creative_model;
      updateQuotaBadge(monCreative, document.getElementById("mon-badge-creative-quota"));
    }
    if (monMech && job.config.models.mechanical_model) {
      monMech.value = job.config.models.mechanical_model;
      updateQuotaBadge(monMech, document.getElementById("mon-badge-mechanical-quota"));
    }
  }

  // Elapsed timer
  updateJobTimer(job);

  // Stage rail
  renderStageRail(job.stages || {}, job.current_stage);

  // Gates & Deliverables
  renderGateA(job);
  renderGateB(job);
  renderDeliveryDeck(job);

  // Antigravity Co-Pilot Workspace
  if (job.copilot) {
    renderCopilotUI(job.copilot);
  }

  // HyperFrames studio badge
  const studioPill = document.getElementById("studio-status-pill");
  const btnStopStudio = document.getElementById("btn-stop-studio");
  if (studioPill) {
    const isOpen = job.studio && job.studio.open;
    studioPill.textContent = isOpen ? `Studio Active (: ${job.studio.port})` : "Studio Inactive";
    studioPill.className = `studio-pill ${isOpen ? 'active' : 'inactive'}`;
    if (btnStopStudio) {
      btnStopStudio.classList.toggle("hidden", !isOpen);
    }
  }

  // Historical logs
  if (job.logs && Array.isArray(job.logs)) {
    job.logs.forEach(msg => appendConsoleLog(msg, "log-info"));
  }
}

function updateJobTimer(job) {
  clearInterval(state.jobTimerInterval);
  const timeElem = document.getElementById("mon-elapsed-time");
  if (!timeElem) return;

  const startTime = job.created_at ? new Date(job.created_at).getTime() : Date.now();

  function tick() {
    const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
    const m = Math.floor(elapsedSec / 60);
    const s = elapsedSec % 60;
    timeElem.textContent = `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  }

  tick();
  if (job.status === "running" || job.status === "pending") {
    state.jobTimerInterval = setInterval(tick, 1000);
  }
}

function renderStageRail(stages = {}, currentStage = null) {
  const rail = document.getElementById("stage-rail");
  if (!rail) return;
  rail.innerHTML = "";

  STAGE_ORDER.forEach((stageName, idx) => {
    const info = stages[stageName] || { status: "pending" };
    const rawStatus = info.status || "pending";
    const status = rawStatus === "completed" ? "done" : rawStatus;

    const stageItem = document.createElement("div");
    stageItem.className = `stage-item ${status}`;
    stageItem.title = `Stage: ${stageName} (${rawStatus})`;

    let iconText = (idx + 1).toString();
    if (status === "done") iconText = "✓";
    else if (status === "failed" || status === "cancelled") iconText = "✗";
    else if (status === "waiting_gate") iconText = "⏸";

    stageItem.innerHTML = `
      <div class="stage-circle">${iconText}</div>
      <div class="stage-name">${stageName}</div>
    `;

    rail.appendChild(stageItem);

    if (idx < STAGE_ORDER.length - 1) {
      const connector = document.createElement("div");
      connector.className = `stage-connector ${status === "done" ? "done" : ""}`;
      rail.appendChild(connector);
    }
  });
}

// -----------------------------------------------------------------------------
// Gate A: Script Approval
// -----------------------------------------------------------------------------

function renderGateA(job) {
  const panel = document.getElementById("panel-gate-a");
  if (!panel) return;

  const isGateA = job.status === "awaiting_script_approval" ||
                  (job.stages?.script?.status === "waiting_gate" || job.stages?.script?.status === "waiting_approval") ||
                  (job.pause_at_script && job.current_stage === "script" && job.stages?.script?.status === "completed") ||
                  job.awaiting_script_approval === true;

  if (!isGateA) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  // Options A, B, C
  const options = Array.isArray(job.script_options) ? job.script_options : [];

  const grid = document.getElementById("gate-a-options");
  if (grid) {
    grid.innerHTML = "";
    if (options.length === 0) {
      grid.innerHTML = '<div class="text-muted p-4 text-xs">Script options are currently generating or awaiting stage completion...</div>';
    } else {
      options.forEach(opt => {
        const isSelected = opt.id === state.selectedScriptOption;
        const card = document.createElement("div");
        card.className = `gate-option-card ${isSelected ? "selected" : ""}`;
        card.innerHTML = `
          <div class="option-badge">Draft ${opt.id}</div>
          <div class="option-thesis">${escapeHtml(opt.thesis || "")}</div>
          <div class="option-meta">
            <span>Est: ~${opt.estimated_duration_s || opt.est_duration_s || "—"}s</span>
            <span class="text-success">Opus Draft</span>
          </div>
          <div class="option-critic"><strong>Critic:</strong> ${escapeHtml(opt.critic_notes || opt.viewer_question || "Passes retention standards.")}</div>
        `;
        card.addEventListener("click", () => {
          state.selectedScriptOption = opt.id;
          renderGateA(job);
          const editor = document.getElementById("gate-a-script-editor");
          if (editor) editor.value = opt.script_text || "";
        });
        grid.appendChild(card);
      });
    }
  }

  const editor = document.getElementById("gate-a-script-editor");
  if (editor && options.length > 0 && !editor.value) {
    const activeOpt = options.find(o => o.id === state.selectedScriptOption) || options[0];
    editor.value = activeOpt?.script_text || "";
  }
}

// -----------------------------------------------------------------------------
// Gate B: Audio & Base Cut Approval
// -----------------------------------------------------------------------------

function renderGateB(job) {
  const panel = document.getElementById("panel-gate-b");
  if (!panel) return;

  const isGateB = job.status === "awaiting_audio_approval" ||
                  (job.stages?.audio?.status === "waiting_gate" || job.stages?.audio?.status === "waiting_approval") ||
                  (job.stages?.base_cut?.status === "waiting_gate" || job.stages?.base_cut?.status === "waiting_approval") ||
                  (job.stages?.bgm_broll?.status === "waiting_gate" || job.stages?.bgm_broll?.status === "waiting_approval") ||
                  (job.pause_at_audio && (job.current_stage === "base_cut" || job.current_stage === "bgm_broll") && job.stages?.[job.current_stage]?.status === "completed") ||
                  job.awaiting_audio_approval === true;

  if (!isGateB) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  // Video Player for Base Cut
  const videoPlayer = document.getElementById("base-video-player");
  if (videoPlayer && job.base_video_url) {
    if (videoPlayer.getAttribute("src") !== job.base_video_url) {
      videoPlayer.src = job.base_video_url;
    }
  }

  // Base QA Metrics
  const qa = job.qa || job.base_qa || {};
  document.getElementById("metric-base-dur").textContent = qa.duration_s != null ? `${qa.duration_s}s` : "—";
  document.getElementById("metric-base-cuts").textContent = qa.parts_count != null ? `${qa.parts_count}` : (qa.join_count != null ? `${qa.join_count}` : "—");
  document.getElementById("metric-base-stutters").textContent = qa.stutters_cut != null ? `${qa.stutters_cut}` : "—";
  document.getElementById("metric-base-lufs").textContent = qa.integrated_loudness_lufs != null ? `${qa.integrated_loudness_lufs} LUFS` : (qa.lufs != null ? `${qa.lufs} LUFS` : "—");

  // BGM Candidates List
  const bgmList = document.getElementById("bgm-candidates-list");
  if (bgmList) {
    const candidates = Array.isArray(job.bgm_candidates) ? job.bgm_candidates : [];

    bgmList.innerHTML = "";
    if (candidates.length === 0) {
      bgmList.innerHTML = '<div class="text-muted p-3 text-xs">No BGM tracks generated yet.</div>';
    } else {
      candidates.forEach((bgm, idx) => {
        const isSelected = state.selectedBgm === bgm.filename || (state.selectedBgm === null && idx === 0);
        if (isSelected && state.selectedBgm === null) state.selectedBgm = bgm.filename;

        const item = document.createElement("div");
        item.className = `bgm-item ${isSelected ? "active" : ""}`;
        item.innerHTML = `
          <div class="bgm-left">
            <input type="radio" name="bgm-radio" ${isSelected ? "checked" : ""}>
            <span class="bgm-title">${escapeHtml(bgm.title || bgm.name || "")}</span>
          </div>
          ${bgm.url ? `<audio controls src="${bgm.url}" style="height: 28px; width: 180px;"></audio>` : '<span class="text-muted text-xs">Preview</span>'}
        `;

        item.addEventListener("click", () => {
          state.selectedBgm = bgm.filename;
          bgmList.querySelectorAll(".bgm-item").forEach(i => i.classList.remove("active"));
          item.classList.add("active");
          item.querySelector("input").checked = true;
        });
        bgmList.appendChild(item);
      });
    }
  }
}

// -----------------------------------------------------------------------------
// Delivery Deck (Final Video, Covers, 5 Descriptions, Studio Bridge)
// -----------------------------------------------------------------------------

function renderDeliveryDeck(job) {
  const panel = document.getElementById("panel-delivery-deck");
  if (!panel) return;

  const delivStatus = job.stages?.deliverables?.status;
  const isDelivered = (job.final_video_url !== null && job.final_video_url !== undefined) ||
                      job.status === "completed" ||
                      delivStatus === "done" ||
                      delivStatus === "completed";

  if (!isDelivered) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  // Video Player
  const finalPlayer = document.getElementById("final-video-player");
  const downloadBtn = document.getElementById("btn-download-final");
  if (finalPlayer && job.final_video_url) {
    if (finalPlayer.getAttribute("src") !== job.final_video_url) {
      finalPlayer.src = job.final_video_url;
    }
  }
  if (downloadBtn && job.final_video_url) {
    downloadBtn.href = job.final_video_url;
  }

  // QA Scorecard
  const qa = job.qa || {};
  document.getElementById("qa-script-match").textContent = qa.script_match_pct != null ? `${qa.script_match_pct}%` : (qa.script_match != null ? `${qa.script_match}%` : "—");
  document.getElementById("qa-duration").textContent = qa.duration_s != null ? `${qa.duration_s}s` : "—";
  document.getElementById("qa-lufs").textContent = qa.integrated_loudness_lufs != null ? `${qa.integrated_loudness_lufs}` : (qa.lufs != null ? `${qa.lufs}` : "—");
  document.getElementById("qa-cuts").textContent = qa.parts_count != null ? `${qa.parts_count}` : (qa.cuts_count != null ? `${qa.cuts_count}` : "—");

  // Covers A & B
  if (job.covers && job.covers.length >= 2) {
    const cA = job.covers[0];
    const cB = job.covers[1];
    const imgA = document.getElementById("img-cover-a");
    const linkA = document.getElementById("link-cover-a");
    const imgB = document.getElementById("img-cover-b");
    const linkB = document.getElementById("link-cover-b");

    if (imgA) imgA.src = cA.url;
    if (linkA) linkA.href = cA.url;
    if (imgB) imgB.src = cB.url;
    if (linkB) linkB.href = cB.url;
  }

  // Descriptions Data
  state.descriptionsData = job.descriptions_data || {};

  renderDescriptionText();
}

function initDescriptionsTabs() {
  const tabs = document.querySelectorAll("#desc-tabs .desc-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      state.activeDescTab = tab.dataset.descType;
      renderDescriptionText();
    });
  });

  const btnCopy = document.getElementById("btn-copy-desc");
  if (btnCopy) {
    btnCopy.addEventListener("click", () => {
      const text = document.getElementById("desc-text-body")?.textContent || "";
      if (text) {
        navigator.clipboard.writeText(text).then(() => {
          showToast("Description copied to clipboard!", "success");
        }).catch(() => {
          showToast("Failed to copy to clipboard", "error");
        });
      }
    });
  }
}

function renderDescriptionText() {
  const body = document.getElementById("desc-text-body");
  const recBadge = document.getElementById("desc-rec-badge");
  if (!body || !state.descriptionsData) return;

  const text = state.descriptionsData[state.activeDescTab] || "No copy generated for this category.";
  body.textContent = text;

  if (recBadge) {
    const recText = (state.descriptionsData.recommendation || "").toLowerCase();
    const tabKey = state.activeDescTab.toLowerCase();
    const isRecommended = recText ? recText.includes(tabKey) : state.activeDescTab === "hook";
    recBadge.classList.toggle("hidden", !isRecommended);
    recBadge.title = state.descriptionsData.recommendation || "";
  }
}

// -----------------------------------------------------------------------------
// Job Interactive Buttons (Cancel, Retry, Gates, HyperFrames)
// -----------------------------------------------------------------------------

function initJobControls() {
  // Cancel Job
  document.getElementById("btn-cancel-job")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    if (!confirm("Are you sure you want to cancel this job?")) return;
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
      if (res.ok) {
        showToast("Job cancelled", "info");
      }
    } catch (err) {
      showToast(`Cancel failed: ${err.message}`, "error");
    }
  });

  // Retry from stage
  document.getElementById("btn-retry-stage")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    const stage = document.getElementById("select-retry-stage")?.value || "ingest";
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from_stage: stage, force: true })
      });
      if (res.ok) {
        showToast(`Retrying from stage: ${stage}`, "success");
      }
    } catch (err) {
      showToast(`Retry failed: ${err.message}`, "error");
    }
  });

  // Gate A: Approve Script
  document.getElementById("btn-approve-gate-a")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    const text = document.getElementById("gate-a-script-editor")?.value || "";
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/approve-script`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ option_id: state.selectedScriptOption, script_text: text })
      });
      if (res.ok) {
        showToast("Script approved! Building base cut...", "success");
        document.getElementById("panel-gate-a")?.classList.add("hidden");
      }
    } catch (err) {
      showToast(`Approval failed: ${err.message}`, "error");
    }
  });

  // Gate B: Approve Audio
  document.getElementById("btn-approve-gate-b")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/approve-audio`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bgm_file: state.selectedBgm })
      });
      if (res.ok) {
        showToast("Base audio approved! Generating motion graphics...", "success");
        document.getElementById("panel-gate-b")?.classList.add("hidden");
      }
    } catch (err) {
      showToast(`Approval failed: ${err.message}`, "error");
    }
  });

  // HyperFrames Studio Bridge
  document.getElementById("btn-open-studio")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/studio/open`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        showToast(`Opened HyperFrames Studio on port ${data.port}`, "success");
        if (data.url) {
          window.open(data.url, "_blank");
        }
      }
    } catch (err) {
      showToast(`Studio open error: ${err.message}`, "error");
    }
  });

  document.getElementById("btn-recomposite")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/recomposite`, { method: "POST" });
      if (res.ok) {
        showToast("Triggered HyperFrames re-composition!", "success");
      }
    } catch (err) {
      showToast(`Recomposite error: ${err.message}`, "error");
    }
  });

  document.getElementById("btn-stop-studio")?.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/studio/stop`, { method: "POST" });
      if (res.ok) {
        showToast("HyperFrames Studio stopped", "info");
      }
    } catch (err) {
      showToast(`Studio stop error: ${err.message}`, "error");
    }
  });
}

// -----------------------------------------------------------------------------
// Live Monospace SSE Console Controls
// -----------------------------------------------------------------------------

let logCounter = 0;

function initConsoleControls() {
  document.getElementById("btn-clear-console")?.addEventListener("click", clearConsoleLogs);

  const btnToggle = document.getElementById("btn-toggle-console");
  const consoleBody = document.getElementById("console-body");
  if (btnToggle && consoleBody) {
    btnToggle.addEventListener("click", () => {
      const isMin = consoleBody.classList.toggle("minimized");
      btnToggle.textContent = isMin ? "Expand" : "Minimize";
    });
  }
}

function clearConsoleLogs() {
  const body = document.getElementById("console-body");
  if (body) body.innerHTML = "";
  logCounter = 0;
  updateLogCounter();
}

function appendConsoleLog(message, className = "log-info") {
  const body = document.getElementById("console-body");
  if (!body) return;

  const line = document.createElement("div");
  line.className = `log-line ${className}`;
  const now = new Date().toLocaleTimeString();
  line.textContent = `[${now}] ${message}`;
  body.appendChild(line);

  logCounter++;
  updateLogCounter();

  const autoscroll = document.getElementById("toggle-autoscroll")?.checked;
  if (autoscroll) {
    body.scrollTop = body.scrollHeight;
  }
}

function updateLogCounter() {
  const elem = document.getElementById("log-count");
  if (elem) elem.textContent = `${logCounter} event${logCounter === 1 ? '' : 's'}`;
}

// -----------------------------------------------------------------------------
// Antigravity Co-Pilot Workspace (agy --dangerously-skip-permissions)
// -----------------------------------------------------------------------------

function initCopilot() {
  const btnSend = document.getElementById("btn-send-copilot");
  const input = document.getElementById("copilot-input");
  const btnReset = document.getElementById("btn-reset-copilot");
  const btnCollapse = document.getElementById("btn-collapse-copilot");
  const btnToggleHeader = document.getElementById("btn-toggle-copilot-header");

  // Send message on click
  btnSend?.addEventListener("click", () => sendCopilotMessage());

  // Enter to send, Shift+Enter for newline
  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendCopilotMessage();
    }
  });

  // Reset conversation session
  btnReset?.addEventListener("click", () => resetCopilotSession());

  // Collapse / expand panel
  btnCollapse?.addEventListener("click", () => toggleCopilotPanel(false));
  btnToggleHeader?.addEventListener("click", () => toggleCopilotPanel());

  // Quick Action Chips
  const chipContainer = document.getElementById("copilot-quick-chips");
  chipContainer?.addEventListener("click", (e) => {
    const btn = e.target.closest(".copilot-chip");
    if (!btn) return;
    const prompt = btn.dataset.prompt;
    if (prompt && input) {
      input.value = prompt;
      input.focus();
    }
  });
}

function toggleCopilotPanel(explicitState = null) {
  const panel = document.getElementById("job-copilot-panel");
  const split = document.getElementById("job-workspace-split");
  if (!panel || !split) return;

  if (explicitState !== null) {
    state.copilotCollapsed = !explicitState;
  } else {
    state.copilotCollapsed = !state.copilotCollapsed;
  }

  panel.classList.toggle("collapsed", state.copilotCollapsed);
  split.classList.toggle("copilot-collapsed", state.copilotCollapsed);

  const toggleBtn = document.getElementById("btn-toggle-copilot-header");
  if (toggleBtn) {
    toggleBtn.classList.toggle("active", !state.copilotCollapsed);
  }
}

function renderCopilotUI(copilot) {
  if (!copilot) return;

  state.copilotSessionId = copilot.conversation_id || null;
  const badge = document.getElementById("copilot-session-badge");
  if (badge) {
    const shortId = state.copilotSessionId ? state.copilotSessionId.slice(0, 8) : "New";
    badge.textContent = `Session: ${shortId}`;
    badge.title = state.copilotSessionId ? `Session ID: ${state.copilotSessionId}` : "New Session";
  }

  if (Array.isArray(copilot.messages)) {
    state.copilotMessages = copilot.messages;
    renderCopilotTimeline();
  }

  setCopilotBusy(copilot.busy === true);
}

function renderCopilotTimeline() {
  const timeline = document.getElementById("copilot-timeline");
  if (!timeline) return;

  timeline.innerHTML = "";

  if (state.copilotMessages.length === 0) {
    const welcome = document.createElement("div");
    welcome.className = "copilot-welcome-card";
    welcome.id = "copilot-welcome-card";
    welcome.innerHTML = `
      <div class="welcome-icon">⚡</div>
      <div class="welcome-title">Zero Window-Switching Workspace</div>
      <p class="welcome-text">
        Ask Antigravity to adjust EDL cut points, pitch alternate hooks, edit motion graphics scripts, inspect QA metrics, or rewrite viral social copy. Edits apply directly to your active job in real-time.
      </p>
    `;
    timeline.appendChild(welcome);
    return;
  }

  state.copilotMessages.forEach(msg => {
    appendCopilotMessage(msg, false);
  });

  scrollCopilotToBottom();
}

function appendCopilotMessage(msg, shouldScroll = true) {
  const timeline = document.getElementById("copilot-timeline");
  if (!timeline) return;

  // Remove welcome card once there are messages
  const welcome = document.getElementById("copilot-welcome-card");
  if (welcome) welcome.remove();

  const msgElem = document.createElement("div");

  if (msg.role === "user") {
    msgElem.className = "copilot-msg user";
    const timeStr = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    msgElem.innerHTML = `
      <div class="msg-bubble">${escapeHtml(msg.content || "")}</div>
      <div class="msg-meta">You · ${timeStr}</div>
    `;
  } else {
    msgElem.className = "copilot-msg assistant";
    const metaParts = [];
    if (msg.model) metaParts.push(msg.model);
    if (msg.duration_seconds != null && !isNaN(msg.duration_seconds)) metaParts.push(`${Number(msg.duration_seconds).toFixed(1)}s`);
    const totalTokens = msg.usage?.total_tokens || (msg.usage?.prompt_tokens && msg.usage?.completion_tokens ? msg.usage.prompt_tokens + msg.usage.completion_tokens : null);
    if (totalTokens) metaParts.push(`${totalTokens} tok`);

    const metaPillText = metaParts.join(" · ") || "agy";

    let filesHtml = "";
    if (Array.isArray(msg.modified_files) && msg.modified_files.length > 0) {
      const chips = msg.modified_files.map(f => {
        const basename = String(f).split("/").pop();
        return `<span class="file-chip" title="${escapeHtml(String(f))}">📄 ${escapeHtml(basename)}</span>`;
      }).join("");
      filesHtml = `
        <div class="msg-modified-files">
          <span class="files-label">Files Modified:</span>
          <div class="file-chips-row">${chips}</div>
        </div>
      `;
    }

    let actionHtml = "";
    if (msg.suggested_action) {
      let label = "Execute Suggested Action";
      if (typeof msg.suggested_action === "string") {
        label = msg.suggested_action;
      } else if (typeof msg.suggested_action === "object") {
        label = msg.suggested_action.label || msg.suggested_action.name || msg.suggested_action.action || "Execute Action";
      }
      actionHtml = `
        <div class="msg-suggested-action">
          <button type="button" class="btn-suggested-action">
            <span>⚡</span> ${escapeHtml(label)}
          </button>
        </div>
      `;
    }

    msgElem.innerHTML = `
      <div class="msg-card">
        <div class="msg-header">
          <div class="copilot-author-group">
            <span class="copilot-avatar">🤖</span>
            <span class="copilot-author-name">Antigravity</span>
          </div>
          <span class="copilot-meta-pill">${escapeHtml(metaPillText)}</span>
        </div>
        <div class="msg-body">${formatMarkdown(msg.content || "")}</div>
        ${filesHtml}
        ${actionHtml}
      </div>
    `;

    if (msg.suggested_action) {
      const actionBtn = msgElem.querySelector(".btn-suggested-action");
      actionBtn?.addEventListener("click", () => executeSuggestedAction(msg.suggested_action));
    }
  }

  timeline.appendChild(msgElem);

  if (shouldScroll) {
    scrollCopilotToBottom();
  }
}

function scrollCopilotToBottom() {
  const timeline = document.getElementById("copilot-timeline");
  if (timeline) {
    timeline.scrollTop = timeline.scrollHeight;
  }
}

function _formatElapsed(startMs) {
  const secs = Math.max(0, Math.floor((Date.now() - startMs) / 1000));
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

function setCopilotBusy(isBusy) {
  const wasBusy = state.copilotBusy;
  state.copilotBusy = isBusy;
  const spinner = document.getElementById("copilot-spinner");
  const btnSend = document.getElementById("btn-send-copilot");
  const timeline = document.getElementById("copilot-timeline");

  if (spinner) spinner.classList.toggle("hidden", !isBusy);
  if (btnSend) btnSend.disabled = isBusy;

  // Manage live typing indicator bubble in timeline
  let typingElem = document.getElementById("copilot-typing-indicator");
  if (isBusy) {
    if (!wasBusy || !state.copilotStartTs) {
      state.copilotStartTs = Date.now();
    }
    if (!typingElem && timeline) {
      typingElem = document.createElement("div");
      typingElem.id = "copilot-typing-indicator";
      typingElem.className = "copilot-typing";
      typingElem.innerHTML = `
        <div class="copilot-typing-head">
          <div class="copilot-typing-dots">
            <span class="copilot-typing-dot"></span>
            <span class="copilot-typing-dot"></span>
            <span class="copilot-typing-dot"></span>
          </div>
          <span class="copilot-typing-text" id="copilot-typing-status">Antigravity is reasoning...</span>
          <span class="copilot-typing-elapsed" id="copilot-typing-elapsed">0s</span>
        </div>
        <div class="copilot-typing-preview hidden" id="copilot-typing-preview"></div>
      `;
      timeline.appendChild(typingElem);
      scrollCopilotToBottom();
    }
    const elapsedEl = document.getElementById("copilot-typing-elapsed");
    if (elapsedEl) elapsedEl.textContent = _formatElapsed(state.copilotStartTs);

    if (!state.copilotElapsedTimer) {
      state.copilotElapsedTimer = setInterval(() => {
        const el = document.getElementById("copilot-typing-elapsed");
        if (el && state.copilotStartTs) {
          el.textContent = _formatElapsed(state.copilotStartTs);
        }
      }, 1000);
    }
  } else {
    state.copilotStartTs = null;
    if (state.copilotElapsedTimer) {
      clearInterval(state.copilotElapsedTimer);
      state.copilotElapsedTimer = null;
    }
    if (typingElem) typingElem.remove();
  }
}

function updateCopilotProgress(event) {
  if (!event || typeof event !== "object") return;
  if (!state.copilotBusy) {
    setCopilotBusy(true);
  }
  const statusEl = document.getElementById("copilot-typing-status");
  const previewEl = document.getElementById("copilot-typing-preview");
  if (event.text && statusEl) {
    statusEl.textContent = event.text;
  }
  if (event.kind === "delta" && event.preview) {
    if (statusEl) statusEl.textContent = "Streaming response...";
    if (previewEl) {
      previewEl.textContent = event.preview;
      previewEl.classList.remove("hidden");
    }
  }
  scrollCopilotToBottom();
}

async function sendCopilotMessage() {
  if (!state.currentJobId) {
    showToast("Open or select an active job first before messaging Co-Pilot.", "error");
    return;
  }

  const input = document.getElementById("copilot-input");
  if (!input) return;
  const messageText = input.value.trim();
  if (!messageText) return;

  const modelSelect = document.getElementById("copilot-model-select");
  const selectedModel = modelSelect?.value || state.defaultModels.copilot_model;
  const currentStage = state.currentJob?.current_stage || null;

  // Optimistically add user bubble to timeline and clear input
  input.value = "";
  appendCopilotMessage({
    role: "user",
    content: messageText,
    timestamp: new Date().toISOString()
  });

  setCopilotBusy(true);

  try {
    const res = await fetch(`/api/jobs/${state.currentJobId}/copilot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: messageText,
        model: selectedModel,
        stage_hint: currentStage
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Co-Pilot request failed");
    }

    const data = await res.json();

    // If enriched job is returned, immediately refresh UI to reflect live edits
    if (data.job) {
      updateJobUI(data.job);
    } else if (data.copilot) {
      renderCopilotUI(data.copilot);
    } else if (data.message) {
      const msgObj = (typeof data.message === "object" && data.message !== null)
        ? data.message
        : {
            role: "assistant",
            content: data.message,
            model: selectedModel,
            modified_files: data.modified_files,
            suggested_action: data.suggested_action,
            timestamp: new Date().toISOString()
          };
      appendCopilotMessage(msgObj);
    }

    setCopilotBusy(false);
  } catch (err) {
    setCopilotBusy(false);
    showToast(`Co-Pilot error: ${err.message}`, "error");
    appendCopilotMessage({
      role: "assistant",
      content: `⚠️ Failed to get response from Antigravity: ${err.message}`,
      model: selectedModel,
      timestamp: new Date().toISOString()
    });
  }
}

async function resetCopilotSession() {
  if (!state.currentJobId) return;
  if (!confirm("Reset Antigravity Co-Pilot conversation session? This clears chat history for this job.")) return;

  try {
    const res = await fetch(`/api/jobs/${state.currentJobId}/copilot/reset`, {
      method: "POST"
    });
    if (res.ok) {
      state.copilotMessages = [];
      state.copilotSessionId = null;
      renderCopilotTimeline();
      const badge = document.getElementById("copilot-session-badge");
      if (badge) badge.textContent = "Session: New";
      showToast("Co-Pilot conversation session reset", "info");
    } else {
      const err = await res.json().catch(() => ({}));
      showToast(`Reset failed: ${err.detail || "Error"}`, "error");
    }
  } catch (err) {
    showToast(`Reset failed: ${err.message}`, "error");
  }
}

async function executeSuggestedAction(action) {
  if (!state.currentJobId) return;

  let label = "";
  let actionType = "start_stage";
  let targetStage = "base_cut";
  let targetGate = null;

  if (typeof action === "string") {
    label = action;
    const lower = action.toLowerCase();
    if (lower.includes("recomposite") || lower.includes("composite") || lower.includes("re-render")) {
      actionType = "recomposite";
    } else if (lower.includes("gate") || lower.includes("refresh")) {
      actionType = "refresh_gate";
      targetGate = lower.includes("gate b") || lower.includes("audio") ? "B" : "A";
    } else if (lower.includes("base_cut") || lower.includes("base cut")) {
      actionType = "start_stage";
      targetStage = "base_cut";
    } else if (lower.includes("script")) {
      actionType = "start_stage";
      targetStage = "script";
    } else if (lower.includes("edl")) {
      actionType = "start_stage";
      targetStage = "edl";
    } else if (lower.includes("graphics")) {
      actionType = "start_stage";
      targetStage = "graphics";
    } else if (lower.includes("deliverables") || lower.includes("render")) {
      actionType = "start_stage";
      targetStage = "deliverables";
    } else if (STAGE_ORDER.includes(action)) {
      actionType = "start_stage";
      targetStage = action;
    }
  } else if (typeof action === "object" && action !== null) {
    label = action.label || action.name || action.type || action.action || "Execute Action";
    actionType = action.type || action.action || (action.stage || action.from_stage ? "start_stage" : "recomposite");
    targetStage = action.from_stage || action.stage || "base_cut";
    targetGate = action.gate || null;
  }

  try {
    if (actionType === "refresh_gate") {
      showToast(`Refreshing ${label || "Gate"}...`, "info");
      const res = await fetch(`/api/jobs/${state.currentJobId}`);
      if (res.ok) {
        const job = await res.json();
        updateJobUI(job);
        const gateId = targetGate === "B" ? "panel-gate-b" : "panel-gate-a";
        const gatePanel = document.getElementById(gateId);
        gatePanel?.scrollIntoView({ behavior: "smooth", block: "nearest" });
        showToast("Updated UI from latest job artifacts!", "success");
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(`Refresh failed: ${err.detail || "Error"}`, "error");
      }
    } else if (actionType === "recomposite") {
      showToast(`Triggering: ${label || "Re-render Studio & Composite"}...`, "info");
      const res = await fetch(`/api/jobs/${state.currentJobId}/recomposite`, { method: "POST" });
      if (res.ok) {
        const job = await res.json().catch(() => null);
        if (job && job.id) updateJobUI(job);
        showToast("Re-composition completed!", "success");
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(`Failed: ${err.detail || "Error"}`, "error");
      }
    } else {
      showToast(`Triggering: ${label || ("Retry " + targetStage)}...`, "info");
      const res = await fetch(`/api/jobs/${state.currentJobId}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from_stage: targetStage, force: true })
      });
      if (res.ok) {
        showToast(`Pipeline restarted from stage: ${targetStage}`, "success");
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(`Failed: ${err.detail || "Error"}`, "error");
      }
    }
  } catch (err) {
    showToast(`Action failed: ${err.message}`, "error");
  }
}

function formatMarkdown(text) {
  if (!text) return "";
  // 1. Escape HTML
  let str = escapeHtml(text);

  // 2. Code blocks (```lang ... ```)
  str = str.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // 3. Inline code (`code`)
  str = str.replace(/`([^`]+)`/g, '<code>$1</code>');

  // 4. Bold (**text**)
  str = str.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // 5. Italics (*text*)
  str = str.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // 6. Bullet lists
  str = str.replace(/(?:^|\n)[ \t]*[-*][ \t]+([^\n]+)/g, '\n<li>$1</li>');
  str = str.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
  str = str.replace(/<\/ul>\s*<ul>/g, '');

  // 7. Paragraphs & Line breaks
  str = str.replace(/\n\n+/g, '</p><p>');
  str = str.replace(/\n/g, '<br>');

  return `<p>${str}</p>`;
}

