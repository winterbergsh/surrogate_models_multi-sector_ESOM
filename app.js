const VARIABLE_GROUPS = {
  "Electricity system capacity": [
    "Electricity capacity_Carrier_sum",
    "Electricity capacity_Carrier_share",
    "Electricity capacity_Dispatch_sum",
    "Electricity capacity_Dispatch_share",
    "Electricity capacity_Fossil_sum",
    "Electricity capacity_Fossil_share",
    "Electricity capacity_Renewable_sum",
    "Electricity capacity_Renewable_share",
  ],
  "Electricity system generation": [
    "Electricity generation_Carrier_sum",
    "Electricity generation_Carrier_share",
    "Electricity generation_Dispatch_sum",
    "Electricity generation_Dispatch_share",
    "Electricity generation_Fossil_sum",
    "Electricity generation_Fossil_share",
    "Electricity generation_Renewable_sum",
    "Electricity generation_Renewable_share",
  ],
  "Energy carriers production": ["BioFuel Production", "Hydrogen Production", "Methanol Production", "SynFuel Production"],
  "Energy carriers imports": ["BioEnergy Imports", "BioFuel Imports", "Biomass Imports", "Hydrogen Imports", "Imported Energy"],
  "Flexibility and storage": [
    "Flexibility Capacity",
    "DR Flexibility Capacity",
    "Storage Flexibility Capacity",
    "Shedding Flexibility Capacity",
    "Hourly Flexibility Capacity",
    "Daily Flexibility Capacity",
    "Weekly Flexibility Capacity",
  ],
  Emissions: ["CO2 Price", "CO2 Storage", "CO2_Price"],
  "Commodity prices": ["comPrices"],
  "Total system costs": ["totalCosts"],
  "Technology operation": ["techUseNet"],
  "Technology capacity": ["techStocks"],
};

const MODEL_ALIASES = {
  XGB_single: "XGB.py",
  AE_MLP: "MLP.py",
  AE_GaussianProcess: "GP.py",
  XGB_HYBRID_tuned: "XGB+classifier.py",
  Informed_XGB_HYBRID: "xgb_informed_classifier.py",
};

const MODEL_DISPLAY_NAMES = {
  XGB_single: "XGBoost (Single Target)",
  AE_MLP: "MLP Autoencoder",
  AE_GaussianProcess: "Gaussian Process Autoencoder",
  XGB_HYBRID_tuned: "XGBoost + Classifier (Hybrid)",
  XGB_HYBRID10_tuned: "XGBoost + Classifier (Hybrid, threshold 10)",
  XGB_HYBRID20_tuned: "XGBoost + Classifier (Hybrid, threshold 20)",
  XGB_HYBRID30_tuned: "XGBoost + Classifier (Hybrid, threshold 30)",
  XGB_HYBRID40_tuned: "XGBoost + Classifier (Hybrid, threshold 40)",
  Informed_XGB_HYBRID: "Informed XGBoost + Classifier",
};

let regressionSummaryData = [];
let jointSummaryData = [];
let currentClassifierRows = [];
let dashboardSourceRows = {
  regression: [],
  classifier: [],
  joint: [],
};
let commonTargetMode = "all";
let recurringOnlyMode = "all";

const trainingDataState = {
  targets: [],
  selected: new Set(),
  maxPanels: 9,
};

const TRAINING_VARIABLE_TYPE_REFERENCE_ROWS = [
  { type: "Technology capacity", count: 218 },
  { type: "Technology operation", count: 200 },
  { type: "Commodity prices", count: 81 },
  { type: "Electricity system capacity", count: 22 },
  { type: "Electricity system generation", count: 22 },
  { type: "Flexibility and storage", count: 7 },
  { type: "Energy carriers imports", count: 5 },
  { type: "Energy carriers production", count: 4 },
  { type: "Emissions", count: 2 },
  { type: "Total system cost (objective function)", count: 1 },
];

const datasheetState = {
  datasets: {
    regression: [],
    classifier: [],
    joint: [],
  },
  activeKey: "regression",
  methodFilter: ["all"],
  search: "",
  page: 1,
  pageSize: 50,
};

const DATASET_FILTER_OPTIONS = {
  regression: [
    { value: "all", label: "All datasets" },
    { value: "xgb", label: "XGB" },
    { value: "mlp", label: "MLP" },
    { value: "gp", label: "GP" },
    { value: "xgb5", label: "XGB 5-fold" },
    { value: "xgb6", label: "XGB 6-fold" },
  ],
  classifier: [
    { value: "all", label: "All datasets" },
    { value: "cr10", label: "Classifier+Regressor 10" },
    { value: "cr20", label: "Classifier+Regressor 20" },
    { value: "cr30", label: "Classifier+Regressor 30" },
    { value: "cr40", label: "Classifier+Regressor 40" },
  ],
  joint: [
    { value: "all", label: "All datasets" },
    { value: "informed", label: "Informed XGB Hybrid" },
  ],
};

const PLOT_FONT = {
  family: "Source Sans 3, sans-serif",
  size: 13,
  color: "#222222",
};

function parseCSV(text) {
  return Papa.parse(text, { header: true, dynamicTyping: false, skipEmptyLines: true }).data;
}

async function loadCSV(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  const text = await response.text();
  return parseCSV(text);
}

async function loadJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

function toNum(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (!s || s === "nan" || s === "null" || s === "none" || s === "na") {
      return null;
    }
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function median(values) {
  const nums = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!nums.length) return null;
  const m = Math.floor(nums.length / 2);
  return nums.length % 2 === 0 ? (nums[m - 1] + nums[m]) / 2 : nums[m];
}

function mean(values) {
  const nums = values.filter((v) => Number.isFinite(v));
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function fmt(v, d = 3) {
  return Number.isFinite(v) ? v.toFixed(d) : "-";
}

function normalizeTargetName(target) {
  return String(target ?? "")
    .toLowerCase()
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function averageMetric(rows, key, scale = 1) {
  const values = rows.map((r) => toNum(r[key])).filter((v) => v !== null);
  if (!values.length) return null;
  return mean(values) * scale;
}

function getMethodRowsMap(regRows, clsRows, jointRows) {
  const clsThreshold20Rows = clsRows.filter((r) => {
    const method = String(r.surrogate_method || "").toLowerCase();
    if (method === "xgb_hybrid20_tuned" || method === "xgb_hybrid_tuned") {
      return true;
    }

    // Fallback: some rows can carry threshold in hyperparameters text.
    const hp = String(r.hyperparameters || "").toLowerCase();
    return method.includes("xgb_hybrid") && hp.includes("categories_threshold") && /categories_threshold['\"]?\s*:\s*20/.test(hp);
  });

  return [
    {
      label: "XGB",
      rows: regRows.filter((r) => r.surrogate_method === "XGB_single"),
    },
    {
      label: "MLP",
      rows: regRows.filter((r) => r.surrogate_method === "AE_MLP"),
    },
    {
      label: "GP",
      rows: regRows.filter((r) => r.surrogate_method === "AE_GaussianProcess"),
    },
    {
      label: "XGB+classifier",
      rows: clsThreshold20Rows,
    },
    {
      label: "informed_classifier",
      rows: jointRows.filter((r) => String(r.surrogate_method || "") === "Informed_XGB_HYBRID"),
    },
  ];
}

function getCommonTargetSet(methodMap) {
  const sets = methodMap
    .map((entry) => new Set(entry.rows.map((r) => normalizeTargetName(r.target_variable)).filter(Boolean)))
    .filter((s) => s.size > 0);

  if (!sets.length) return new Set();
  let common = new Set(sets[0]);
  for (let i = 1; i < sets.length; i += 1) {
    common = new Set([...common].filter((x) => sets[i].has(x)));
  }
  return common;
}

function dedupeRowsByTarget(rows) {
  const byTarget = new Map();
  for (const row of rows) {
    const key = normalizeTargetName(row.target_variable);
    if (!key) continue;
    const ts = Date.parse(row.timestamp || "") || 0;
    const prev = byTarget.get(key);
    if (!prev || ts >= prev.ts) {
      byTarget.set(key, { ts, row });
    }
  }
  return [...byTarget.values()].map((x) => x.row);
}

function getPairIntersection(setA, setB) {
  return new Set([...setA].filter((v) => setB.has(v)));
}

function getStrictSection2TargetSet(classifierRows) {
  const eligible = new Set();
  for (const row of classifierRows || []) {
    const target = normalizeTargetName(row.target_variable);
    if (!target) continue;

    // Stricter paper-consistent rule:
    // - Discrete: REG_cat == 0
    // - Semi-continuous: REG_cat > 0 and CLAS_cat > 0
    const regCat = toNum(row.REG_cat);
    const clasCat = toNum(row.CLAS_cat);
    const isDiscrete = regCat !== null && regCat === 0;
    const isSemiContinuous = regCat !== null && clasCat !== null && regCat > 0 && clasCat > 0;

    if (isDiscrete || isSemiContinuous) {
      eligible.add(target);
    }
  }
  return eligible;
}

function buildMethodSummaries(regRows, clsRows, jointRows, mode) {
  const methodMap = getMethodRowsMap(regRows, clsRows, jointRows).map((entry) => ({
    ...entry,
    rows: dedupeRowsByTarget(entry.rows),
  }));

  const commonTargets = getCommonTargetSet(methodMap);

  const xgbTargets = new Set(
    (methodMap.find((m) => m.label === "XGB")?.rows || []).map((r) => normalizeTargetName(r.target_variable))
  );
  const xgbClassifierTargets = new Set(
    (methodMap.find((m) => m.label === "XGB+classifier")?.rows || []).map((r) => normalizeTargetName(r.target_variable))
  );
  const xgbPairCommonTargets = getPairIntersection(xgbTargets, xgbClassifierTargets);

  const summaries = methodMap.map((entry) => {
    let filteredRows = entry.rows;
    if (mode === "common") {
      if (entry.label === "XGB" || entry.label === "XGB+classifier") {
        filteredRows = entry.rows.filter((r) => xgbPairCommonTargets.has(normalizeTargetName(r.target_variable)));
      } else {
        filteredRows = entry.rows.filter((r) => commonTargets.has(normalizeTargetName(r.target_variable)));
      }
    }

    return {
      model: entry.label,
      rowCount: filteredRows.length,
      nrmse: averageMetric(filteredRows, "nrmse", 1),
      nmae: averageMetric(filteredRows, "nmae", 1),
      rta1: averageMetric(filteredRows, "RTA1pct", 100),
      rta5: averageMetric(filteredRows, "RTA5pct", 100),
      rta10: averageMetric(filteredRows, "RTA10pct", 100),
      training: averageMetric(filteredRows, "training_time_s", 1),
      running: averageMetric(filteredRows, "running_time_s", 1000),
    };
  });

  return {
    summaries,
    commonTargetCount: commonTargets.size,
    xgbPairCommonTargetCount: xgbPairCommonTargets.size,
  };
}

function summarizeRows(label, rows) {
  return {
    model: label,
    rowCount: rows.length,
    nrmse: averageMetric(rows, "nrmse", 1),
    nmae: averageMetric(rows, "nmae", 1),
    rta1: averageMetric(rows, "RTA1pct", 100),
    rta5: averageMetric(rows, "RTA5pct", 100),
    rta10: averageMetric(rows, "RTA10pct", 100),
    training: averageMetric(rows, "training_time_s", 1),
    running: averageMetric(rows, "running_time_s", 1000),
  };
}

function renderOverviewComparisonTable(tableId, summaries) {
  const table = document.getElementById(tableId);
  if (!table) return;

  table.innerHTML = [
    "<thead><tr><th>Method</th><th>Rows</th><th>Avg NRMSE</th><th>Avg NMAE</th><th>Avg RTA1% [%]</th><th>Avg RTA5% [%]</th><th>Avg RTA10% [%]</th><th>Avg Training Time [s]</th><th>Avg Running Time [ms]</th></tr></thead>",
    "<tbody>",
    ...summaries.map(
      (s) => `<tr><td>${escapeHTML(s.model)}</td><td>${s.rowCount}</td><td>${fmt(s.nrmse, 3)}</td><td>${fmt(s.nmae, 3)}</td><td>${fmt(s.rta1, 2)}</td><td>${fmt(s.rta5, 2)}</td><td>${fmt(s.rta10, 2)}</td><td>${fmt(s.training, 2)}</td><td>${fmt(s.running, 4)}</td></tr>`
    ),
    "</tbody>",
  ].join("");
}

function displayModelName(name) {
  const key = String(name ?? "").trim();
  return MODEL_DISPLAY_NAMES[key] || key;
}

function isRowMeaningful(row) {
  return Object.values(row || {}).some((v) => {
    if (v === null || v === undefined) return false;
    return String(v).trim() !== "";
  });
}

function sanitizeDatasheetRows(rows) {
  return (rows || [])
    .filter((row) => isRowMeaningful(row))
    .map((row) => {
      const cleaned = { ...row };
      const originalMethod = cleaned.surrogate_method;

      // Remove noisy timestamp from full datasheet table.
      delete cleaned.timestamp;

      if (cleaned.surrogate_method !== undefined) {
        cleaned.surrogate_model = displayModelName(cleaned.surrogate_method);
        delete cleaned.surrogate_method;
      }

      if (cleaned.stock_source_surrogate_method !== undefined) {
        cleaned.stock_source_surrogate_model = displayModelName(cleaned.stock_source_surrogate_method);
        delete cleaned.stock_source_surrogate_method;
      }

      // Keep a hidden source key for advanced filtering.
      cleaned.__source_method = String(originalMethod ?? "");

      return cleaned;
    });
}

function matchesMethodFilter(row, filterKey) {
  const method = String(row.__source_method ?? "").toLowerCase();
  const target = String(row.target_variable ?? "").toLowerCase();
  const combined = `${method} ${target}`;

  if (filterKey === "all") return true;
  if (filterKey === "xgb") return method.includes("xgb_single") || /(^|\W)xgb(\W|$)/.test(method);
  if (filterKey === "mlp") return method.includes("ae_mlp") || method.includes("mlp");
  if (filterKey === "gp") return method.includes("ae_gaussianprocess") || method.includes("gaussianprocess") || method.includes("gp");
  if (filterKey === "cr10") return combined.includes("xgb_hybrid10") || combined.includes("threshold 10");
  if (filterKey === "cr20") return combined.includes("xgb_hybrid20") || combined.includes("xgb_hybrid_tuned");
  if (filterKey === "cr30") return combined.includes("xgb_hybrid30") || combined.includes("threshold 30");
  if (filterKey === "cr40") return combined.includes("xgb_hybrid40") || combined.includes("threshold 40");
  if (filterKey === "xgb5") return /xgb/.test(combined) && /(5fold|5-fold|5_fold)/.test(combined);
  if (filterKey === "xgb6") return /xgb/.test(combined) && /(6fold|6-fold|6_fold)/.test(combined);
  if (filterKey === "informed") return combined.includes("informed_xgb_hybrid") || combined.includes("informed");
  return true;
}

function getSelectedOptions(selectEl) {
  return [...selectEl.options].filter((opt) => opt.selected).map((opt) => opt.value);
}

function ensureFilterSelectionWithinOptions(activeKey) {
  const allowed = new Set((DATASET_FILTER_OPTIONS[activeKey] || []).map((opt) => opt.value));
  const kept = datasheetState.methodFilter.filter((key) => allowed.has(key));
  datasheetState.methodFilter = kept.length ? kept : ["all"];
}

function renderDatasetMethodFilterOptions(selectEl) {
  const options = DATASET_FILTER_OPTIONS[datasheetState.activeKey] || DATASET_FILTER_OPTIONS.regression;
  ensureFilterSelectionWithinOptions(datasheetState.activeKey);

  selectEl.innerHTML = options
    .map((opt) => `<option value="${opt.value}" ${datasheetState.methodFilter.includes(opt.value) ? "selected" : ""}>${opt.label}</option>`)
    .join("");
}

function toCSV(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const escaped = (value) => {
    const raw = value ?? "";
    const text = String(raw).replace(/\r?\n/g, " ");
    if (/[",]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  };
  const header = cols.join(",");
  const body = rows.map((row) => cols.map((c) => escaped(row[c])).join(",")).join("\n");
  return `${header}\n${body}`;
}

function downloadCSV(filename, rows) {
  const csv = toCSV(rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function bindCSVButton(id, filename, getRows) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("click", () => {
    const rows = getRows();
    if (!rows.length) return;
    downloadCSV(filename, rows);
  });
}

function bindPlotDownloadButton(buttonId, chartId, filename) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.addEventListener("click", () => {
    const chart = document.getElementById(chartId);
    if (!chart) return;
    Plotly.downloadImage(chart, {
      format: "png",
      filename,
      width: 1700,
      height: 980,
      scale: 2,
    });
  });
}

function baseLayout(overrides = {}) {
  return {
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: PLOT_FONT,
    margin: { t: 28, r: 16, l: 58, b: 62 },
    ...overrides,
  };
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");
  const contextEl = document.getElementById("tab-context");
  const tabItems = [...buttons].map((btn) => ({
    target: btn.getAttribute("data-tab-target"),
    label: btn.textContent.trim(),
  }));
  const dataMethodsPanelTargets = ["tab-training", "tab-datasheets", "tab-method-code"];
  const dataMethodsButtons = [...document.querySelectorAll(".data-methods-btn")];
  let activeDataMethodsTarget = "tab-training";

  function normalizeTopTarget(target) {
    if (dataMethodsPanelTargets.includes(target)) return "tab-data-methods";
    return target;
  }

  function renderTabContext(target) {
    if (!contextEl || !target) return;
    const currentTopTarget = normalizeTopTarget(target);
    const currentTop = tabItems.find((item) => item.target === currentTopTarget);
    if (!currentTop) return;

    const otherTabs = tabItems
      .filter((item) => item.target !== currentTopTarget)
      .map((item) => `<button type="button" class="tab-context-link" data-tab-target="${item.target}">${item.label}</button>`)
      .join("");

    contextEl.innerHTML = `
      <span class="tab-context-label">Current tab:</span>
      <span class="tab-context-active">${currentTop.label}</span>
      <span class="tab-context-label">Other tabs:</span>
      ${otherTabs}
    `;

    contextEl.querySelectorAll(".tab-context-link").forEach((link) => {
      link.addEventListener("click", () => {
        activateTab(link.getAttribute("data-tab-target"));
      });
    });
  }

  function resizeChartsIn(rootEl) {
    if (!rootEl || typeof Plotly === "undefined" || !Plotly.Plots || !Plotly.Plots.resize) return;
    rootEl.querySelectorAll(".chart").forEach((chartEl) => {
      if (chartEl.classList.contains("js-plotly-plot")) {
        Plotly.Plots.resize(chartEl);
      }
    });
  }

  function resizeVisibleChartsForTopTab(currentTopTarget) {
    if (currentTopTarget === "tab-insights") {
      const activeChapter = document.querySelector("#tab-insights .insights-chapter.active");
      if (activeChapter) {
        resizeChartsIn(activeChapter);
      }
      return;
    }

    if (currentTopTarget === "tab-data-methods") {
      const activeDataPanel = document.querySelector("#tab-training.active, #tab-datasheets.active, #tab-method-code.active");
      if (activeDataPanel) {
        resizeChartsIn(activeDataPanel);
      }
      return;
    }

    const activeTopPanel = document.getElementById(currentTopTarget);
    if (activeTopPanel) {
      resizeChartsIn(activeTopPanel);
    }
  }

  function activateTab(target) {
    if (!target) return;
    const currentTopTarget = normalizeTopTarget(target);
    if (dataMethodsPanelTargets.includes(target)) {
      activeDataMethodsTarget = target;
    }
    const visibleDataMethodsTarget =
      currentTopTarget === "tab-data-methods"
        ? dataMethodsPanelTargets.includes(target)
          ? target
          : activeDataMethodsTarget
        : null;

    buttons.forEach((btn) => {
      const btnTarget = btn.getAttribute("data-tab-target");
      const active = btnTarget === currentTopTarget;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    panels.forEach((panel) => {
      const isActive =
        panel.id === currentTopTarget ||
        (visibleDataMethodsTarget !== null && panel.id === visibleDataMethodsTarget);
      panel.classList.toggle("active", isActive);
      panel.setAttribute("aria-hidden", isActive ? "false" : "true");
    });

    if (visibleDataMethodsTarget && dataMethodsButtons.length) {
      dataMethodsButtons.forEach((btn) => {
        const active = btn.getAttribute("data-dm-target") === visibleDataMethodsTarget;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
      });
    }

    renderTabContext(currentTopTarget);

    // Charts may be initially rendered while hidden; resize after panel becomes visible.
    window.requestAnimationFrame(() => {
      resizeVisibleChartsForTopTab(currentTopTarget);
    });
  }

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab-target");
      activateTab(target);
    });
  });

  document.querySelectorAll(".tab-jump").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const target = link.getAttribute("data-tab-target");
      const chapterTarget = link.getAttribute("data-chapter-target");
      activateTab(target);

      if (chapterTarget) {
        const chapterBtn = document.querySelector(`.subchapter-btn[data-chapter-target="${chapterTarget}"]`);
        chapterBtn?.click();
      }

      const panel = target ? document.getElementById(target) : null;
      panel?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  const initialTarget = [...buttons].find((btn) => btn.classList.contains("active"))?.getAttribute("data-tab-target");
  activateTab(initialTarget || "tab-home");
  window.dashboardActivateTab = activateTab;
}

function initInsightsSubchapters() {
  const buttons = document.querySelectorAll("#tab-insights .subchapter-btn");
  const chapters = document.querySelectorAll(".insights-chapter");
  if (!buttons.length || !chapters.length) return;

  function showChapter(targetId) {
    buttons.forEach((btn) => {
      const active = btn.getAttribute("data-chapter-target") === targetId;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    chapters.forEach((chapter) => {
      const active = chapter.id === targetId;
      chapter.classList.toggle("active", active);
      chapter.setAttribute("aria-hidden", active ? "false" : "true");
    });

    document.dispatchEvent(new CustomEvent("insights:chapter-changed", { detail: { targetId } }));

    const activeChapter = document.getElementById(targetId);
    if (!activeChapter || typeof Plotly === "undefined" || !Plotly.Plots || !Plotly.Plots.resize) return;

    activeChapter.querySelectorAll(".chart").forEach((chartEl) => {
      if (chartEl.classList.contains("js-plotly-plot")) {
        Plotly.Plots.resize(chartEl);
      }
    });
  }

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-chapter-target");
      if (!targetId) return;
      showChapter(targetId);
    });
  });
}

function initSupplementarySubchapters() {
  const buttons = document.querySelectorAll(".supplementary-btn");
  const chapters = document.querySelectorAll(".supplementary-chapter");
  if (!buttons.length || !chapters.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-supp-target");
      if (!targetId) return;
      buttons.forEach((b) => {
        const active = b === btn;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      chapters.forEach((chapter) => {
        const active = chapter.id === targetId;
        chapter.classList.toggle("active", active);
        chapter.setAttribute("aria-hidden", active ? "false" : "true");
      });
    });
  });
}

function initDataMethodsSubtabs() {
  const buttons = document.querySelectorAll(".data-methods-btn");
  if (!buttons.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-dm-target");
      if (!target) return;

      buttons.forEach((b) => {
        const active = b === btn;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      if (typeof window.dashboardActivateTab === "function") {
        window.dashboardActivateTab(target);
      }
    });
  });
}

function escapeHTML(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildFoldSensitivityRows(regRows) {
  const methodMap = {
    "5-fold cross-validation": "XGB_single_5fold",
    "6-fold cross-validation": "XGB_single",
    "7-fold cross-validation": "XGB_single_7-fold",
  };

  const baselineLabel = "6-fold cross-validation";
  const baselineMethod = methodMap[baselineLabel];

  function rowsForMethod(methodName) {
    return dedupeRowsByTarget(regRows.filter((r) => String(r.surrogate_method || "") === methodName));
  }

  const baselineRows = rowsForMethod(baselineMethod);
  const baselineByTarget = new Map(
    baselineRows.map((r) => [normalizeTargetName(r.target_variable), r]).filter(([k]) => Boolean(k))
  );

  function foldPercentages(k) {
    const trainValPoolPct = 85;
    const trainPct = (trainValPoolPct * (k - 1)) / k;
    const valPct = trainValPoolPct / k;
    return {
      train: `${Math.round(trainPct)}%`,
      validation: `${Math.round(valPct)}%`,
    };
  }

  function fmtDeltaTimes10Minus4(value) {
    if (!Number.isFinite(value)) return "-";
    const scaled = value * 1e4;
    const sign = scaled > 0 ? "+" : "-";
    return `${sign} ${fmt(Math.abs(scaled), 1)} x 10^-4`;
  }

  function buildComparison(altRows) {
    const pairs = [];
    for (const row of altRows) {
      const key = normalizeTargetName(row.target_variable);
      if (!key || !baselineByTarget.has(key)) continue;
      pairs.push({ base: baselineByTarget.get(key), alt: row });
    }

    const baseNrmse = pairs.map((p) => toNum(p.base.nrmse)).filter((v) => v !== null);
    const altNrmse = pairs.map((p) => toNum(p.alt.nrmse)).filter((v) => v !== null);
    const baseNmae = pairs.map((p) => toNum(p.base.nmae)).filter((v) => v !== null);
    const altNmae = pairs.map((p) => toNum(p.alt.nmae)).filter((v) => v !== null);

    const nrmseDelta = mean(altNrmse) - mean(baseNrmse);
    const nmaeDelta = mean(altNmae) - mean(baseNmae);

    const differentSettings = pairs.filter((p) => String(p.base.hyperparameters || "") !== String(p.alt.hyperparameters || "")).length;

    return {
      nrmseDelta,
      nmaeDelta,
      differentSettings,
    };
  }

  const rows = [
    {
      label: "5-fold cross-validation",
      ...foldPercentages(5),
      ...buildComparison(rowsForMethod(methodMap["5-fold cross-validation"])),
    },
    {
      label: "6-fold cross-validation",
      ...foldPercentages(6),
      nrmseDelta: null,
      nmaeDelta: null,
      differentSettings: null,
    },
    {
      label: "7-fold cross-validation",
      ...foldPercentages(7),
      ...buildComparison(rowsForMethod(methodMap["7-fold cross-validation"])),
    },
  ];

  return rows.map((r) => ({
    method: r.label,
    train: r.train,
    validation: r.validation,
    nrmse: r.nrmseDelta === null ? "-" : fmtDeltaTimes10Minus4(r.nrmseDelta),
    nmae: r.nmaeDelta === null ? "-" : fmtDeltaTimes10Minus4(r.nmaeDelta),
    settings: r.differentSettings === null ? "-" : String(r.differentSettings),
  }));
}

function renderFoldSensitivityTable(regRows) {
  const table = document.getElementById("regression-fold-sensitivity-table");
  if (!table) return;

  const rows = buildFoldSensitivityRows(regRows);
  table.innerHTML = [
    "<thead><tr><th>Cross-validation method</th><th>Percentage training data</th><th>Percentage validation data</th><th>Change in average NMRSE</th><th>Change in average NMAE</th><th>Amount of target variables different model settings chosen compared to 6-fold method</th></tr></thead>",
    "<tbody>",
    ...rows.map(
      (r) => `<tr><td>${escapeHTML(r.method)}</td><td>${escapeHTML(r.train)}</td><td>${escapeHTML(r.validation)}</td><td>${escapeHTML(r.nrmse)}</td><td>${escapeHTML(r.nmae)}</td><td>${escapeHTML(r.settings)}</td></tr>`
    ),
    "</tbody>",
  ].join("");
}

function renderBoundarySensitivityTable(classifierRows) {
  const table = document.getElementById("boundary-sensitivity-metrics-table");
  if (!table) return;
  const filterSelect = document.getElementById("boundary-target-filter");

  const methodDefs = [
    { label: "X=10", method: "XGB_HYBRID10_tuned" },
    { label: "X=20", method: "XGB_HYBRID_tuned" },
    { label: "X=30", method: "XGB_HYBRID30_tuned" },
    { label: "X=40", method: "XGB_HYBRID40_tuned" },
  ];

  const rowsByMethod = Object.fromEntries(
    methodDefs.map((m) => [m.label, dedupeRowsByTarget((classifierRows || []).filter((r) => String(r.surrogate_method || "") === m.method))])
  );

  const targetSets = methodDefs.map((m) => new Set(rowsByMethod[m.label].map((r) => normalizeTargetName(r.target_variable)).filter(Boolean)));
  const commonTargets = targetSets.reduce((acc, set, idx) => {
    if (idx === 0) return new Set(set);
    return new Set([...acc].filter((t) => set.has(t)));
  }, new Set());

  const alignedRowsByMethod = Object.fromEntries(
    methodDefs.map((m) => [
      m.label,
      rowsByMethod[m.label].filter((r) => commonTargets.has(normalizeTargetName(r.target_variable))),
    ])
  );

  function getFilteredTargets(mode) {
    if (mode !== "recurring") return new Set(commonTargets);

    // Use boundary X=10 as the recurring baseline (lowest boundary):
    // recurring-type targets are discrete or semi-continuous under that setting.
    const recurringTargets = new Set();
    (alignedRowsByMethod["X=10"] || []).forEach((r) => {
      const regCat = toNum(r.REG_cat);
      const clasCat = toNum(r.CLAS_cat);
      const target = normalizeTargetName(r.target_variable);
      if (!target) return;
      const isDiscrete = regCat !== null && regCat === 0;
      const isSemiContinuous = regCat !== null && clasCat !== null && regCat > 0 && clasCat > 0;
      if (isDiscrete || isSemiContinuous) {
        recurringTargets.add(target);
      }
    });
    return recurringTargets;
  }

  function renderWithMode(mode) {
    const targetSet = getFilteredTargets(mode);
    const filteredRowsByMethod = Object.fromEntries(
      methodDefs.map((m) => [
        m.label,
        alignedRowsByMethod[m.label].filter((r) => targetSet.has(normalizeTargetName(r.target_variable))),
      ])
    );

    function avg(label, key, scale = 1) {
      return averageMetric(filteredRowsByMethod[label], key, scale);
    }

    const metricRows = [
      {
        metric: "NRMSE",
        values: methodDefs.map((m) => {
          const v = avg(m.label, "nrmse", 1);
          return v === null ? "-" : fmt(v, 2);
        }),
      },
      {
        metric: "NMAE",
        values: methodDefs.map((m) => {
          const v = avg(m.label, "nmae", 1);
          return v === null ? "-" : fmt(v, 2);
        }),
      },
      {
        metric: "RTA1%",
        values: methodDefs.map((m) => {
          const v = avg(m.label, "RTA1pct", 100);
          return v === null ? "-" : `${fmt(v, 1)}%`;
        }),
      },
      {
        metric: "RTA5%",
        values: methodDefs.map((m) => {
          const v = avg(m.label, "RTA5pct", 100);
          return v === null ? "-" : `${fmt(v, 1)}%`;
        }),
      },
      {
        metric: "RTA10%",
        values: methodDefs.map((m) => {
          const v = avg(m.label, "RTA10pct", 100);
          return v === null ? "-" : `${fmt(v, 1)}%`;
        }),
      },
    ];

    const modeLabel = mode === "recurring"
      ? "targets with recurring values by X=10 type rule (discrete or semi-continuous)"
      : "shared target variables across X=10/20/30/40 methods";

    table.innerHTML = [
      `<thead><tr><th></th>${methodDefs.map((m) => `<th>${escapeHTML(m.label)}</th>`).join("")}</tr></thead>`,
      "<tbody>",
      ...metricRows.map((row) => `<tr><td>${escapeHTML(row.metric)}</td>${row.values.map((v) => `<td>${escapeHTML(v)}</td>`).join("")}</tr>`),
      `</tbody><caption>Based on ${targetSet.size} ${modeLabel}.</caption>`,
    ].join("");
  }

  if (filterSelect && !filterSelect.dataset.boundarySensitivityBound) {
    filterSelect.addEventListener("change", () => {
      renderWithMode(filterSelect.value || "all");
    });
    filterSelect.dataset.boundarySensitivityBound = "1";
  }

  const mode = filterSelect ? (filterSelect.value || "all") : "all";
  renderWithMode(mode);
}

function renderPerformanceLevelMetricsTable() {
  const table = document.getElementById("performance-level-metrics-table");
  const note = document.getElementById("performance-level-metrics-note");
  if (!table) return;

  const rows = [
    { metric: "NRMSE", technology: "0.48", subsystem: "0.43", system: "0.34" },
    { metric: "NMAE", technology: "0.21", subsystem: "0.27", system: "0.20" },
    { metric: "RTA1%", technology: "59.1%", subsystem: "24.4%", system: "39.8%" },
    { metric: "RTA5%", technology: "70.6%", subsystem: "53.0%", system: "64.2%" },
    { metric: "RTA10%", technology: "77.5%", subsystem: "71.5%", system: "76.4%" },
  ];

  table.innerHTML = [
    "<thead><tr><th>Metric</th><th>Technology-level</th><th>Subsystem-level</th><th>System-level</th></tr></thead>",
    "<tbody>",
    ...rows.map(
      (r) => `<tr><td>${escapeHTML(r.metric)}</td><td>${escapeHTML(r.technology)}</td><td>${escapeHTML(r.subsystem)}</td><td>${escapeHTML(r.system)}</td></tr>`
    ),
    "</tbody>",
  ].join("");

  if (note) {
    note.textContent = "Level definitions: Technology-level includes all Technology capacity and Technology operation target variables. Subsystem-level includes Electricity system capacity, Electricity system generation, Energy carriers imports, Energy carriers production, Flexibility and storage, and Commodity prices target variables. System-level includes Emissions and Total system costs target variables.";
  }
}

function renderOutputTypeComparisonTable() {
  const table = document.getElementById("output-type-comparison-table");
  const note = document.getElementById("output-type-comparison-note");
  if (!table) return;

  const rows = [
    { metric: "NRMSE", direct: "0.48", postProcessed: "0.43" },
    { metric: "NMAE", direct: "0.21", postProcessed: "0.27" },
    { metric: "RTA1%", direct: "59.0%", postProcessed: "25.1%" },
    { metric: "RTA5%", direct: "70.7%", postProcessed: "53.4%" },
    { metric: "RTA10%", direct: "77.7%", postProcessed: "71.5%" },
  ];

  table.innerHTML = [
    "<thead><tr><th>Metric</th><th>Direct output</th><th>Post-processed output</th></tr></thead>",
    "<tbody>",
    ...rows.map(
      (r) => `<tr><td>${escapeHTML(r.metric)}</td><td>${escapeHTML(r.direct)}</td><td>${escapeHTML(r.postProcessed)}</td></tr>`
    ),
    "</tbody>",
  ].join("");

  if (note) {
    note.textContent = "Post-processed targets are the user-defined subset. All remaining targets are treated as direct output.";
  }
}

function getDatasheetFilteredRows() {
  const rows = datasheetState.datasets[datasheetState.activeKey] || [];
  const query = datasheetState.search.trim().toLowerCase();

  const selectedFilters = datasheetState.methodFilter.length ? datasheetState.methodFilter : ["all"];
  const includeAll = selectedFilters.includes("all");
  const methodFilteredRows = rows
    .filter((row) => isRowMeaningful(row))
    .filter((row) => includeAll || selectedFilters.some((key) => matchesMethodFilter(row, key)));

  if (!query) return methodFilteredRows;

  return methodFilteredRows
    .filter((row) => Object.values(row).some((v) => String(v ?? "").toLowerCase().includes(query)));
}

function getDatasheetPageRows() {
  const rows = getDatasheetFilteredRows();
  const start = (datasheetState.page - 1) * datasheetState.pageSize;
  return rows.slice(start, start + datasheetState.pageSize);
}

function renderDatasheetTable() {
  const table = document.getElementById("datasheet-table");
  const summary = document.getElementById("datasheet-summary");
  const pageInfo = document.getElementById("datasheet-page-info");
  const prevBtn = document.getElementById("datasheet-prev");
  const nextBtn = document.getElementById("datasheet-next");

  const allRows = datasheetState.datasets[datasheetState.activeKey] || [];
  const filteredRows = getDatasheetFilteredRows();
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / datasheetState.pageSize));
  if (datasheetState.page > totalPages) datasheetState.page = totalPages;

  const pageRows = getDatasheetPageRows();
  const cols = pageRows.length
    ? Object.keys(pageRows[0])
    : allRows.length
      ? Object.keys(allRows[0])
      : [];
  const visibleCols = cols.filter((c) => !c.startsWith("__"));

  table.innerHTML = [
    "<thead><tr>",
    ...visibleCols.map((c) => `<th>${escapeHTML(c)}</th>`),
    "</tr></thead>",
    "<tbody>",
    ...pageRows.map((row) => {
      const cells = visibleCols
        .map((c) => {
          const raw = row[c] ?? "";
          const text = escapeHTML(raw);
          return `<td title="${text}">${text}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    }),
    "</tbody>",
  ].join("");

  const rowStart = filteredRows.length ? (datasheetState.page - 1) * datasheetState.pageSize + 1 : 0;
  const rowEnd = Math.min(datasheetState.page * datasheetState.pageSize, filteredRows.length);
  const optionMap = Object.fromEntries(
    (DATASET_FILTER_OPTIONS[datasheetState.activeKey] || []).map((opt) => [opt.value, opt.label])
  );
  const filterLabels = (datasheetState.methodFilter.length ? datasheetState.methodFilter : ["all"])
    .map((key) => optionMap[key] || key)
    .join(", ");
  summary.textContent = `Surrogate method: ${datasheetState.activeKey} | Dataset filter: ${filterLabels} | Showing ${rowStart}-${rowEnd} of ${filteredRows.length} filtered rows (${allRows.length} total)`;
  pageInfo.textContent = `Page ${datasheetState.page} / ${totalPages}`;

  prevBtn.disabled = datasheetState.page <= 1;
  nextBtn.disabled = datasheetState.page >= totalPages;
}

function initDatasheetsTab(regRows, clsRows, jointRows) {
  datasheetState.datasets.regression = sanitizeDatasheetRows(regRows);
  datasheetState.datasets.classifier = sanitizeDatasheetRows(clsRows);
  datasheetState.datasets.joint = sanitizeDatasheetRows(jointRows);

  const select = document.getElementById("datasheet-select");
  const methodFilter = document.getElementById("datasheet-method-filter");
  const search = document.getElementById("datasheet-search");
  const pageSize = document.getElementById("datasheet-page-size");
  const prevBtn = document.getElementById("datasheet-prev");
  const nextBtn = document.getElementById("datasheet-next");
  const dlFilteredBtn = document.getElementById("datasheet-download-filtered");
  const dlPageBtn = document.getElementById("datasheet-download-page");

  renderDatasetMethodFilterOptions(methodFilter);

  select.addEventListener("change", () => {
    datasheetState.activeKey = select.value;
    datasheetState.methodFilter = ["all"];
    renderDatasetMethodFilterOptions(methodFilter);
    datasheetState.page = 1;
    renderDatasheetTable();
  });

  methodFilter.addEventListener("change", () => {
    let selected = getSelectedOptions(methodFilter);
    if (!selected.length) {
      selected = ["all"];
      renderDatasetMethodFilterOptions(methodFilter);
    } else if (selected.includes("all") && selected.length > 1) {
      selected = selected.filter((v) => v !== "all");
      datasheetState.methodFilter = selected;
      renderDatasetMethodFilterOptions(methodFilter);
    }

    datasheetState.methodFilter = selected;
    datasheetState.page = 1;
    renderDatasheetTable();
  });

  search.addEventListener("input", () => {
    datasheetState.search = search.value;
    datasheetState.page = 1;
    renderDatasheetTable();
  });

  pageSize.addEventListener("change", () => {
    datasheetState.pageSize = Number(pageSize.value) || 50;
    datasheetState.page = 1;
    renderDatasheetTable();
  });

  prevBtn.addEventListener("click", () => {
    if (datasheetState.page > 1) {
      datasheetState.page -= 1;
      renderDatasheetTable();
    }
  });

  nextBtn.addEventListener("click", () => {
    const total = getDatasheetFilteredRows().length;
    const totalPages = Math.max(1, Math.ceil(total / datasheetState.pageSize));
    if (datasheetState.page < totalPages) {
      datasheetState.page += 1;
      renderDatasheetTable();
    }
  });

  if (dlFilteredBtn) {
    dlFilteredBtn.addEventListener("click", () => {
      const rows = getDatasheetFilteredRows();
      if (!rows.length) return;
      downloadCSV(`${datasheetState.activeKey}_filtered_results.csv`, rows);
    });
  }

  if (dlPageBtn) {
    dlPageBtn.addEventListener("click", () => {
      const rows = getDatasheetPageRows();
      if (!rows.length) return;
      downloadCSV(`${datasheetState.activeKey}_current_page_results.csv`, rows);
    });
  }

  renderDatasheetTable();
}

function initTrainingDataTab(trainingPayload) {
  const targets = Array.isArray(trainingPayload?.targets)
    ? trainingPayload.targets
        .filter((t) => t && t.outcome && Array.isArray(t.bin_centers) && Array.isArray(t.bin_counts))
        .map((t) => ({
          ...t,
          variable_type: extractVariableGroup(t.outcome),
        }))
        .sort((a, b) => String(a.outcome).localeCompare(String(b.outcome)))
    : [];

  trainingDataState.targets = targets;

  const countEl = document.getElementById("training-target-count");
  const typeFilterEl = document.getElementById("training-variable-type-filter");
  const searchEl = document.getElementById("training-target-search");
  const selectEl = document.getElementById("training-target-select");
  const selectAllBtn = document.getElementById("training-select-all");
  const clearBtn = document.getElementById("training-clear");
  const noteEl = document.getElementById("training-selection-note");
  const dataTypeTableEl = document.getElementById("training-data-type-table");
  const dataTypeAnalysisTableEl = document.getElementById("training-data-type-analysis-table");
  const chartId = "training-distribution-plot";

  if (!countEl || !typeFilterEl || !searchEl || !selectEl || !selectAllBtn || !clearBtn || !noteEl || !dataTypeTableEl || !dataTypeAnalysisTableEl) return;

  countEl.textContent = String(
    TRAINING_VARIABLE_TYPE_REFERENCE_ROWS.reduce((sum, row) => sum + row.count, 0)
  );

  let filteredTargets = targets;
  const availableTypes = [...new Set(targets.map((t) => String(t.variable_type || "Other")))];
  const orderedTypes = [...Object.keys(VARIABLE_GROUPS), "Other"].filter((t) => availableTypes.includes(t));
  let selectedTypes = new Set(orderedTypes);

  typeFilterEl.innerHTML = orderedTypes
    .map((type) => `<option value="${escapeHTML(type)}" selected>${escapeHTML(type)}</option>`)
    .join("");

  function remapOutcomeName(outcome) {
    return String(outcome || "")
      .replace(/techUseNet/gi, "Technology Operation")
      .replace(/techStocks?/gi, "Technology Capacity")
      .replace(/comPrices/gi, "Commodity Prices");
  }

  function stripEmbeddedTechId(text, techId) {
    let cleaned = String(text || "");
    const id = String(techId || "").trim();
    if (!id) return cleaned;

    // Remove direct occurrences of the technology ID inside the outcome label.
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    cleaned = cleaned.replace(new RegExp(`\\b${escaped}\\b`, "g"), "");

    // Clean up duplicate whitespace left after stripping ID tokens.
    cleaned = cleaned.replace(/\s+/g, " ").trim();
    return cleaned;
  }

  function formatTrainingTargetLabel(target) {
    const techId = String(target.technology_id || "").trim();
    const baseRaw = remapOutcomeName(target.outcome);
    const base = stripEmbeddedTechId(baseRaw, techId);
    const techName = String(target.technology_name || "").trim();
    const withTechName = techName ? `${base} ${techName}` : base;
    return techId ? `${withTechName} [${techId}]` : withTechName;
  }

  function optionLabel(t) {
    return formatTrainingTargetLabel(t);
  }

  function formatRecurringValueLabel(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return String(value);
    if (num === 0) return "0";

    const abs = Math.abs(num);
    if (abs < 0.1) return num.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    if (abs < 1) return num.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    return num.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  function formatRecurringValues(target) {
    const values = Array.isArray(target.recurring_values) ? target.recurring_values : [];
    if (!values.length) return "[]";
    const labels = [...new Set(values.map((v) => formatRecurringValueLabel(v)))];
    return `[${labels.map((label) => `'${label}'`).join(", ")}]`;
  }

  function renderOptions() {
    const chosenTypes = getSelectedOptions(typeFilterEl);
    selectedTypes = new Set(chosenTypes);

    const query = searchEl.value.trim().toLowerCase();
    filteredTargets = targets
      .filter((t) => selectedTypes.has(String(t.variable_type || "Other")))
      .filter((t) => {
        if (!query) return true;
        const out = String(t.outcome || "").toLowerCase();
        const tech = String(t.technology_name || "").toLowerCase();
        const techId = String(t.technology_id || "").toLowerCase();
        const label = formatTrainingTargetLabel(t).toLowerCase();
        return out.includes(query) || tech.includes(query) || techId.includes(query) || label.includes(query);
      });

    selectEl.innerHTML = filteredTargets
      .map((t) => {
        const value = escapeHTML(String(t.outcome));
        const selected = trainingDataState.selected.has(String(t.outcome)) ? "selected" : "";
        return `<option value="${value}" ${selected}>${escapeHTML(optionLabel(t))}</option>`;
      })
      .join("");
  }

  function renderDataTypeSummaryTable() {
    const total = TRAINING_VARIABLE_TYPE_REFERENCE_ROWS.reduce((sum, row) => sum + row.count, 0);
    dataTypeTableEl.innerHTML = [
      "<thead><tr><th>Variable type</th><th>Amount of TVs</th></tr></thead>",
      "<tbody>",
      ...TRAINING_VARIABLE_TYPE_REFERENCE_ROWS.map(
        (row) => `<tr><td>${escapeHTML(row.type)}</td><td>${row.count}</td></tr>`
      ),
      `<tr><td><strong>Total</strong></td><td><strong>${total}</strong></td></tr>`,
      "</tbody>",
    ].join("");
  }

  function renderDataTypeAnalysisTable() {
    const counts = {
      Continuous: 0,
      "Semi-continuous": 0,
      Discrete: 0,
    };

    filteredTargets.forEach((target) => {
      const key = String(target.data_type || "");
      if (Object.prototype.hasOwnProperty.call(counts, key)) {
        counts[key] += 1;
      }
    });

    const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
    const pct = (value) => (total > 0 ? `${((100 * value) / total).toFixed(1)}%` : "0.0%");

    dataTypeAnalysisTableEl.innerHTML = [
      "<thead><tr><th>Training data type</th><th>Target variables</th><th>Share</th><th>Definition</th></tr></thead>",
      "<tbody>",
      `<tr><td>Continuous</td><td>${counts.Continuous}</td><td>${pct(counts.Continuous)}</td><td>No recurring values appear 20 times or more in the training set.</td></tr>`,
      `<tr><td>Semi-continuous</td><td>${counts["Semi-continuous"]}</td><td>${pct(counts["Semi-continuous"])}</td><td>Mix of recurring values and non-recurring values in the training set.</td></tr>`,
      `<tr><td>Discrete</td><td>${counts.Discrete}</td><td>${pct(counts.Discrete)}</td><td>Recurring values only, with each recurring value appearing 20 times or more.</td></tr>`,
      `<tr><td><strong>Total</strong></td><td><strong>${total}</strong></td><td><strong>100.0%</strong></td><td><strong>Filtered varying target variables currently in scope.</strong></td></tr>`,
      "</tbody>",
    ].join("");
  }

  function autoSelectFilteredTargets() {
    trainingDataState.selected = new Set(filteredTargets.map((t) => String(t.outcome)));
    renderOptions();
  }

  function syncSelectionFromSelect() {
    trainingDataState.selected = new Set(getSelectedOptions(selectEl));
  }

  function renderDistributions() {
    const selectedOutcomes = [...trainingDataState.selected];
    const allowedOutcomes = new Set(filteredTargets.map((t) => String(t.outcome)));
    const selectedTargets = selectedOutcomes
      .filter((o) => allowedOutcomes.has(String(o)))
      .map((o) => trainingDataState.targets.find((t) => String(t.outcome) === String(o)))
      .filter(Boolean);

    if (!selectedTargets.length) {
      noteEl.textContent = "Select at least one target variable type or target variable to show value distributions.";
      Plotly.newPlot(
        chartId,
        [],
        baseLayout({
          height: 460,
          annotations: [
            {
              x: 0.5,
              y: 0.5,
              xref: "paper",
              yref: "paper",
              text: "No target selected",
              showarrow: false,
              font: { size: 16, color: "#667085" },
            },
          ],
        }),
        { responsive: true, displayModeBar: false }
      );
      return;
    }

    const renderedTargets = selectedTargets.slice(0, trainingDataState.maxPanels);
    const maxHit = selectedTargets.length > trainingDataState.maxPanels;
    const activeTypes = [...selectedTypes];
    const typeText = activeTypes.length ? `Variable types: ${activeTypes.join(", ")}. ` : "Variable types: none selected. ";
    noteEl.textContent = maxHit
      ? `${typeText}Showing ${renderedTargets.length} of ${selectedTargets.length} matching targets to keep the page responsive.`
      : `${typeText}Showing ${renderedTargets.length} matching target variable distribution(s).`;

    const cols = Math.min(3, Math.max(1, renderedTargets.length));
    const rows = Math.max(1, Math.ceil(renderedTargets.length / cols));
    const layout = baseLayout({
      showlegend: false,
      height: Math.max(540, rows * 245),
      margin: { t: 34, r: 16, l: 42, b: 38 },
      grid: { rows, columns: cols, pattern: "independent" },
      annotations: [],
    });

    const traces = [];
    renderedTargets.forEach((target, idx) => {
      const axisNum = idx + 1;
      const xref = axisNum === 1 ? "x" : `x${axisNum}`;
      const yref = axisNum === 1 ? "y" : `y${axisNum}`;

      const xs = (target.bin_centers || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
      const ys = (target.bin_counts || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
      const n = Math.min(xs.length, ys.length);

      traces.push({
        type: "bar",
        x: xs.slice(0, n),
        y: ys.slice(0, n),
        marker: { color: "#6f95bd", line: { color: "#4d6e92", width: 0.45 } },
        opacity: 0.9,
        xaxis: xref,
        yaxis: yref,
        hovertemplate: "Value=%{x:.4g}<br>Frequency=%{y}<extra></extra>",
      });

      const unit = String(target.unit || "").trim() || "N/A";
      const title = formatTrainingTargetLabel(target);
      const recurringText = formatRecurringValues(target);

      layout[`xaxis${axisNum === 1 ? "" : axisNum}`] = {
        title: `Value (${unit})`,
        titlefont: { size: 11 },
        tickfont: { size: 10 },
        showgrid: false,
        zeroline: false,
        showline: true,
        linecolor: "#3f3f3f",
        linewidth: 1,
        mirror: false,
      };
      layout[`yaxis${axisNum === 1 ? "" : axisNum}`] = {
        title: "Frequency",
        titlefont: { size: 11 },
        tickfont: { size: 10 },
        showgrid: false,
        zeroline: false,
        showline: true,
        linecolor: "#3f3f3f",
        linewidth: 1,
        mirror: false,
      };

      const xDomainRef = axisNum === 1 ? "x domain" : `x${axisNum} domain`;
      const yDomainRef = axisNum === 1 ? "y domain" : `y${axisNum} domain`;
      layout.annotations.push({
        text: `<b>${escapeHTML(title)}</b><br>(${escapeHTML(unit)})`,
        xref: xDomainRef,
        yref: yDomainRef,
        x: 0.5,
        y: 1.06,
        showarrow: false,
        font: { size: 11, color: "#1f2a37" },
      });

      layout.annotations.push({
        text: `Recurring: ${escapeHTML(recurringText)}`,
        xref: xDomainRef,
        yref: yDomainRef,
        x: 0.995,
        y: 1.01,
        showarrow: false,
        xanchor: "right",
        yanchor: "top",
        align: "right",
        bordercolor: "#868686",
        borderwidth: 0.8,
        borderpad: 2,
        bgcolor: "rgba(255,255,255,0.9)",
        font: { size: 9, color: "#333333" },
      });
    });

    Plotly.newPlot(chartId, traces, layout, { responsive: true, displayModeBar: false });
  }

  searchEl.addEventListener("input", () => {
    renderOptions();
    autoSelectFilteredTargets();
    renderDataTypeAnalysisTable();
    renderDistributions();
  });

  typeFilterEl.addEventListener("change", () => {
    renderOptions();
    autoSelectFilteredTargets();
    renderDataTypeAnalysisTable();
    renderDistributions();
  });

  selectEl.addEventListener("change", () => {
    syncSelectionFromSelect();
    renderDistributions();
  });

  selectAllBtn.addEventListener("click", () => {
    trainingDataState.selected = new Set(filteredTargets.map((t) => String(t.outcome)));
    renderOptions();
    renderDistributions();
  });

  clearBtn.addEventListener("click", () => {
    trainingDataState.selected = new Set();
    renderOptions();
    renderDistributions();
  });

  if (targets.length) {
    trainingDataState.selected = new Set(filteredTargets.slice(0, trainingDataState.maxPanels).map((t) => String(t.outcome)));
  }
  renderOptions();
  renderDataTypeSummaryTable();
  renderDataTypeAnalysisTable();
  renderDistributions();
}

function extractVariableGroup(targetVariable) {
  const tv = String(targetVariable || "").toLowerCase();
  for (const [groupName, patterns] of Object.entries(VARIABLE_GROUPS)) {
    for (const p of patterns) {
      const pLow = p.toLowerCase();
      if (tv.includes(pLow) || tv.includes(pLow.replace(/ /g, "_"))) {
        return groupName;
      }
    }
  }
  return "Other";
}

function classifyDataType(row) {
  const reg = toNum(row.REG_cat);
  const cls = toNum(row.CLAS_cat);
  if (reg === 0) return "Discrete";
  if (cls === 0) return "Continuous";
  return "Semi-continuous";
}

function newestByTarget(rows) {
  const map = new Map();
  for (const row of rows) {
    const k = row.target_variable;
    if (!k) continue;
    const ts = Date.parse(row.timestamp || "") || 0;
    const prev = map.get(k);
    if (!prev || ts >= prev.ts) {
      map.set(k, { ts, row });
    }
  }
  return [...map.values()].map((x) => x.row);
}

function averageRowsByTarget(rows) {
  const metricKeys = ["nrmse", "nmae", "RTA1pct", "RTA5pct", "RTA10pct", "training_time_s", "running_time_s"];
  const byTarget = new Map();

  rows.forEach((row) => {
    const targetKey = normalizeTargetName(row.target_variable);
    if (!targetKey) return;

    if (!byTarget.has(targetKey)) {
      const seed = { target_variable: row.target_variable };
      metricKeys.forEach((key) => {
        seed[key] = { sum: 0, count: 0 };
      });
      byTarget.set(targetKey, seed);
    }

    const agg = byTarget.get(targetKey);
    metricKeys.forEach((key) => {
      const value = toNum(row[key]);
      if (value === null) return;
      agg[key].sum += value;
      agg[key].count += 1;
    });
  });

  return [...byTarget.values()].map((agg) => {
    const out = { target_variable: agg.target_variable };
    metricKeys.forEach((key) => {
      out[key] = agg[key].count ? agg[key].sum / agg[key].count : null;
    });
    return out;
  });
}

function percentileLinear(sortedValues, p) {
  if (!sortedValues.length) return null;
  if (sortedValues.length === 1) return sortedValues[0];
  const idx = (sortedValues.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedValues[lo];
  const w = idx - lo;
  return sortedValues[lo] * (1 - w) + sortedValues[hi] * w;
}

function computeMatplotlibBoxStats(values) {
  const nums = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!nums.length) {
    return {
      q1: null,
      median: null,
      q3: null,
      lowerFence: null,
      upperFence: null,
    };
  }

  const q1 = percentileLinear(nums, 0.25);
  const medianValue = percentileLinear(nums, 0.5);
  const q3 = percentileLinear(nums, 0.75);
  const iqr = q3 - q1;
  const lowBound = q1 - 1.5 * iqr;
  const highBound = q3 + 1.5 * iqr;

  const inlierLow = nums.find((v) => v >= lowBound);
  const inlierHigh = [...nums].reverse().find((v) => v <= highBound);

  return {
    q1,
    median: medianValue,
    q3,
    lowerFence: inlierLow !== undefined ? inlierLow : nums[0],
    upperFence: inlierHigh !== undefined ? inlierHigh : nums[nums.length - 1],
  };
}

function getSection2AlignedRows(regRows, clsRows, jointRows) {
  const xgbRaw = regRows.filter((r) => String(r.surrogate_method || "").toLowerCase() === "xgb_single");
  const hybridRaw = clsRows.filter((r) => String(r.surrogate_method || "").toLowerCase() === "xgb_hybrid_tuned");

  const recurringTargetSet = getStrictSection2TargetSet(dedupeRowsByTarget(hybridRaw));
  const setXGB = new Set(xgbRaw.map((r) => normalizeTargetName(r.target_variable)).filter(Boolean));
  const setClf = new Set(hybridRaw.map((r) => normalizeTargetName(r.target_variable)).filter(Boolean));
  const commonTargets = getPairIntersection(setXGB, setClf);

  const xgbCommon = xgbRaw.filter((r) => commonTargets.has(normalizeTargetName(r.target_variable)));
  const hybridCommon = hybridRaw.filter((r) => commonTargets.has(normalizeTargetName(r.target_variable)));

  function applyRecurringOnlySection2(rows) {
    if (recurringOnlyMode !== "recurring") return rows;
    return rows.filter((r) => recurringTargetSet.has(normalizeTargetName(r.target_variable)));
  }

  const xgbFiltered = applyRecurringOnlySection2(xgbCommon);
  const hybridFiltered = applyRecurringOnlySection2(hybridCommon);

  return {
    xgbRows: averageRowsByTarget(xgbFiltered),
    xgbClassifierRows: averageRowsByTarget(hybridFiltered),
    commonTargets,
    recurringTargetSet,
  };
}

function renderSection2MetricDistributions(regRows, clsRows, jointRows) {
  const chartEl = document.getElementById("xgb-clf-metrics-boxplots");
  if (!chartEl) return;

  const noteEl = document.getElementById("xgb-clf-metrics-note");
  const aligned = getSection2AlignedRows(regRows, clsRows, jointRows);

  const methods = [
    { label: "Regressor", rows: aligned.xgbRows, color: "#FF9999" },
    { label: "Classifier-regressor", rows: aligned.xgbClassifierRows, color: "#99CCFF" },
  ];

  const specs = [
    { key: "nrmse", noteLabel: "NRMSE", title: "NRMSE Distribution", yTitle: "NRMSE", scale: 1, decimals: 3 },
    { key: "nmae", noteLabel: "NMAE", title: "NMAE Distribution", yTitle: "NMAE", scale: 1, decimals: 3 },
    { key: "RTA1pct", noteLabel: "RTA1%", title: "1% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { key: "RTA5pct", noteLabel: "RTA5%", title: "5% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { key: "RTA10pct", noteLabel: "RTA10%", title: "10% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { key: "training_time_s", noteLabel: "Training time", title: "Training Time Distribution", yTitle: "Training Time (s)", scale: 1, decimals: 2 },
    { key: "running_time_s", noteLabel: "Running time", title: "Running Time Distribution", yTitle: "Running Time (s)", scale: 1, decimals: 6 },
  ];

  const layout = baseLayout({
    showlegend: false,
    margin: { t: 56, r: 24, l: 40, b: 34 },
    grid: { rows: 2, columns: 4, pattern: "independent" },
    height: 820,
    annotations: [],
  });

  const xDomains = [
    [0.0, 0.215],
    [0.265, 0.48],
    [0.53, 0.745],
    [0.795, 0.995],
  ];
  const yDomainTop = [0.56, 1.0];
  const yDomainBottom = [0.0, 0.44];

  const traces = [];
  const excludedTargetsAny = new Set();
  const excludedTargetsByMetric = new Map(specs.map((spec) => [spec.key, new Set()]));
  specs.forEach((spec, idx) => {
    const axisNum = idx + 1;
    const xref = axisNum === 1 ? "x" : `x${axisNum}`;
    const yref = axisNum === 1 ? "y" : `y${axisNum}`;

    methods.forEach((method) => {
      const points = (method.rows || [])
        .map((r) => ({
          target: normalizeTargetName(r.target_variable),
          value: toNum(r[spec.key]),
        }))
        .filter((point) => point.target && point.value !== null)
        .map((point) => ({ ...point, value: point.value * spec.scale }));
      const values = points.map((point) => point.value);

      const stats = computeMatplotlibBoxStats(values);
      if (stats.q1 === null) return;

      const iqr = stats.q3 - stats.q1;
      const lowBound = stats.q1 - 1.5 * iqr;
      const highBound = stats.q3 + 1.5 * iqr;
      points.forEach((point) => {
        if (point.value < lowBound || point.value > highBound) {
          excludedTargetsAny.add(point.target);
          excludedTargetsByMetric.get(spec.key).add(point.target);
        }
      });

      traces.push({
        type: "box",
        name: method.label,
        x: [method.label],
        q1: [stats.q1],
        median: [stats.median],
        q3: [stats.q3],
        lowerfence: [stats.lowerFence],
        upperfence: [stats.upperFence],
        xaxis: xref,
        yaxis: yref,
        marker: { color: method.color },
        line: { color: "#666666", width: 1 },
        fillcolor: method.color,
        boxmean: false,
        boxpoints: false,
        whiskerwidth: 0.8,
        yhoverformat: ".2f",
        hovertemplate: `${method.label}<br>min: %{min:.2f}<br>q1: %{q1:.2f}<br>median: %{median:.2f}<br>q3: %{q3:.2f}<br>max: %{max:.2f}<extra></extra>`,
      });
    });

    layout[`xaxis${axisNum === 1 ? "" : axisNum}`] = {
      tickangle: 0,
      showline: true,
      linecolor: "#111111",
      linewidth: 1.5,
      mirror: true,
      categoryorder: "array",
      categoryarray: ["Regressor", "Classifier-regressor"],
      tickfont: { size: 11 },
      domain: xDomains[idx % 4],
    };

    layout[`yaxis${axisNum === 1 ? "" : axisNum}`] = {
      title: spec.yTitle,
      showgrid: false,
      zeroline: false,
      showline: true,
      linecolor: "#111111",
      linewidth: 1.5,
      mirror: true,
      hoverformat: ".2f",
      range: spec.range,
      tickfont: { size: 11 },
      titlefont: { size: 13 },
      title_standoff: 8,
      domain: idx < 4 ? yDomainTop : yDomainBottom,
    };

    const xDomainRef = axisNum === 1 ? "x domain" : `x${axisNum} domain`;
    const yDomainRef = axisNum === 1 ? "y domain" : `y${axisNum} domain`;
    layout.annotations.push({
      text: `<b>${spec.title}</b>`,
      xref: xDomainRef,
      yref: yDomainRef,
      x: 0.5,
      y: 1.02,
      showarrow: false,
      font: { size: 15, color: "#1f2a37" },
      xanchor: "center",
      yanchor: "bottom",
    });
  });

  Plotly.newPlot("xgb-clf-metrics-boxplots", traces, layout, {
    responsive: true,
    displayModeBar: false,
  });

  if (noteEl) {
    const modeSuffix = recurringOnlyMode === "recurring"
      ? ` (filtered to semi-continuous/discrete targets: ${aligned.recurringTargetSet.size} eligible)`
      : "";
    const metricSummary = specs
      .map((spec) => `${spec.noteLabel}: ${excludedTargetsByMetric.get(spec.key).size}`)
      .join("; ");
    noteEl.textContent = `Included shared targets: ${aligned.xgbRows.length}${modeSuffix}. Outlier note (1.5×IQR, hidden in boxplots): ${excludedTargetsAny.size} targets are excluded in at least one metric. Per metric: ${metricSummary}.`;
  }
}

function renderOverviewSummary() {
  const { xgbPairCommonTargetCount } = buildMethodSummaries(
    dashboardSourceRows.regression,
    dashboardSourceRows.classifier,
    dashboardSourceRows.joint,
    "common"
  );

  const methodMapRaw = getMethodRowsMap(
    dashboardSourceRows.regression,
    dashboardSourceRows.classifier,
    dashboardSourceRows.joint
  );
  const methodMap = Object.fromEntries(
    methodMapRaw.map((entry) => {
      const deduped = dedupeRowsByTarget(entry.rows);
      return [entry.label, deduped];
    })
  );
  const recurringTargetSet = getStrictSection2TargetSet(methodMap["XGB+classifier"] || []);

  const setXGB = new Set((methodMap.XGB || []).map((r) => normalizeTargetName(r.target_variable)));
  const setMLP = new Set((methodMap.MLP || []).map((r) => normalizeTargetName(r.target_variable)));
  const setGP = new Set((methodMap.GP || []).map((r) => normalizeTargetName(r.target_variable)));
  const setClf = new Set((methodMap["XGB+classifier"] || []).map((r) => normalizeTargetName(r.target_variable)));
  const setInf = new Set((methodMap.informed_classifier || []).map((r) => normalizeTargetName(r.target_variable)));

  const commonConventional = new Set([...setXGB].filter((t) => setMLP.has(t) && setGP.has(t)));
  const commonXgbClf = getPairIntersection(setXGB, setClf);
  const commonClfInf = getPairIntersection(setClf, setInf);

  function alignedRows(label, targetSet) {
    const rows = methodMap[label] || [];
    // Always align by shared targets for fair per-section comparison.
    return rows.filter((r) => targetSet.has(normalizeTargetName(r.target_variable)));
  }

  function applyRecurringOnlySection2(rows) {
    if (recurringOnlyMode !== "recurring") return rows;
    return rows.filter((r) => recurringTargetSet.has(normalizeTargetName(r.target_variable)));
  }

  const conventionalSummaries = [
    summarizeRows("XGB", alignedRows("XGB", commonConventional)),
    summarizeRows("MLP", alignedRows("MLP", commonConventional)),
    summarizeRows("GP", alignedRows("GP", commonConventional)),
  ];

  const xgbSectionRows = applyRecurringOnlySection2(alignedRows("XGB", commonXgbClf));
  const xgbClassifierSectionRows = applyRecurringOnlySection2(alignedRows("XGB+classifier", commonXgbClf));
  const xgbVsClassifierSummaries = [
    summarizeRows("XGB", xgbSectionRows),
    summarizeRows("XGB+classifier", xgbClassifierSectionRows),
  ];

  const classifierVsInformedSummaries = [
    summarizeRows("XGB+classifier", alignedRows("XGB+classifier", commonClfInf)),
    summarizeRows("informed_classifier", alignedRows("informed_classifier", commonClfInf)),
  ];

  renderOverviewComparisonTable("overview-conventional-table", conventionalSummaries);
  renderOverviewComparisonTable("overview-xgb-vs-clf-table", xgbVsClassifierSummaries);
  renderOverviewComparisonTable("overview-clf-vs-informed-table", classifierVsInformedSummaries);

  const note = document.getElementById("common-target-note");
  if (note) {
    const recurringSuffix = recurringOnlyMode === "recurring"
      ? ` Stricter filter is applied to section 2 only (${recurringTargetSet.size} semi-continuous/discrete targets by REG_cat/CLAS_cat).`
      : "";
    note.textContent = `Shared-target mode enabled: GP/MLP/XGB share ${commonConventional.size} targets, XGB/XGB+classifier share ${xgbPairCommonTargetCount}, and XGB+classifier/informed_classifier share ${commonClfInf.size}.${recurringSuffix}`;
  }

  renderSection2MetricDistributions(
    dashboardSourceRows.regression,
    dashboardSourceRows.classifier,
    dashboardSourceRows.joint
  );
}

function renderRegressionSection(regRows, violationsPayload, trainingPayload) {
  const clean = regRows.filter((r) => ["XGB_single", "AE_MLP", "AE_GaussianProcess"].includes(r.surrogate_method));
  const methods = ["XGB_single", "AE_MLP", "AE_GaussianProcess"];
  const labels = {
    XGB_single: "XGB",
    AE_MLP: "MLP",
    AE_GaussianProcess: "GP",
  };
  // Match Streamlit/matplotlib Set3 tone used in main.py.
  const colors = ["#8dd3c7", "#ffffb3", "#bebada"];

  // Mirror main.py behavior: plot only targets that are shared by XGB/MLP/GP.
  const methodRows = Object.fromEntries(
    methods.map((m) => [m, dedupeRowsByTarget(clean.filter((r) => r.surrogate_method === m))])
  );
  const targetSets = methods.map(
    (m) => new Set(methodRows[m].map((r) => normalizeTargetName(r.target_variable)).filter(Boolean))
  );
  const commonTargets = targetSets.reduce((acc, set, idx) => {
    if (idx === 0) return new Set(set);
    return new Set([...acc].filter((t) => set.has(t)));
  }, new Set());

  const commonRows = Object.fromEntries(
    methods.map((m) => [
      m,
      methodRows[m].filter((r) => commonTargets.has(normalizeTargetName(r.target_variable))),
    ])
  );

  const specs = [
    { sourceKey: "nrmse", noteLabel: "NRMSE", title: "NRMSE", yTitle: "NRMSE", scale: 1, decimals: 3 },
    { sourceKey: "nmae", noteLabel: "NMAE", title: "NMAE", yTitle: "NMAE", scale: 1, decimals: 3 },
    { sourceKey: "RTA1pct", noteLabel: "RTA1%", title: "RTA1%", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { sourceKey: "RTA5pct", noteLabel: "RTA5%", title: "RTA5%", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { sourceKey: "RTA10pct", noteLabel: "RTA10%", title: "RTA10%", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { sourceKey: "training_time_s", noteLabel: "Training time", title: "Training Time", yTitle: "Training Time (s)", scale: 1, decimals: 2 },
    { sourceKey: "running_time_s", noteLabel: "Running time", title: "Running Time", yTitle: "Running Time (ms)", scale: 1000, decimals: 2 },
  ];

  const traces = [];
  const excludedTargets = new Set();
  const excludedTargetsByMetric = new Map(specs.map((spec) => [spec.sourceKey, new Set()]));

  function getIqrFilteredSeries(rows, sourceKey, scale) {
    const points = rows
      .map((row) => ({
        target: normalizeTargetName(row.target_variable),
        value: toNum(row[sourceKey]),
      }))
      .filter((point) => point.target && point.value !== null)
      .map((point) => ({ ...point, value: point.value * scale }));

    const nums = points.map((point) => point.value).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    if (nums.length < 4) {
      return {
        values: nums,
        excluded: new Set(),
      };
    }

    const q = (arr, p) => {
      const idx = (arr.length - 1) * p;
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      if (lo === hi) return arr[lo];
      const w = idx - lo;
      return arr[lo] * (1 - w) + arr[hi] * w;
    };

    const q1 = q(nums, 0.25);
    const q3 = q(nums, 0.75);
    const iqr = q3 - q1;
    const low = q1 - 1.5 * iqr;
    const high = q3 + 1.5 * iqr;

    const kept = [];
    const excluded = new Set();
    points.forEach((point) => {
      if (point.value >= low && point.value <= high) {
        kept.push(point.value);
      } else {
        excluded.add(point.target);
      }
    });

    kept.sort((a, b) => a - b);
    return { values: kept, excluded };
  }

  const layout = baseLayout({
    showlegend: false,
    margin: { t: 56, r: 24, l: 36, b: 34 },
    grid: { rows: 2, columns: 4, pattern: "independent" },
    height: 820,
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    annotations: [],
  });

  const xDomains = [
    [0.0, 0.215],
    [0.265, 0.48],
    [0.53, 0.745],
    [0.795, 0.995],
  ];
  const yDomainTop = [0.56, 1.0];
  const yDomainBottom = [0.0, 0.44];

  specs.forEach((spec, idx) => {
    const axisNum = idx + 1;
    const xref = axisNum === 1 ? "x" : `x${axisNum}`;
    const yref = axisNum === 1 ? "y" : `y${axisNum}`;

    methods.forEach((m, i) => {
      const series = getIqrFilteredSeries(commonRows[m], spec.sourceKey, spec.scale);
      const values = series.values;
      series.excluded.forEach((target) => {
        excludedTargets.add(target);
        excludedTargetsByMetric.get(spec.sourceKey).add(target);
      });

      traces.push({
        type: "box",
        name: labels[m],
        x: Array(values.length).fill(labels[m]),
        y: values,
        xaxis: xref,
        yaxis: yref,
        marker: { color: colors[i] },
        line: { color: "#666666", width: 1 },
        fillcolor: colors[i],
        boxmean: false,
        boxpoints: false,
        whiskerwidth: 0.8,
        yhoverformat: ".2f",
        hovertemplate: `${labels[m]}<br>min: %{min:.2f}<br>q1: %{q1:.2f}<br>median: %{median:.2f}<br>q3: %{q3:.2f}<br>max: %{max:.2f}<extra></extra>`,
      });
    });

    layout[`xaxis${axisNum === 1 ? "" : axisNum}`] = {
      tickangle: 0,
      showline: true,
      linecolor: "#111111",
      linewidth: 1.5,
      mirror: true,
      categoryorder: "array",
      categoryarray: ["XGB", "MLP", "GP"],
      tickfont: { size: 12 },
      domain: xDomains[idx % 4],
    };

    layout[`yaxis${axisNum === 1 ? "" : axisNum}`] = {
      title: spec.yTitle,
      showgrid: false,
      zeroline: false,
      showline: true,
      linecolor: "#111111",
      linewidth: 1.5,
      mirror: true,
      hoverformat: ".2f",
      range: spec.range,
      tickfont: { size: 12 },
      titlefont: { size: 14 },
      title_standoff: 8,
      domain: idx < 4 ? yDomainTop : yDomainBottom,
    };

    const xDomainRef = axisNum === 1 ? "x domain" : `x${axisNum} domain`;
    const yDomainRef = axisNum === 1 ? "y domain" : `y${axisNum} domain`;

    layout.annotations.push({
      text: `<b>${spec.title}</b>`,
      xref: xDomainRef,
      yref: yDomainRef,
      x: 0.5,
      y: 1.01,
      showarrow: false,
      font: { size: 13, color: "#111111" },
      xanchor: "center",
      yanchor: "bottom",
    });
  });

  // Hide the empty 8th subplot cell to mirror main.py's blank panel.
  layout.xaxis8 = {
    visible: false,
  };
  layout.yaxis8 = {
    visible: false,
  };

  Plotly.newPlot("regression-metrics-subplots", traces, layout, {
    responsive: true,
    displayModeBar: false,
  });

  const outlierNote = document.getElementById("regression-metrics-outlier-note");
  if (outlierNote) {
    const metricSummary = specs
      .map((spec) => `${spec.noteLabel}: ${excludedTargetsByMetric.get(spec.sourceKey).size}`)
      .join("; ");
    outlierNote.textContent = `Outlier note: ${excludedTargets.size} of ${commonTargets.size} shared target variables are excluded from at least one boxplot by the 1.5×IQR filter. Per metric, excluded shared targets are ${metricSummary}.`;
  }

  function pearsonCorrelation(a, b) {
    if (!a.length || !b.length || a.length !== b.length) return null;
    const n = a.length;
    const meanA = a.reduce((s, v) => s + v, 0) / n;
    const meanB = b.reduce((s, v) => s + v, 0) / n;

    let cov = 0;
    let varA = 0;
    let varB = 0;
    for (let i = 0; i < n; i += 1) {
      const da = a[i] - meanA;
      const db = b[i] - meanB;
      cov += da * db;
      varA += da * da;
      varB += db * db;
    }
    if (varA <= 0 || varB <= 0) return null;
    return cov / Math.sqrt(varA * varB);
  }

  function alignedMetricValues(rowsA, rowsB, key) {
    const mapA = new Map();
    for (const row of rowsA) {
      const target = normalizeTargetName(row.target_variable);
      const value = toNum(row[key]);
      if (!target || value === null) continue;
      mapA.set(target, value);
    }

    const valsA = [];
    const valsB = [];
    for (const row of rowsB) {
      const target = normalizeTargetName(row.target_variable);
      const valueB = toNum(row[key]);
      if (!target || valueB === null || !mapA.has(target)) continue;
      valsA.push(mapA.get(target));
      valsB.push(valueB);
    }
    return [valsA, valsB];
  }

  const pairs = [
    ["XGB_single", "AE_MLP"],
    ["XGB_single", "AE_GaussianProcess"],
    ["AE_MLP", "AE_GaussianProcess"],
  ];
  const corrMetrics = [
    { key: "nrmse", label: "Corr NRMSE" },
    { key: "nmae", label: "Corr NMAE" },
    { key: "RTA1pct", label: "Corr RTA1%" },
    { key: "RTA5pct", label: "Corr RTA5%" },
    { key: "RTA10pct", label: "Corr RTA10%" },
  ];

  const corrRows = pairs.map(([a, b]) => {
    const row = {
      pair: `${labels[a]} vs ${labels[b]}`,
      correlations: {},
    };

    const rowCorrValues = [];
    corrMetrics.forEach((metric) => {
      const [valsA, valsB] = alignedMetricValues(methodRows[a], methodRows[b], metric.key);
      const corr = pearsonCorrelation(valsA, valsB);
      row.correlations[metric.key] = corr;
      if (corr !== null) rowCorrValues.push(corr);
    });

    row.average = rowCorrValues.length ? mean(rowCorrValues) : null;
    return row;
  });

  const metricAverages = Object.fromEntries(
    corrMetrics.map((metric) => {
      const values = corrRows.map((r) => r.correlations[metric.key]).filter((v) => v !== null);
      return [metric.key, values.length ? mean(values) : null];
    })
  );
  const overallAverageValues = Object.values(metricAverages).filter((v) => v !== null);
  const overallAverage = overallAverageValues.length ? mean(overallAverageValues) : null;

  const corrTable = document.getElementById("regression-correlation-table");
  if (corrTable) {
    corrTable.innerHTML = [
      "<thead><tr><th>Method Combination</th><th>Corr NRMSE</th><th>Corr NMAE</th><th>Corr RTA1%</th><th>Corr RTA5%</th><th>Corr RTA10%</th><th>Pair Average</th></tr></thead>",
      "<tbody>",
      ...corrRows.map((r) => {
        const nrmse = r.correlations.nrmse;
        const nmae = r.correlations.nmae;
        const rta1 = r.correlations.RTA1pct;
        const rta5 = r.correlations.RTA5pct;
        const rta10 = r.correlations.RTA10pct;
        return `<tr><td>${escapeHTML(r.pair)}</td><td>${nrmse === null ? "-" : fmt(nrmse, 3)}</td><td>${nmae === null ? "-" : fmt(nmae, 3)}</td><td>${rta1 === null ? "-" : fmt(rta1, 3)}</td><td>${rta5 === null ? "-" : fmt(rta5, 3)}</td><td>${rta10 === null ? "-" : fmt(rta10, 3)}</td><td>${r.average === null ? "-" : fmt(r.average, 3)}</td></tr>`;
      }),
      `<tr><td><strong>Average Across Combinations</strong></td><td><strong>${metricAverages.nrmse === null ? "-" : fmt(metricAverages.nrmse, 3)}</strong></td><td><strong>${metricAverages.nmae === null ? "-" : fmt(metricAverages.nmae, 3)}</strong></td><td><strong>${metricAverages.RTA1pct === null ? "-" : fmt(metricAverages.RTA1pct, 3)}</strong></td><td><strong>${metricAverages.RTA5pct === null ? "-" : fmt(metricAverages.RTA5pct, 3)}</strong></td><td><strong>${metricAverages.RTA10pct === null ? "-" : fmt(metricAverages.RTA10pct, 3)}</strong></td><td><strong>${overallAverage === null ? "-" : fmt(overallAverage, 3)}</strong></td></tr>`,
      "</tbody>",
    ].join("");
  }

  renderFoldSensitivityTable(regRows);
  renderBoundarySensitivityTable(dashboardSourceRows.classifier);

  const violationTable = document.getElementById("regression-violations-table");
  if (violationTable) {
    const methodsFromPayload = Array.isArray(violationsPayload?.methods) ? violationsPayload.methods : [];

    if (!methodsFromPayload.length) {
      violationTable.innerHTML = "<tbody><tr><td>Constraint violation summary data not available.</td></tr></tbody>";
    } else {
      violationTable.innerHTML = [
        "<thead><tr><th>Method</th><th>Tech-Period Pairs</th><th>Constraint Violations [%]</th></tr></thead>",
        "<tbody>",
        ...methodsFromPayload.map(
          (r) => `<tr><td>${escapeHTML(r.method)}</td><td>${Number(r.pair_count || 0)}</td><td>${r.constraint_violations_pct === null || r.constraint_violations_pct === undefined ? "-" : fmt(Number(r.constraint_violations_pct), 2)}</td></tr>`
        ),
        "</tbody>",
      ].join("");
    }
  }

  const violationTechnologyTable = document.getElementById("regression-violation-technologies-table");
  const violationTechnologyAverages = document.getElementById("regression-violation-technology-averages");
  const violationTechnologyFilter = document.getElementById("regression-violation-technology-filter");
  const violationTechnologySearch = document.getElementById("regression-violation-technology-search");
  const violationTechnologyNote = document.getElementById("regression-violation-technology-note");
  if (violationTechnologyTable && violationTechnologyAverages && violationTechnologyFilter && violationTechnologySearch && violationTechnologyNote) {
    const technologyNameMap = new Map(
      (Array.isArray(trainingPayload?.targets) ? trainingPayload.targets : [])
        .map((target) => [String(target.technology_id || "").trim(), String(target.technology_name || "").trim()])
        .filter(([technologyId, technologyName]) => technologyId && technologyName)
    );
    const comparableRows = Array.isArray(violationsPayload?.comparable_pair_rows)
      ? violationsPayload.comparable_pair_rows
      : [];
    const comparableMethodRows = (Array.isArray(violationsPayload?.methods) ? violationsPayload.methods : []).filter((row) =>
      ["XGB", "XGB+classifier", "informed_classifier"].includes(String(row.method || ""))
    );

    const formatViolationCell = (entry) => {
      if (!entry || entry.constraint_violations_pct === null || entry.constraint_violations_pct === undefined) return "-";
      if (!entry.violates_constraint) return "";
      return `Violation (${fmt(Number(entry.constraint_violations_pct), 2)}%)`;
    };

    const matchesViolationFilter = (row, filterValue) => {
      if (filterValue === "all") return true;
      if (filterValue === "any") return Boolean(row.XGB?.violates_constraint || row["XGB+classifier"]?.violates_constraint || row.informed_classifier?.violates_constraint);
      if (filterValue === "xgb") return Boolean(row.XGB?.violates_constraint);
      if (filterValue === "xgbclf") return Boolean(row["XGB+classifier"]?.violates_constraint);
      if (filterValue === "informed") return Boolean(row.informed_classifier?.violates_constraint);
      if (filterValue === "clean") return !row.XGB?.violates_constraint && !row["XGB+classifier"]?.violates_constraint && !row.informed_classifier?.violates_constraint;
      return true;
    };

    const renderViolationTechnologyTable = () => {
      const filterValue = violationTechnologyFilter.value || "all";
      const query = violationTechnologySearch.value.trim().toLowerCase();
      const filteredRows = comparableRows.filter((row) => {
        const technology = String(row.technology || "").toLowerCase();
        const textMatch = !query || technology.includes(query);
        return textMatch && matchesViolationFilter(row, filterValue);
      });

      violationTechnologyNote.textContent = `Showing ${filteredRows.length} of ${comparableRows.length} technologies.`;

      violationTechnologyTable.innerHTML = [
        "<thead><tr><th>Technology</th><th>XGB</th><th>XGB+classifier</th><th>informed_classifier</th></tr></thead>",
        "<tbody>",
        ...filteredRows.map((row) => {
          const technologyId = String(row.technology || "");
          const technologyLabel = technologyNameMap.get(technologyId) || technologyId;
          return `<tr><td>${escapeHTML(technologyLabel)}</td><td>${formatViolationCell(row.XGB)}</td><td>${formatViolationCell(row["XGB+classifier"])}</td><td>${formatViolationCell(row.informed_classifier)}</td></tr>`;
        }),
        "</tbody>",
      ].join("");
    };

    if (!comparableRows.length) {
      violationTechnologyTable.innerHTML = "<tbody><tr><td>Per-technology constraint violation data not available.</td></tr></tbody>";
      violationTechnologyAverages.innerHTML = "";
      violationTechnologyNote.textContent = "";
    } else {
      violationTechnologyAverages.innerHTML = comparableMethodRows
        .map(
          (row) => `<div class="card"><h3>${escapeHTML(String(row.method || ""))}</h3><div class="metric">Avg violation: ${row.constraint_violations_pct === null || row.constraint_violations_pct === undefined ? "-" : `${fmt(Number(row.constraint_violations_pct), 2)}%`}</div></div>`
        )
        .join("");

      violationTechnologyFilter.addEventListener("change", renderViolationTechnologyTable);
      violationTechnologySearch.addEventListener("input", renderViolationTechnologyTable);
      renderViolationTechnologyTable();
    }
  }
}

function buildHeatmapTable(rows, metricCol) {
  const useRows = newestByTarget(rows).map((r) => ({
    ...r,
    variable_group: extractVariableGroup(r.target_variable),
    data_type: classifyDataType(r),
  }));

  const groups = [...Object.keys(VARIABLE_GROUPS), "Total"];
  const cols = ["Continuous", "Semi-continuous", "Discrete", "Total"];

  function groupFilter(group, col) {
    let subset = useRows;
    if (group !== "Total") subset = subset.filter((r) => r.variable_group === group);
    if (col !== "Total") subset = subset.filter((r) => r.data_type === col);
    return subset;
  }

  function metricValue(subset) {
    const values = subset.map((r) => toNum(r[metricCol])).filter((v) => v !== null);
    if (!values.length) return null;
    const m = mean(values);
    if (["RTA1pct", "RTA5pct", "RTA10pct"].includes(metricCol)) {
      return m * 100;
    }
    return m;
  }

  function annotation(subset) {
    const values = subset.map((r) => toNum(r[metricCol])).filter((v) => v !== null);
    if (!values.length) return "";
    const m = mean(values);
    if (["RTA1pct", "RTA5pct", "RTA10pct"].includes(metricCol)) {
      return `${(m * 100).toFixed(2)} (${values.length})`;
    }
    return `${m.toFixed(2)} (${values.length})`;
  }

  const z = groups.map((g) => cols.map((c) => metricValue(groupFilter(g, c))));
  const text = groups.map((g) => cols.map((c) => annotation(groupFilter(g, c))));
  return { groups, cols, z, text };
}

function renderValueDistributionAccuracyFigure(payload) {
  const chartId = "value-distribution-accuracy-chart";
  const noteEl = document.getElementById("value-distribution-accuracy-note");
  const chartEl = document.getElementById(chartId);
  if (!chartEl) return;

  const categories = Array.isArray(payload?.categories) ? payload.categories : [];
  if (!categories.length) {
    if (noteEl) {
      noteEl.textContent = "Value-distribution summary data is not available.";
    }
    Plotly.newPlot(
      chartId,
      [],
      baseLayout({
        height: 430,
        annotations: [
          {
            x: 0.5,
            y: 0.5,
            xref: "paper",
            yref: "paper",
            text: "No value-distribution metrics available",
            showarrow: false,
            font: { size: 16, color: "#667085" },
          },
        ],
      }),
      { responsive: true, displayModeBar: false }
    );
    return;
  }

  const order = payload?.order || ["Normal", "Skewed", "Uniform", "Multi regime"];
  const byName = new Map(categories.map((c) => [String(c.name || ""), c]));
  const ordered = order.map((name) => byName.get(name)).filter(Boolean);

  const x = ordered.map((c) => c.name);
  const nrmse = ordered.map((c) => toNum(c.nrmse));
  const nmae = ordered.map((c) => toNum(c.nmae));
  const rta1 = ordered.map((c) => toNum(c.rta1_pct));
  const rta5 = ordered.map((c) => toNum(c.rta5_pct));
  const rta10 = ordered.map((c) => toNum(c.rta10_pct));
  const counts = ordered.map((c) => Number(c.n || 0));

  const leftMax = Math.max(...nrmse.filter(Number.isFinite), ...nmae.filter(Number.isFinite), 0.05);
  const rightMax = Math.max(...rta1.filter(Number.isFinite), ...rta5.filter(Number.isFinite), ...rta10.filter(Number.isFinite), 100);

  const traces = [
    {
      type: "bar",
      name: "NRMSE",
      x,
      y: nrmse,
      marker: { color: "#4c72b0" },
      yaxis: "y",
      offsetgroup: "nrmse",
      hovertemplate: "%{x}<br>NRMSE: %{y:.3f}<extra></extra>",
    },
    {
      type: "bar",
      name: "NMAE",
      x,
      y: nmae,
      marker: { color: "#dd8452" },
      yaxis: "y",
      offsetgroup: "nmae",
      hovertemplate: "%{x}<br>NMAE: %{y:.3f}<extra></extra>",
    },
    {
      type: "bar",
      name: "RTA1%",
      x,
      y: rta1,
      marker: { color: "#7bc87c" },
      yaxis: "y2",
      offsetgroup: "rta1",
      hovertemplate: "%{x}<br>RTA1%%: %{y:.2f}<extra></extra>",
    },
    {
      type: "bar",
      name: "RTA5%",
      x,
      y: rta5,
      marker: { color: "#2f8f57" },
      yaxis: "y2",
      offsetgroup: "rta5",
      hovertemplate: "%{x}<br>RTA5%%: %{y:.2f}<extra></extra>",
    },
    {
      type: "bar",
      name: "RTA10%",
      x,
      y: rta10,
      marker: { color: "#1b5e20" },
      yaxis: "y2",
      offsetgroup: "rta10",
      hovertemplate: "%{x}<br>RTA10%%: %{y:.2f}<extra></extra>",
    },
  ];

  const annotations = x.map((name, idx) => ({
    x: name,
    y: rightMax * 1.03,
    xref: "x",
    yref: "y2",
    text: `n=${counts[idx]}`,
    showarrow: false,
    font: { size: 12, color: "#303030" },
  }));

  Plotly.newPlot(
    chartId,
    traces,
    baseLayout({
      barmode: "group",
      height: 470,
      margin: { t: 30, r: 64, l: 68, b: 72 },
      xaxis: { title: "Data distribution type", tickangle: -15, showgrid: false },
      yaxis: { title: "NRMSE / NMAE", range: [0, leftMax * 1.25], zeroline: false, gridcolor: "#ececec" },
      yaxis2: {
        title: "RTA [%]",
        overlaying: "y",
        side: "right",
        range: [0, rightMax * 1.12],
        zeroline: false,
        showgrid: false,
      },
      legend: { orientation: "h", y: 1.16, x: 0 },
      annotations,
    }),
    { responsive: true, displayModeBar: false }
  );

  if (noteEl) {
    const included = ordered.map((c) => `${c.name} (n=${c.n})`).join(", ");
    const ruleText = payload?.category_rules
      ? ` Category mapping: ${payload.category_rules}.`
      : "";
    noteEl.textContent = `Included target variables by distribution: ${included}.${ruleText}`;
  }
}

function initClassifierCorrelationMatrixSection(clsRows) {
  const tableEl = document.getElementById("classifier-correlation-matrix-table");
  const noteEl = document.getElementById("classifier-correlation-matrix-note");
  const filterEl = document.getElementById("corr-data-type-filter");
  if (!tableEl || !noteEl || !filterEl) return;

  // Table 16 targets the classifier-regressor model (threshold 20 baseline).
  const baseRows = dedupeRowsByTarget(
    (clsRows || []).filter((r) => {
      const method = String(r.surrogate_method || "").toLowerCase();
      return method === "xgb_hybrid_tuned" || method === "xgb_hybrid20_tuned";
    })
  );

  const columns = [
    { key: "nrmse", label: "nrmse" },
    { key: "nmae", label: "nmae" },
    { key: "RTA1pct", label: "RTA1pct" },
    { key: "RTA5pct", label: "RTA5pct" },
    { key: "RTA10pct", label: "RTA10pct" },
    { key: "training_time_s", label: "Training time" },
    { key: "running_time_s", label: "Running time" },
    { key: "std_target", label: "Standard deviation" },
    { key: "REG_cat", label: "# continuous values" },
    { key: "CLAS_cat", label: "# discrete values" },
    { key: "classifier_accuracy", label: "Classifier accuracy" },
  ];

  function getMetricValue(row, key) {
    // Keep Table 16 consistent with the paper table: use classifier_accuracy only.
    if (key === "classifier_accuracy") {
      return toNum(row.classifier_accuracy);
    }
    return toNum(row[key]);
  }

  function pearsonForRows(rows, keyA, keyB) {
    const pairs = [];
    for (const row of rows) {
      const a = getMetricValue(row, keyA);
      const b = getMetricValue(row, keyB);
      if (a === null || b === null) continue;
      pairs.push([a, b]);
    }
    if (pairs.length < 2) return null;

    const xs = pairs.map((p) => p[0]);
    const ys = pairs.map((p) => p[1]);
    const mx = mean(xs);
    const my = mean(ys);
    if (mx === null || my === null) return null;

    let cov = 0;
    let varX = 0;
    let varY = 0;
    for (let i = 0; i < pairs.length; i += 1) {
      const dx = xs[i] - mx;
      const dy = ys[i] - my;
      cov += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }
    if (varX <= 0 || varY <= 0) return null;
    return cov / Math.sqrt(varX * varY);
  }

  function fmtCorr(v) {
    if (!Number.isFinite(v)) return "-";
    if (Math.abs(v - 1) < 1e-12) return "1";
    return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }

  function applyDataTypeFilter(rows, mode) {
    if (mode === "all") return rows;
    return rows.filter((row) => {
      let type = "Unknown";
      const nCategories = toNum(row.n_categories);
      if (nCategories !== null) {
        const n = Math.trunc(nCategories);
        if (n === 0) type = "Continuous";
        else if (n <= 20) type = "Semi-continuous";
        else type = "Discrete";
      } else {
        // Fallback if n_categories is unavailable.
        type = classifyDataType(row);
      }
      if (mode === "continuous") return type === "Continuous";
      if (mode === "semi") return type === "Semi-continuous";
      if (mode === "discrete") return type === "Discrete";
      return true;
    });
  }

  function render() {
    const mode = filterEl.value || "all";
    const filteredRows = applyDataTypeFilter(baseRows, mode);
    const hideClassifierAccuracy = mode === "continuous";

    const headerCells = [`<th></th>`, ...columns.map((c) => `<th>${escapeHTML(c.label)}</th>`)].join("");
    const bodyRows = columns.map((rowCol, rowIdx) => {
      const cells = columns
        .map((col, colIdx) => {
          if (hideClassifierAccuracy && (rowCol.key === "classifier_accuracy" || col.key === "classifier_accuracy")) {
            return "<td></td>";
          }
          if (colIdx > rowIdx) return "<td></td>";
          if (colIdx === rowIdx) return "<td>1</td>";
          const corr = pearsonForRows(filteredRows, rowCol.key, col.key);
          return `<td>${fmtCorr(corr)}</td>`;
        })
        .join("");
      return `<tr><th>${escapeHTML(rowCol.label)}</th>${cells}</tr>`;
    });

    tableEl.innerHTML = [
      `<thead><tr>${headerCells}</tr></thead>`,
      "<tbody>",
      ...bodyRows,
      "</tbody>",
    ].join("");

    const modeLabel = mode === "all" ? "all target variables" : mode === "continuous" ? "continuous only" : mode === "semi" ? "semi-continuous only" : "discrete only";
    noteEl.textContent = hideClassifierAccuracy
      ? `Rows included: ${filteredRows.length} (${modeLabel}) from classifier-regressor baseline (XGB_HYBRID_tuned). Classifier accuracy is shown as not applicable for this filter.`
      : `Rows included: ${filteredRows.length} (${modeLabel}) from classifier-regressor baseline (XGB_HYBRID_tuned).`;
  }

  filterEl.addEventListener("change", render);
  render();
}

function renderClassifierSection(clsRows) {
  const methodSelect = document.getElementById("classifier-method-filter");
  const metricSelect = document.getElementById("heatmap-metric-filter");
  const rtaStackedNote = document.getElementById("classifier-rta-stacked-note");
  const classifierAccuracyNote = document.getElementById("classifier-accuracy-bar-note");
  const rtaOrder = [
    "Commodity prices",
    "Electricity system capacity",
    "Electricity system generation",
    "Emissions",
    "Energy carriers imports",
    "Energy carriers production",
    "Flexibility and storage",
    "Technology capacity",
    "Technology operation",
    "Total system costs",
  ];
  const methods = [...new Set(clsRows.map((r) => r.surrogate_method).filter(Boolean))].sort();

  methodSelect.innerHTML = methods.map((m) => `<option value="${m}" ${m.includes("XGB_HYBRID_tuned") ? "selected" : ""}>${m}</option>`).join("");

  function rerender() {
    const selectedMethod = methodSelect.value;
    const metricCol = metricSelect.value;
    const rows = clsRows.filter((r) => r.surrogate_method === selectedMethod);
    const strictRows = rows.filter((r) => {
      const regCat = toNum(r.REG_cat);
      const clasCat = toNum(r.CLAS_cat);
      const isDiscrete = regCat !== null && regCat === 0;
      const isSemiContinuous = regCat !== null && clasCat !== null && regCat > 0 && clasCat > 0;
      return isDiscrete || isSemiContinuous;
    });
    currentClassifierRows = rows;

    const hm = buildHeatmapTable(rows, metricCol);
    const totalColIdx = hm.cols.indexOf("Total");
    const totalRowIdx = hm.groups.indexOf("Total");
    Plotly.newPlot(
      "classifier-heatmap",
      [
        {
          type: "heatmap",
          x: hm.cols,
          y: hm.groups,
          z: hm.z,
          text: hm.text,
          texttemplate: "%{text}",
          hovertemplate: "<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
          colorscale: metricCol.startsWith("RTA")
            ? [
                [0, "#f2f7d9"],
                [0.5, "#b7d98a"],
                [1, "#5f8f37"],
              ]
            : [
                [0, "#f7e7ad"],
                [0.45, "#f6a043"],
                [1, "#ef2f24"],
              ],
          xgap: 0,
          ygap: 0,
          colorbar: { title: metricCol.startsWith("RTA") ? `${metricCol} (%)` : metricCol.toUpperCase(), thickness: 22 },
          textfont: { color: "#222222", size: 14 },
        },
      ],
      baseLayout({
        margin: { t: 28, r: 50, l: 235, b: 62 },
        xaxis: { title: "Data Type", side: "bottom", tickfont: { size: 16 }, showgrid: false },
        yaxis: { title: "Variable Type", autorange: "reversed", tickfont: { size: 15 }, showgrid: false },
        shapes: [
          {
            type: "line",
            xref: "x",
            yref: "paper",
            x0: totalColIdx - 0.5,
            x1: totalColIdx - 0.5,
            y0: 0,
            y1: 1,
            line: { color: "#000000", width: 3 },
          },
          {
            type: "line",
            xref: "paper",
            yref: "y",
            x0: 0,
            x1: 1,
            y0: totalRowIdx - 0.5,
            y1: totalRowIdx - 0.5,
            line: { color: "#000000", width: 3 },
          },
          {
            type: "rect",
            xref: "paper",
            yref: "paper",
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
            line: { color: "#000000", width: 1.8 },
            fillcolor: "rgba(0,0,0,0)",
          },
        ],
      }),
      { responsive: true, displayModeBar: false }
    );

    const grouped = {};
    strictRows.forEach((r) => {
      const g = extractVariableGroup(r.target_variable);
      if (!grouped[g]) grouped[g] = [];
      const acc = toNum(r.classifier_accuracy);
      if (acc !== null) {
        grouped[g].push(acc);
      }
    });

    const presentGroups = Object.keys(grouped).filter((g) => grouped[g].length > 0);
    const orderedGroups = [
      ...rtaOrder.filter((g) => presentGroups.includes(g)),
      ...presentGroups.filter((g) => !rtaOrder.includes(g)).sort(),
    ];

    const x = orderedGroups;
    const y = x.map((k) => mean(grouped[k]));
    const counts = x.map((k) => grouped[k].length);
    const countLabels = counts.map((n) => `n=${n}`);

    Plotly.newPlot(
      "classifier-accuracy-bar",
      [
        {
          type: "bar",
          x,
          y,
          marker: { color: "#2a6f97" },
          text: countLabels,
          textposition: "outside",
          cliponaxis: false,
          customdata: counts,
          hovertemplate: "%{x}<br>Classifier Accuracy: %{y:.3f}<br>Target variables: %{customdata}<extra></extra>",
        },
      ],
      baseLayout({
        margin: { t: 26, r: 12, l: 56, b: 135 },
        yaxis: { title: "Classifier Accuracy", gridcolor: "#ececec", zeroline: false },
      }),
      { responsive: true, displayModeBar: false }
    );

    if (classifierAccuracyNote) {
      const details = x.map((group, idx) => `${group} (n=${counts[idx]})`).join(", ");
      classifierAccuracyNote.textContent = `Included target variables per variable type (discrete/semi-continuous only): ${details}.`;
    }

    const rtaByGroup = new Map();
    const includedTargets = new Set();
    rows.forEach((r) => {
      const group = extractVariableGroup(r.target_variable);
      const r1 = toNum(r.RTA1pct);
      const r5 = toNum(r.RTA5pct);
      const r10 = toNum(r.RTA10pct);
      if (r1 === null || r5 === null || r10 === null) return;
      const targetKey = normalizeTargetName(r.target_variable);
      if (targetKey) includedTargets.add(targetKey);
      if (!rtaByGroup.has(group)) {
        rtaByGroup.set(group, { r1: [], r5: [], r10: [] });
      }
      const bucket = rtaByGroup.get(group);
      bucket.r1.push(r1 * 100);
      bucket.r5.push(r5 * 100);
      bucket.r10.push(r10 * 100);
    });

    const orderedRtaGroups = [
      ...rtaOrder,
      ...[...rtaByGroup.keys()].filter((g) => !rtaOrder.includes(g)).sort(),
    ];
    const segWithin1 = [];
    const seg1to5 = [];
    const seg5to10 = [];

    orderedRtaGroups.forEach((group) => {
      const bucket = rtaByGroup.get(group);
      if (!bucket || !bucket.r1.length || !bucket.r5.length || !bucket.r10.length) {
        segWithin1.push(0);
        seg1to5.push(0);
        seg5to10.push(0);
        return;
      }
      const r1 = Math.max(0, Math.min(100, mean(bucket.r1)));
      const r5 = Math.max(r1, Math.min(100, mean(bucket.r5)));
      const r10 = Math.max(r5, Math.min(100, mean(bucket.r10)));
      segWithin1.push(r1);
      seg1to5.push(r5 - r1);
      seg5to10.push(r10 - r5);
    });

    Plotly.newPlot(
      "classifier-rta-stacked",
      [
        {
          type: "bar",
          name: "RTA1%",
          x: orderedRtaGroups,
          y: segWithin1,
          marker: { color: "#2e7d32" },
          hovertemplate: "%{x}<br>RTA1%: %{y:.2f}%<extra></extra>",
        },
        {
          type: "bar",
          name: "RTA5%-RTA1%",
          x: orderedRtaGroups,
          y: seg1to5,
          marker: { color: "#d32f2f" },
          hovertemplate: "%{x}<br>1%-5% interval: %{y:.2f}%<extra></extra>",
        },
        {
          type: "bar",
          name: "RTA10%-RTA5%",
          x: orderedRtaGroups,
          y: seg5to10,
          marker: { color: "#6a1b9a" },
          hovertemplate: "%{x}<br>5%-10% interval: %{y:.2f}%<extra></extra>",
        },
      ],
      baseLayout({
        barmode: "stack",
        margin: { t: 26, r: 16, l: 68, b: 145 },
        xaxis: { title: "Variable Type", tickangle: -35 },
        yaxis: { title: "Percentage (%)", range: [0, 100], gridcolor: "#ececec", zeroline: false },
        legend: { orientation: "h", y: 1.15, x: 0 },
      }),
      { responsive: true, displayModeBar: false }
    );

    if (rtaStackedNote) {
      rtaStackedNote.textContent = `Included target variables: ${includedTargets.size} (all data types, with valid RTA1%, RTA5%, and RTA10%).`;
    }
  }

  methodSelect.addEventListener("change", rerender);
  metricSelect.addEventListener("change", rerender);
  rerender();
}

function renderJointSection(classifierRows, jointRows) {
  const clsThreshold20Rows = classifierRows.filter((r) => {
    const method = String(r.surrogate_method || "").toLowerCase();
    if (method === "xgb_hybrid20_tuned" || method === "xgb_hybrid_tuned") {
      return true;
    }

    const hp = String(r.hyperparameters || "").toLowerCase();
    return method.includes("xgb_hybrid") && hp.includes("categories_threshold") && /categories_threshold['\"]?\s*:\s*20/.test(hp);
  });

  const informedRows = jointRows.filter((r) => String(r.surrogate_method || "") === "Informed_XGB_HYBRID");

  const clsTargets = new Set(clsThreshold20Rows.map((r) => normalizeTargetName(r.target_variable)).filter(Boolean));
  const informedTargets = new Set(informedRows.map((r) => normalizeTargetName(r.target_variable)).filter(Boolean));
  const pairCommonTargets = getPairIntersection(clsTargets, informedTargets);

  const methods = [
    {
      label: "XGB+classifier",
      rows: clsThreshold20Rows.filter((r) => pairCommonTargets.has(normalizeTargetName(r.target_variable))),
      color: "#fcbf49",
    },
    {
      label: "informed_classifier",
      rows: informedRows.filter((r) => pairCommonTargets.has(normalizeTargetName(r.target_variable))),
      color: "#6a994e",
    },
  ];

  const specs = [
    { key: "nrmse", title: "NRMSE Distribution", yTitle: "NRMSE", scale: 1, decimals: 3 },
    { key: "nmae", title: "NMAE Distribution", yTitle: "NMAE", scale: 1, decimals: 3 },
    { key: "RTA1pct", title: "1% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { key: "RTA5pct", title: "5% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { key: "RTA10pct", title: "10% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    {
      key: "classifier_accuracy_train",
      fallbackKey: "classifier_accuracy",
      title: "Classifier Accuracy Distribution",
      yTitle: "Classifier Accuracy",
      scale: 1,
      decimals: 3,
      range: [0, 1],
    },
  ];

  const cols = 3;
  const rowsN = 2;
  const layout = baseLayout({
    showlegend: false,
    height: 700,
    margin: { t: 56, r: 20, l: 44, b: 36 },
    grid: { rows: rowsN, columns: cols, pattern: "independent" },
    annotations: [],
  });

  const traces = [];

  specs.forEach((spec, idx) => {
    const axisNum = idx + 1;
    const xaxisRef = axisNum === 1 ? "x" : `x${axisNum}`;
    const yaxisRef = axisNum === 1 ? "y" : `y${axisNum}`;

    methods.forEach((method) => {
      const values = (method.rows || [])
        .map((r) => {
          const primary = toNum(r[spec.key]);
          if (primary !== null) return primary;
          return spec.fallbackKey ? toNum(r[spec.fallbackKey]) : null;
        })
        .filter((v) => v !== null)
        .map((v) => v * spec.scale);

      traces.push({
        type: "box",
        name: method.label,
        x: Array(values.length).fill(method.label),
        y: values,
        xaxis: xaxisRef,
        yaxis: yaxisRef,
        marker: { color: method.color },
        line: { color: "#5b5b5b", width: 1 },
        fillcolor: method.color,
        boxmean: false,
        boxpoints: false,
        yhoverformat: ".2f",
        hovertemplate: `${method.label}<br>min: %{min:.2f}<br>q1: %{q1:.2f}<br>median: %{median:.2f}<br>q3: %{q3:.2f}<br>max: %{max:.2f}<extra></extra>`,
      });
    });

    layout[`xaxis${axisNum === 1 ? "" : axisNum}`] = {
      showline: true,
      linecolor: "#111111",
      linewidth: 1.2,
      tickfont: { size: 12 },
      categoryorder: "array",
      categoryarray: methods.map((m) => m.label),
    };

    layout[`yaxis${axisNum === 1 ? "" : axisNum}`] = {
      title: spec.yTitle,
      showgrid: false,
      zeroline: false,
      showline: true,
      linecolor: "#111111",
      linewidth: 1.2,
      hoverformat: ".2f",
      tickfont: { size: 12 },
      range: spec.range,
    };

    const xDomainRef = axisNum === 1 ? "x domain" : `x${axisNum} domain`;
    const yDomainRef = axisNum === 1 ? "y domain" : `y${axisNum} domain`;
    layout.annotations.push({
      text: `<b>${spec.title}</b>`,
      xref: xDomainRef,
      yref: yDomainRef,
      x: 0.5,
      y: 1.02,
      showarrow: false,
      font: { size: 15, color: "#1f2a37" },
    });
  });

  Plotly.newPlot("joint-metrics-boxplots", traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

const CONFUSION_METHODS = [
  { key: "XGB_HYBRID10_tuned", label: "Threshold 10" },
  { key: "XGB_HYBRID_tuned", label: "Threshold 20" },
  { key: "XGB_HYBRID30_tuned", label: "Threshold 30" },
  { key: "XGB_HYBRID40_tuned", label: "Threshold 40" },
];

const CONFUSION_ERROR_TYPES = [
  { key: "NR", label: "Non-recurring value assigned to a recurring value" },
  { key: "ZN", label: "Zero predicted as non-recurring" },
  { key: "RN", label: "Recurring value predicted as non-recurring" },
  { key: "RR", label: "Confusion between non-zero recurring values" },
  { key: "AZ", label: "Active value predicted as zero" },
  { key: "ZR", label: "Zero predicted as active (non-zero recurring)" },
];

function buildConfusionStore(rows) {
  const store = {};
  rows.forEach((row) => {
    const count = toNum(row.count);
    const actual = toNum(row.actual_class);
    const predicted = toNum(row.predicted_class);
    if (count === null || actual === null || predicted === null) return;
    const methodStore = (store[row.method] ??= {});
    const target = (methodStore[row.target_variable] ??= { cells: [], labels: new Map(), maxClass: 0 });
    target.cells.push({ actual, predicted, count });
    target.labels.set(actual, String(row.actual_recurring_value));
    target.maxClass = Math.max(target.maxClass, actual, predicted);
  });
  return store;
}

function confusionKind(target, classIdx) {
  if (target.labels.get(classIdx) === "0.0" && classIdx === 0) return "Z";
  if (classIdx === target.maxClass && target.labels.get(classIdx) === "REST") return "N";
  return "R";
}

function confusionClassLabel(target, classIdx) {
  const raw = target.labels.get(classIdx);
  if (raw === undefined) return `${classIdx}: class ${classIdx}`;
  if (raw === "REST") return `${classIdx}: REST`;
  const num = toNum(raw);
  return `${classIdx}: ${num === null ? raw : Number(num.toPrecision(5))}`;
}

function summarizeConfusionErrors(methodStore) {
  const counts = Object.fromEntries(CONFUSION_ERROR_TYPES.map((t) => [t.key, 0]));
  let total = 0;
  Object.values(methodStore || {}).forEach((target) => {
    target.cells.forEach((cell) => {
      if (cell.actual === cell.predicted) return;
      const a = confusionKind(target, cell.actual);
      const p = confusionKind(target, cell.predicted);
      const key = p === "Z" ? "AZ" : a + p;
      if (key in counts) {
        counts[key] += cell.count;
        total += cell.count;
      }
    });
  });
  return { counts, total };
}

function initConfusionMatrixSection(rows) {
  const methodSelect = document.getElementById("confusion-method-select");
  const targetSelect = document.getElementById("confusion-target-select");
  const normalizeSelect = document.getElementById("confusion-normalize-select");
  const noteEl = document.getElementById("confusion-summary-note");
  const chartEl = document.getElementById("confusion-matrix-chart");
  const typesTable = document.getElementById("confusion-types-table");
  const typesIntro = document.getElementById("confusion-types-intro");
  if (!methodSelect || !targetSelect || !normalizeSelect || !chartEl || !typesTable) return;

  const store = buildConfusionStore(rows);
  const methods = CONFUSION_METHODS.filter((m) => store[m.key]);
  if (!methods.length) {
    noteEl.textContent = "No confusion matrix data available.";
    return;
  }

  methodSelect.innerHTML = methods
    .map((m) => `<option value="${m.key}"${m.label === "Threshold 20" ? " selected" : ""}>${m.label}</option>`)
    .join("");

  function fillTargets() {
    const previous = targetSelect.value;
    const targets = Object.keys(store[methodSelect.value]).sort((a, b) => a.localeCompare(b));
    targetSelect.innerHTML = targets
      .map((t) => `<option value="${escapeHTML(t)}">${escapeHTML(t)}</option>`)
      .join("");
    if (targets.includes(previous)) targetSelect.value = previous;
  }

  function renderMatrix() {
    const target = store[methodSelect.value]?.[targetSelect.value];
    if (!target) return;
    const n = target.maxClass + 1;
    const matrix = Array.from({ length: n }, () => Array(n).fill(0));
    target.cells.forEach((c) => {
      matrix[c.actual][c.predicted] += c.count;
    });

    const labels = Array.from({ length: n }, (_, i) => confusionClassLabel(target, i));
    const rowNormalized = normalizeSelect.value === "row";
    const total = matrix.flat().reduce((a, b) => a + b, 0);
    const correct = matrix.reduce((sum, row, i) => sum + row[i], 0);
    const z = matrix.map((row) => {
      const rowSum = row.reduce((a, b) => a + b, 0);
      return row.map((v) => (rowNormalized ? (rowSum ? (v / rowSum) * 100 : null) : v));
    });
    const showText = n <= 15;
    const text = z.map((row, i) =>
      row.map((v, j) => {
        if (v === null || matrix[i][j] === 0) return "";
        return rowNormalized ? `${v.toFixed(0)}%` : String(matrix[i][j]);
      })
    );

    noteEl.textContent = `${n} classes, ${total.toLocaleString()} test cases, accuracy ${fmt(total ? (correct / total) * 100 : null, 1)}%.`;

    const size = Math.min(900, Math.max(420, 40 * n + 160));
    chartEl.style.minHeight = `${size}px`;
    Plotly.react(
      chartEl,
      [
        {
          type: "heatmap",
          x: labels,
          y: labels,
          z,
          customdata: matrix,
          text: showText ? text : undefined,
          texttemplate: showText ? "%{text}" : undefined,
          colorscale: "Blues",
          hoverongaps: false,
          hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>Count: %{customdata}<extra></extra>",
          colorbar: { title: { text: rowNormalized ? "% of actual" : "Count" } },
        },
      ],
      baseLayout({
        height: size,
        margin: { t: 20, r: 20, l: 120, b: 120 },
        xaxis: { title: { text: "Predicted class" }, type: "category", tickangle: -45, showticklabels: n <= 40 },
        yaxis: { title: { text: "Actual class" }, type: "category", autorange: "reversed", showticklabels: n <= 40 },
      }),
      { responsive: true, displaylogo: false }
    );
  }

  function renderTypes() {
    const summaries = methods.map((m) => ({ ...m, ...summarizeConfusionErrors(store[m.key]) }));
    typesIntro.textContent = summaries
      .map((s) => `${s.label}: ${s.total.toLocaleString()} misclassified target-scenario cases`)
      .join(" | ");
    const head = `<tr><th>Error type</th>${summaries.map((s) => `<th>${s.label}<br />Share (cases)</th>`).join("")}</tr>`;
    const body = CONFUSION_ERROR_TYPES.map(
      (t) =>
        `<tr><td>${t.label}</td>${summaries
          .map((s) => `<td>${fmt(s.total ? (s.counts[t.key] / s.total) * 100 : null, 1)}% (${s.counts[t.key].toLocaleString()})</td>`)
          .join("")}</tr>`
    ).join("");
    const foot = `<tr><td><b>Total misclassified</b></td>${summaries.map((s) => `<td><b>${s.total.toLocaleString()}</b></td>`).join("")}</tr>`;
    typesTable.innerHTML = `<thead>${head}</thead><tbody>${body}${foot}</tbody>`;
  }

  methodSelect.addEventListener("change", () => {
    fillTargets();
    renderMatrix();
  });
  targetSelect.addEventListener("change", renderMatrix);
  normalizeSelect.addEventListener("change", renderMatrix);

  fillTargets();
  renderMatrix();
  renderTypes();

  document.querySelector('.supplementary-btn[data-supp-target="chapter-confusion"]')?.addEventListener("click", () => {
    window.requestAnimationFrame(() => Plotly.Plots.resize(chartEl));
  });
}

async function main() {
  try {
    const [regRows, clsRows, jointRows, regSheetRows, clsSheetRows, jointSheetRows, trainingPayload, violationsPayload, valueDistributionPayload, confusionRows] = await Promise.all([
      loadCSV("./data/regression_surrogate_run_log.csv"),
      loadCSV("./data/classifier_2fold_performance.csv"),
      loadCSV("./data/joint_learning_informed_classifier_run_log.csv"),
      loadCSV("./data/regression_datasheet.csv"),
      loadCSV("./data/classifier_datasheet.csv"),
      loadCSV("./data/joint_datasheet.csv"),
      loadJSON("./data/training_value_distributions.json").catch(() => ({ targets: [], target_count_with_variation: 0 })),
      loadJSON("./data/regression_constraint_violations.json").catch(() => ({ methods: [], common_pair_count: 0 })),
      loadJSON("./data/value_distribution_metrics.json").catch(() => ({ categories: [] })),
      loadCSV("./data/saved_classifier_confusion_counts.csv").catch(() => []),
    ]);

    dashboardSourceRows = {
      regression: regRows,
      classifier: clsRows,
      joint: jointRows,
    };

    const commonModeSelect = document.getElementById("common-target-mode");
    const recurringOnlySelect = document.getElementById("recurring-only-mode");
    if (commonModeSelect) {
      commonTargetMode = "common";
      commonModeSelect.value = "common";
      commonModeSelect.disabled = true;
      commonModeSelect.addEventListener("change", () => {
        commonTargetMode = "common";
        renderOverviewSummary();
      });
    }

    if (recurringOnlySelect) {
      recurringOnlySelect.addEventListener("change", () => {
        recurringOnlyMode = recurringOnlySelect.value;
        renderOverviewSummary();
      });
    }

    renderOverviewSummary();
    renderRegressionSection(regRows, violationsPayload, trainingPayload);
    renderClassifierSection(clsRows);
    initClassifierCorrelationMatrixSection(clsRows);
    renderJointSection(clsRows, jointRows);
    renderPerformanceLevelMetricsTable();
    renderOutputTypeComparisonTable();
    renderValueDistributionAccuracyFigure(valueDistributionPayload);
    initTabs();
    initInsightsSubchapters();
    initDataMethodsSubtabs();
    initSupplementarySubchapters();
    initConfusionMatrixSection(confusionRows);
    initDatasheetsTab(regSheetRows, clsSheetRows, jointSheetRows);
    initTrainingDataTab(trainingPayload);

  } catch (err) {
    document.body.innerHTML = `<main class="container"><section class="panel"><h2>Dashboard load error</h2><p>${err.message}</p></section></main>`;
  }
}

main();
