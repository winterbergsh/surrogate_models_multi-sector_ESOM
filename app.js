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
  maxPanels: 12,
};

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
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  const text = await response.text();
  return parseCSV(text);
}

async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

function toNum(v) {
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

function getRecurringTargetSet(classifierRows) {
  const recurring = new Set();
  for (const row of classifierRows || []) {
    const target = normalizeTargetName(row.target_variable);
    if (!target) continue;

    const nCategories = toNum(row.n_categories);
    const categoriesRaw = String(row.categories ?? "").trim();
    const hasExplicitCategories = categoriesRaw && categoriesRaw !== "[]";
    if ((nCategories !== null && nCategories > 0) || hasExplicitCategories) {
      recurring.add(target);
    }
  }
  return recurring;
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

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab-target");
      if (!target) return;

      buttons.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      panels.forEach((panel) => {
        const isActive = panel.id === target;
        panel.classList.toggle("active", isActive);
        panel.setAttribute("aria-hidden", isActive ? "false" : "true");
      });
    });
  });
}

function initInsightsSubchapters() {
  const buttons = document.querySelectorAll(".subchapter-btn");
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

function escapeHTML(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
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
  const chartId = "training-distribution-plot";

  if (!countEl || !typeFilterEl || !searchEl || !selectEl || !selectAllBtn || !clearBtn || !noteEl) return;

  countEl.textContent = String(trainingPayload?.target_count_with_variation ?? targets.length);

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
      noteEl.textContent = "Select at least one target variable to show value distribution.";
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
      ? `${typeText}Showing ${renderedTargets.length} of ${selectedTargets.length} selected targets to keep the page responsive.`
      : `${typeText}Showing ${renderedTargets.length} selected target variable distribution(s).`;

    const cols = 2;
    const rows = Math.max(1, Math.ceil(renderedTargets.length / cols));
    const layout = baseLayout({
      showlegend: false,
      height: Math.max(420, rows * 320),
      margin: { t: 52, r: 24, l: 54, b: 54 },
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
        marker: { color: "#82AFDA", line: { color: "#0a3a52", width: 0.5 } },
        opacity: 0.82,
        xaxis: xref,
        yaxis: yref,
        hovertemplate: "Value=%{x:.4g}<br>Frequency=%{y}<extra></extra>",
      });

      const unit = String(target.unit || "").trim() || "N/A";
      const title = formatTrainingTargetLabel(target);

      layout[`xaxis${axisNum === 1 ? "" : axisNum}`] = {
        title: `Value (${unit})`,
        showgrid: true,
        gridcolor: "#e9edf2",
        zeroline: false,
      };
      layout[`yaxis${axisNum === 1 ? "" : axisNum}`] = {
        title: "Frequency",
        showgrid: true,
        gridcolor: "#eef2f6",
        zeroline: false,
      };

      const xDomainRef = axisNum === 1 ? "x domain" : `x${axisNum} domain`;
      const yDomainRef = axisNum === 1 ? "y domain" : `y${axisNum} domain`;
      layout.annotations.push({
        text: `<b>${escapeHTML(title)}</b>`,
        xref: xDomainRef,
        yref: yDomainRef,
        x: 0.5,
        y: 1.02,
        showarrow: false,
        font: { size: 12, color: "#1f2a37" },
      });
    });

    Plotly.newPlot(chartId, traces, layout, { responsive: true, displayModeBar: false });
  }

  searchEl.addEventListener("input", () => {
    renderOptions();
  });

  typeFilterEl.addEventListener("change", () => {
    renderOptions();
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
    trainingDataState.selected = new Set([String(targets[0].outcome)]);
  }
  renderOptions();
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

function renderOverviewSummary() {
  const { commonTargetCount, xgbPairCommonTargetCount } = buildMethodSummaries(
    dashboardSourceRows.regression,
    dashboardSourceRows.classifier,
    dashboardSourceRows.joint,
    commonTargetMode
  );

  const methodMapRaw = getMethodRowsMap(
    dashboardSourceRows.regression,
    dashboardSourceRows.classifier,
    dashboardSourceRows.joint
  );
  const recurringTargetSet = getRecurringTargetSet(dashboardSourceRows.classifier);
  const methodMap = Object.fromEntries(
    methodMapRaw.map((entry) => {
      const deduped = dedupeRowsByTarget(entry.rows);
      return [entry.label, deduped];
    })
  );

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
    if (commonTargetMode !== "common") return rows;
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
      ? ` Recurring-only filter is applied to section 2 only (${recurringTargetSet.size} recurring targets detected from classifier results).`
      : "";

    if (commonTargetMode === "common") {
      note.textContent = `Common-target mode enabled: GP/MLP/XGB share ${commonConventional.size} targets, XGB/XGB+classifier share ${xgbPairCommonTargetCount}, and XGB+classifier/informed_classifier share ${commonClfInf.size}.${recurringSuffix}`;
    } else {
      note.textContent = `All-target mode enabled: each method uses all available targets from its own results.${recurringSuffix}`;
    }
  }
}

function renderRegressionSection(regRows) {
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
    { sourceKey: "nrmse", title: "NRMSE Distribution", yTitle: "NRMSE", scale: 1, decimals: 3 },
    { sourceKey: "nmae", title: "NMAE Distribution", yTitle: "NMAE", scale: 1, decimals: 3 },
    { sourceKey: "RTA1pct", title: "1% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { sourceKey: "RTA5pct", title: "5% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { sourceKey: "RTA10pct", title: "10% RTA Distribution", yTitle: "RTA (%)", scale: 100, decimals: 1, range: [0, 100] },
    { sourceKey: "training_time_s", title: "Training Time Distribution", yTitle: "Training Time (s)", scale: 1, decimals: 2 },
    { sourceKey: "running_time_s", title: "Running Time Distribution", yTitle: "Running Time (ms)", scale: 1000, decimals: 2 },
  ];

  const traces = [];

  function filterOutliersIQR(values) {
    const nums = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    if (nums.length < 4) return nums;

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
    return nums.filter((v) => v >= low && v <= high);
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
      const valuesRaw = commonRows[m]
        .map((r) => toNum(r[spec.sourceKey]))
        .filter((v) => v !== null)
        .map((v) => v * spec.scale);
      const values = filterOutliersIQR(valuesRaw);

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
        hovertemplate: `${labels[m]}<br>%{y:.${spec.decimals}f}<extra></extra>`,
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
      y: 1.02,
      showarrow: false,
      font: { size: 16, color: "#111111" },
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

function renderClassifierSection(clsRows) {
  const methodSelect = document.getElementById("classifier-method-filter");
  const metricSelect = document.getElementById("heatmap-metric-filter");
  const methods = [...new Set(clsRows.map((r) => r.surrogate_method).filter(Boolean))].sort();

  methodSelect.innerHTML = methods.map((m) => `<option value="${m}" ${m.includes("XGB_HYBRID_tuned") ? "selected" : ""}>${m}</option>`).join("");

  function rerender() {
    const selectedMethod = methodSelect.value;
    const metricCol = metricSelect.value;
    const rows = clsRows.filter((r) => r.surrogate_method === selectedMethod);
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
    rows.forEach((r) => {
      const g = extractVariableGroup(r.target_variable);
      if (!grouped[g]) grouped[g] = [];
      grouped[g].push(toNum(r.classifier_accuracy));
    });

    const x = Object.keys(grouped);
    const y = x.map((k) => mean(grouped[k]));

    Plotly.newPlot(
      "classifier-accuracy-bar",
      [
        {
          type: "bar",
          x,
          y,
          marker: { color: "#2a6f97" },
        },
      ],
      baseLayout({
        margin: { t: 26, r: 12, l: 56, b: 135 },
        yaxis: { title: "Classifier Accuracy", gridcolor: "#ececec", zeroline: false },
      }),
      { responsive: true, displayModeBar: false }
    );

    const scatterRows = rows
      .map((r) => ({
        x: toNum(r.RTA5reg),
        y: toNum(r.RTA5pct),
        label: r.target_variable,
      }))
      .filter((r) => r.x !== null && r.y !== null);

    Plotly.newPlot(
      "classifier-rta-scatter",
      [
        {
          type: "scatter",
          mode: "markers",
          x: scatterRows.map((r) => r.x),
          y: scatterRows.map((r) => r.y),
          text: scatterRows.map((r) => r.label),
          marker: { color: "#d94e18", size: 7, opacity: 0.7 },
          hovertemplate: "%{text}<br>RTA5reg=%{x}<br>RTA5hybrid=%{y}<extra></extra>",
        },
      ],
      baseLayout({
        xaxis: { title: "Regularized RTA5 count" },
        yaxis: { title: "Hybrid RTA5", gridcolor: "#ececec", zeroline: false },
      }),
      { responsive: true, displayModeBar: false }
    );
  }

  methodSelect.addEventListener("change", rerender);
  metricSelect.addEventListener("change", rerender);
  rerender();
}

function renderJointSection(jointRows) {
  const rows = jointRows.filter((r) => r.surrogate_method === "Informed_XGB_HYBRID");

  const families = ["techStocks", "techUseNet"];
  const familyRows = families.map((f) => rows.filter((r) => String(r.target_variable || "").includes(f)));

  Plotly.newPlot(
    "joint-family-bar",
    [
      {
        type: "bar",
        x: families,
        y: familyRows.map((arr) => mean(arr.map((r) => toNum(r.nrmse)))),
        marker: { color: ["#f77f00", "#118ab2"] },
      },
    ],
    baseLayout({
      yaxis: { title: "Mean NRMSE", gridcolor: "#ececec", zeroline: false },
    }),
    { responsive: true, displayModeBar: false }
  );

  Plotly.newPlot(
    "joint-family-rta",
    [
      {
        type: "bar",
        x: families,
        y: familyRows.map((arr) => mean(arr.map((r) => toNum(r.RTA5pct)))),
        marker: { color: ["#fcbf49", "#4f772d"] },
      },
    ],
    baseLayout({
      yaxis: { title: "Mean RTA5", gridcolor: "#ececec", zeroline: false },
    }),
    { responsive: true, displayModeBar: false }
  );

  const summaryRows = [
    {
      model: "xgb_informed_classifier.py",
      n: rows.length,
      nrmse: mean(rows.map((r) => toNum(r.nrmse))),
      nmae: mean(rows.map((r) => toNum(r.nmae))),
      rta1: mean(rows.map((r) => toNum(r.RTA1pct))),
      rta5: mean(rows.map((r) => toNum(r.RTA5pct))),
      rta10: mean(rows.map((r) => toNum(r.RTA10pct))),
      cls_acc: mean(rows.map((r) => toNum(r.classifier_accuracy_train) ?? toNum(r.classifier_accuracy))),
    },
  ];

  const table = document.getElementById("joint-summary-table");
  table.innerHTML = [
    "<thead><tr><th>Model</th><th>Rows</th><th>Mean NRMSE</th><th>Mean NMAE</th><th>Mean RTA1</th><th>Mean RTA5</th><th>Mean RTA10</th><th>Mean Classifier Accuracy</th></tr></thead>",
    "<tbody>",
    ...summaryRows.map(
      (s) => `<tr><td>${s.model}</td><td>${s.n}</td><td>${fmt(s.nrmse)}</td><td>${fmt(s.nmae)}</td><td>${fmt(s.rta1)}</td><td>${fmt(s.rta5)}</td><td>${fmt(s.rta10)}</td><td>${fmt(s.cls_acc)}</td></tr>`
    ),
    "</tbody>",
  ].join("");

  jointSummaryData = summaryRows.map((s) => ({
    model: s.model,
    rows: s.n,
    mean_nrmse: fmt(s.nrmse),
    mean_nmae: fmt(s.nmae),
    mean_rta1: fmt(s.rta1),
    mean_rta5: fmt(s.rta5),
    mean_rta10: fmt(s.rta10),
    mean_classifier_accuracy: fmt(s.cls_acc),
  }));
}

async function main() {
  try {
    const [regRows, clsRows, jointRows, regSheetRows, clsSheetRows, jointSheetRows, trainingPayload] = await Promise.all([
      loadCSV("./data/regression_surrogate_run_log.csv"),
      loadCSV("./data/classifier_2fold_performance.csv"),
      loadCSV("./data/joint_learning_informed_classifier_run_log.csv"),
      loadCSV("./data/regression_datasheet.csv"),
      loadCSV("./data/classifier_datasheet.csv"),
      loadCSV("./data/joint_datasheet.csv"),
      loadJSON("./data/training_value_distributions.json").catch(() => ({ targets: [], target_count_with_variation: 0 })),
    ]);

    dashboardSourceRows = {
      regression: regRows,
      classifier: clsRows,
      joint: jointRows,
    };

    const commonModeSelect = document.getElementById("common-target-mode");
    const recurringOnlySelect = document.getElementById("recurring-only-mode");
    if (commonModeSelect) {
      commonModeSelect.addEventListener("change", () => {
        commonTargetMode = commonModeSelect.value;
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
    renderRegressionSection(regRows);
    renderClassifierSection(clsRows);
    renderJointSection(jointRows);
    initTabs();
    initInsightsSubchapters();
    initDatasheetsTab(regSheetRows, clsSheetRows, jointSheetRows);
    initTrainingDataTab(trainingPayload);

  } catch (err) {
    document.body.innerHTML = `<main class="container"><section class="panel"><h2>Dashboard load error</h2><p>${err.message}</p></section></main>`;
  }
}

main();
