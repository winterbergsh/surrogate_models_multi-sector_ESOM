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

let regressionSummaryData = [];
let jointSummaryData = [];
let currentClassifierRows = [];

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

function renderCards(regRows, clsRows, jointRows) {
  const cardRoot = document.getElementById("model-cards");
  const cards = [
    {
      label: "XGB.py",
      rows: regRows.filter((r) => r.surrogate_method === "XGB_single"),
    },
    {
      label: "MLP.py",
      rows: regRows.filter((r) => r.surrogate_method === "AE_MLP"),
    },
    {
      label: "GP.py",
      rows: regRows.filter((r) => r.surrogate_method === "AE_GaussianProcess"),
    },
    {
      label: "XGB+classifier.py",
      rows: clsRows.filter((r) => String(r.surrogate_method || "").includes("XGB_HYBRID_tuned")),
    },
    {
      label: "xgb_informed_classifier.py",
      rows: jointRows.filter((r) => String(r.surrogate_method || "") === "Informed_XGB_HYBRID"),
    },
  ];

  cardRoot.innerHTML = cards
    .map((c) => {
      const nrmse = median(c.rows.map((r) => toNum(r.nrmse)));
      const rta5 = median(c.rows.map((r) => toNum(r.RTA5pct)));
      return `<article class="card"><h3>${c.label}</h3><div class="metric">Median NRMSE: ${fmt(nrmse, 3)}</div><div class="metric">Median RTA5: ${fmt(rta5, 3)}</div><div class="metric">Rows: ${c.rows.length}</div></article>`;
    })
    .join("");
}

function renderRegressionSection(regRows) {
  const clean = regRows.filter((r) => ["XGB_single", "AE_MLP", "AE_GaussianProcess"].includes(r.surrogate_method));
  const methods = ["XGB_single", "AE_MLP", "AE_GaussianProcess"];
  const colors = ["#f28f3b", "#4c78a8", "#59a14f"];

  const boxNRMSE = methods.map((m, i) => ({
    type: "box",
    name: MODEL_ALIASES[m],
    y: clean.filter((r) => r.surrogate_method === m).map((r) => toNum(r.nrmse)).filter((v) => v !== null),
    marker: { color: colors[i] },
    boxmean: true,
  }));

  Plotly.newPlot("regression-boxplot", boxNRMSE, {
    ...baseLayout(),
    yaxis: { title: "NRMSE", gridcolor: "#ececec", zeroline: false },
    xaxis: { tickangle: -8 },
  }, { responsive: true, displayModeBar: false });

  const boxRTA = methods.map((m, i) => ({
    type: "box",
    name: MODEL_ALIASES[m],
    y: clean.filter((r) => r.surrogate_method === m).map((r) => toNum(r.RTA5pct)).filter((v) => v !== null),
    marker: { color: colors[i] },
    boxmean: true,
  }));

  Plotly.newPlot("regression-rta5-boxplot", boxRTA, {
    ...baseLayout(),
    yaxis: { title: "RTA5", gridcolor: "#ececec", zeroline: false },
    xaxis: { tickangle: -8 },
  }, { responsive: true, displayModeBar: false });

  const summary = methods.map((m) => {
    const rows = clean.filter((r) => r.surrogate_method === m);
    return {
      model: MODEL_ALIASES[m],
      n: rows.length,
      nrmse: mean(rows.map((r) => toNum(r.nrmse))),
      nmae: mean(rows.map((r) => toNum(r.nmae))),
      rta1: mean(rows.map((r) => toNum(r.RTA1pct))),
      rta5: mean(rows.map((r) => toNum(r.RTA5pct))),
      rta10: mean(rows.map((r) => toNum(r.RTA10pct))),
      train: mean(rows.map((r) => toNum(r.training_time_s))),
    };
  });

  const table = document.getElementById("regression-summary-table");
  table.innerHTML = [
    "<thead><tr><th>Model</th><th>Rows</th><th>Mean NRMSE</th><th>Mean NMAE</th><th>Mean RTA1</th><th>Mean RTA5</th><th>Mean RTA10</th><th>Mean Training Time (s)</th></tr></thead>",
    "<tbody>",
    ...summary.map(
      (s) => `<tr><td>${s.model}</td><td>${s.n}</td><td>${fmt(s.nrmse)}</td><td>${fmt(s.nmae)}</td><td>${fmt(s.rta1)}</td><td>${fmt(s.rta5)}</td><td>${fmt(s.rta10)}</td><td>${fmt(s.train, 2)}</td></tr>`
    ),
    "</tbody>",
  ].join("");

  regressionSummaryData = summary.map((s) => ({
    model: s.model,
    rows: s.n,
    mean_nrmse: fmt(s.nrmse),
    mean_nmae: fmt(s.nmae),
    mean_rta1: fmt(s.rta1),
    mean_rta5: fmt(s.rta5),
    mean_rta10: fmt(s.rta10),
    mean_training_time_s: fmt(s.train, 2),
  }));
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
    const [regRows, clsRows, jointRows] = await Promise.all([
      loadCSV("./data/regression_surrogate_run_log.csv"),
      loadCSV("./data/classifier_2fold_performance.csv"),
      loadCSV("./data/joint_learning_informed_classifier_run_log.csv"),
    ]);

    renderCards(regRows, clsRows, jointRows);
    renderRegressionSection(regRows);
    renderClassifierSection(clsRows);
    renderJointSection(jointRows);

    bindCSVButton("download-regression-summary-csv", "regression_summary_table.csv", () => regressionSummaryData);
    bindCSVButton("download-joint-summary-csv", "joint_learning_summary_table.csv", () => jointSummaryData);
    bindCSVButton("download-classifier-filtered-csv", "classifier_filtered_rows.csv", () => currentClassifierRows);

    bindPlotDownloadButton("download-regression-nrmse-png", "regression-boxplot", "regression_nrmse_distribution");
    bindPlotDownloadButton("download-regression-rta5-png", "regression-rta5-boxplot", "regression_rta5_distribution");
    bindPlotDownloadButton("download-classifier-heatmap-png", "classifier-heatmap", "classifier_variable_type_heatmap");
  } catch (err) {
    document.body.innerHTML = `<main class="container"><section class="panel"><h2>Dashboard load error</h2><p>${err.message}</p></section></main>`;
  }
}

main();
