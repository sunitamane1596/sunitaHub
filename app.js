const metricDefinitions = [
  { key: 'mae', label: 'MAE', formatter: (value) => Number(value).toFixed(3) },
  { key: 'rmse', label: 'RMSE', formatter: (value) => Number(value).toFixed(3) },
  { key: 'r2', label: 'R²', formatter: (value) => Number(value).toFixed(3) },
  { key: 'baseline_mae', label: 'Baseline MAE', formatter: (value) => Number(value).toFixed(3) },
  { key: 'baseline_rmse', label: 'Baseline RMSE', formatter: (value) => Number(value).toFixed(3) },
  { key: 'baseline_r2', label: 'Baseline R²', formatter: (value) => Number(value).toFixed(3) },
  { key: 'negative_raw_prediction_count', label: 'Negative raw preds', formatter: (value) => Number(value).toLocaleString() },
  { key: 'test_rows', label: 'Test rows', formatter: (value) => Number(value).toLocaleString() },
];

const plotDefinitions = [
  { title: 'Actual vs predicted', src: 'outputs/evaluation/actual_vs_predicted.png' },
  { title: 'Residuals', src: 'outputs/evaluation/residuals.png' },
  { title: 'EDA numeric distributions', src: 'outputs/eda/numeric_distributions.png' },
  { title: 'Target distribution', src: 'outputs/eda/target_distribution.png' },
  { title: 'Correlation matrix', src: 'outputs/eda/correlation_matrix.png' },
];

function renderMetrics(metrics) {
  const grid = document.getElementById('metricsGrid');
  grid.innerHTML = metricDefinitions
    .map(({ key, label, formatter }) => {
      const value = metrics[key] ?? 'N/A';
      return `
        <article class="metric-card">
          <span class="label">${label}</span>
          <div class="value">${formatter(value)}</div>
          <div class="meta">${key === 'r2' || key === 'baseline_r2' ? 'Higher is better' : 'Lower is better'}</div>
        </article>
      `;
    })
    .join('');

  const summary = document.getElementById('datasetSummary');
  const items = [
    { label: 'Test rows', value: Number(metrics.test_rows || 0).toLocaleString() },
    { label: 'Negative raw predictions', value: Number(metrics.negative_raw_prediction_count || 0).toLocaleString() },
    { label: 'MAE', value: Number(metrics.mae || 0).toFixed(3) },
    { label: 'RMSE', value: Number(metrics.rmse || 0).toFixed(3) },
  ];

  summary.innerHTML = items
    .map(
      (item) => `
        <div class="stat-item">
          <span class="label">${item.label}</span>
          <span class="value">${item.value}</span>
        </div>
      `
    )
    .join('');
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return [];

  const headers = splitCSVLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const values = splitCSVLine(line);
    return headers.reduce((record, header, index) => {
      record[header] = values[index] ?? '';
      return record;
    }, {});
  });

  return rows.filter((row) => Object.values(row).some((value) => value !== ''));
}

function splitCSVLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      values.push(current);
      current = '';
    } else {
      current += char;
    }
  }

  values.push(current);
  return values.map((value) => value.trim());
}

function renderPredictionTable(rows) {
  const thead = document.getElementById('tableHead');
  const tbody = document.getElementById('tableBody');

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No prediction rows available.</td></tr>';
    return;
  }

  const headers = Object.keys(rows[0]);

  thead.innerHTML = `<tr>${headers.map((header) => `<th>${header}</th>`).join('')}</tr>`;
  tbody.innerHTML = rows
    .slice(0, 8)
    .map(
      (row) =>
        `<tr>${headers
          .map((header) => `<td>${row[header] || '-'}</td>`)
          .join('')}</tr>`
    )
    .join('');
}

function renderPlots() {
  const grid = document.getElementById('plotGrid');
  grid.innerHTML = plotDefinitions
    .map(
      (plot) => `
        <article class="plot-card">
          <img src="${plot.src}" alt="${plot.title}" />
          <h3>${plot.title}</h3>
        </article>
      `
    )
    .join('');
}

async function loadMetrics() {
  const response = await fetch('outputs/evaluation/evaluation_metrics.json');
  if (!response.ok) {
    throw new Error('Unable to load evaluation metrics');
  }
  return response.json();
}

async function loadPredictionRows() {
  const response = await fetch('outputs/evaluation/test_predictions.csv');
  if (!response.ok) {
    throw new Error('Unable to load prediction CSV');
  }
  const csv = await response.text();
  return parseCSV(csv);
}

async function initDashboard() {
  try {
    const metrics = await loadMetrics();
    renderMetrics(metrics);
  } catch (error) {
    document.getElementById('metricsGrid').innerHTML = `
      <div class="empty-state">Could not load metrics: ${error.message}</div>
    `;
  }

  try {
    const rows = await loadPredictionRows();
    renderPredictionTable(rows);
  } catch (error) {
    document.getElementById('tableBody').innerHTML = `
      <tr><td colspan="4" class="empty-state">Could not load prediction rows: ${error.message}</td></tr>
    `;
  }

  renderPlots();
}

document.getElementById('refreshButton').addEventListener('click', initDashboard);
window.addEventListener('DOMContentLoaded', initDashboard);
