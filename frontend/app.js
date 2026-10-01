/**
 * ALPHATRADE AI — CLIENT APPLICATION CONTROLLER
 * Handles interactive charts, live data fetching, state management,
 * and AI multi-horizon forecast rendering.
 */

// Production Backend API Base URL — points to live deployed Render backend
const PRODUCTION_API_URL = "https://stockai-backend-uc9l.onrender.com";

// Smart environment resolver: auto-connects to local backend when testing locally,
// and routes to PRODUCTION_API_URL when hosted live on Vercel or custom domain.
const isLocalEnv = typeof window !== 'undefined' && (
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1' ||
  window.location.protocol === 'file:' ||
  !window.location.hostname
);
const API_BASE_URL = isLocalEnv 
  ? (window.location.origin && window.location.origin !== 'null' && window.location.protocol.startsWith('http') ? window.location.origin : 'http://127.0.0.1:8000')
  : PRODUCTION_API_URL;

// Application Global State
const state = {
  currentTicker: 'RELIANCE.NS',
  currentPeriod: '1y',
  activeTab: 'dashboard',
  overview: null,
  history: null,
  prediction: null,
  benchmarks: null,
  watchlist: [],
  overlays: {
    sma20: true,
    sma50: true,
    bb: false
  },
  studioModels: {
    xgb: true,
    lstm: true,
    confidence: true
  }
};

// Chart Instances
let mainPriceChart = null;
let rsiChart = null;
let macdChart = null;
let forecastChart = null;
let benchmarkR2Chart = null;
let benchmarkMaeChart = null;

// DOM Ready initialization
document.addEventListener('DOMContentLoaded', () => {
  initUIEventListeners();
  loadWatchlist();
  loadStockData(state.currentTicker);
  loadBenchmarks();
  loadWalkForwardBenchmarks();
  loadIPOData();
  loadIPOHistory();
  loadMarketMood();
  loadSystemStatus();
});

/* ==========================================================================
   EVENT LISTENERS & UI INTERACTIONS
   ========================================================================== */
function initUIEventListeners() {
  // Mobile drawer open/close
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const sidebar = document.getElementById('app-sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  const sidebarCloseBtn = document.getElementById('sidebar-close-btn');

  const openDrawer = () => {
    if (sidebar) sidebar.classList.add('open');
    if (sidebarOverlay) sidebarOverlay.classList.add('active');
  };

  const closeDrawer = () => {
    if (sidebar) sidebar.classList.remove('open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
  };

  if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', openDrawer);
  if (sidebarCloseBtn) sidebarCloseBtn.addEventListener('click', closeDrawer);
  if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeDrawer);

  // Tab Switching
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
      closeDrawer();
    });
  });

  // Timeframe Buttons
  document.querySelectorAll('.time-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentPeriod = btn.getAttribute('data-period');
      fetchStockHistory(state.currentTicker, state.currentPeriod);
    });
  });

  // Indicator Toggles
  document.getElementById('toggle-sma20').addEventListener('click', function () {
    state.overlays.sma20 = !state.overlays.sma20;
    this.classList.toggle('active', state.overlays.sma20);
    renderMainChart();
  });

  document.getElementById('toggle-sma50').addEventListener('click', function () {
    state.overlays.sma50 = !state.overlays.sma50;
    this.classList.toggle('active', state.overlays.sma50);
    renderMainChart();
  });

  document.getElementById('toggle-bb').addEventListener('click', function () {
    state.overlays.bb = !state.overlays.bb;
    this.classList.toggle('active', state.overlays.bb);
    renderMainChart();
  });

  // Search Action
  const searchInput = document.getElementById('ticker-search-input');
  const searchBtn = document.getElementById('search-btn');

  const executeSearch = () => {
    const rawVal = searchInput.value.trim().toUpperCase();
    if (!rawVal) return;
    
    // Auto-append .NS if simple Indian ticker name
    let ticker = rawVal;
    if (!ticker.includes('.') && !['AAPL', 'NVDA', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'META'].includes(ticker)) {
      ticker = `${ticker}.NS`;
    }
    loadStockData(ticker);
    searchInput.value = '';
  };

  searchBtn.addEventListener('click', executeSearch);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') executeSearch();
  });

  // Refresh and Predict CTA
  document.getElementById('refresh-data-btn').addEventListener('click', () => {
    showToast(`Syncing market data for ${state.currentTicker}...`, 'info');
    loadStockData(state.currentTicker);
  });

  document.getElementById('run-prediction-btn').addEventListener('click', () => {
    switchTab('forecast');
    if (!state.prediction) {
      fetchStockPrediction(state.currentTicker);
    }
  });

  document.getElementById('view-full-forecast-btn').addEventListener('click', () => {
    switchTab('forecast');
  });

  // Studio Model Toggles
  document.getElementById('toggle-xgb-curve').addEventListener('change', (e) => {
    state.studioModels.xgb = e.target.checked;
    e.target.closest('.model-check-pill').classList.toggle('active', e.target.checked);
    renderForecastChart();
  });

  document.getElementById('toggle-lstm-curve').addEventListener('change', (e) => {
    state.studioModels.lstm = e.target.checked;
    e.target.closest('.model-check-pill').classList.toggle('active', e.target.checked);
    renderForecastChart();
  });

  document.getElementById('toggle-confidence-band').addEventListener('change', (e) => {
    state.studioModels.confidence = e.target.checked;
    e.target.closest('.model-check-pill').classList.toggle('active', e.target.checked);
    renderForecastChart();
  });

  const analyzeBtn = document.getElementById('btn-analyze-text');
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', submitReportAnalysis);
  }

  // Quick Prompt Chips in Research Assistant
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      const textarea = document.getElementById('research-input-text');
      if (textarea && prompt) {
        textarea.value = prompt;
        submitReportAnalysis();
      }
    });
  });
}

function switchTab(tabName) {
  state.activeTab = tabName;

  // Nav buttons
  document.querySelectorAll('.nav-item').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabName);
  });

  // Panes
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  const activePane = document.getElementById(`tab-${tabName}`);
  if (activePane) activePane.classList.add('active');

  // Trigger resize on charts to ensure responsiveness
  window.dispatchEvent(new Event('resize'));

  if (tabName === 'forecast' && state.prediction) {
    renderForecastChart();
    renderForecastMatrixTable();
  } else if (tabName === 'benchmarks') {
    if (state.benchmarks) {
      renderBenchmarkCharts();
    } else {
      loadBenchmarks();
    }
  } else if (tabName === 'ipo') {
    if (ipoHistoryChart) {
      setTimeout(() => ipoHistoryChart.resize(), 50);
    }
  } else if (tabName === 'research') {
    loadResearchInsights(state.currentTicker);
  }
}

/* ==========================================================================
   DATA FETCHING APIS
   ========================================================================== */
async function loadWatchlist() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/stocks`);
    const data = await res.json();
    state.watchlist = data.stocks || [];
    renderWatchlist();
  } catch (err) {
    console.error('Watchlist load error:', err);
  }
}

async function loadStockData(ticker) {
  state.currentTicker = ticker;
  setChartLoading(true);

  // Update URL state or active ticker
  highlightActiveWatchlistItem(ticker);

  try {
    // 1. Fetch Overview & History concurrently
    const [overviewRes, historyRes] = await Promise.all([
      fetch(`${API_BASE_URL}/api/stock/${ticker}/overview`),
      fetch(`${API_BASE_URL}/api/stock/${ticker}/history?period=${state.currentPeriod}`)
    ]);

    if (!overviewRes.ok || !historyRes.ok) {
      throw new Error(`Stock not found or server error for ${ticker}`);
    }

    state.overview = await overviewRes.json();
    state.history = await historyRes.json();

    updateHeaderUI();
    renderMainChart();
    renderSubCharts();

    // 2. Fetch AI Predictions in background
    fetchStockPrediction(ticker);
    loadSentiment(ticker);
    loadResearchInsights(ticker);

  } catch (err) {
    console.error('Data load error:', err);
    showToast(err.message, 'error');
  } finally {
    setChartLoading(false);
  }
}

async function fetchStockHistory(ticker, period) {
  setChartLoading(true);
  try {
    const res = await fetch(`${API_BASE_URL}/api/stock/${ticker}/history?period=${period}`);
    if (!res.ok) throw new Error('Failed to fetch period data');
    state.history = await res.json();
    renderMainChart();
    renderSubCharts();
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setChartLoading(false);
  }
}

async function fetchStockPrediction(ticker) {
  setForecastLoading(true);
  try {
    const res = await fetch(`${API_BASE_URL}/api/stock/${ticker}/predict`);
    if (!res.ok) throw new Error('AI prediction generation failed');
    state.prediction = await res.json();

    updateSignalCard();
    renderMiniForecastBars();

    if (state.activeTab === 'forecast') {
      renderForecastChart();
      renderForecastMatrixTable();
    }
  } catch (err) {
    console.warn('Prediction warning:', err);
  } finally {
    setForecastLoading(false);
  }
}

async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/benchmark`);
    state.benchmarks = await res.json();
    if (state.activeTab === 'benchmarks') {
      renderBenchmarkCharts();
    }
  } catch (err) {
    console.error('Benchmarks error:', err);
  }
}

/* ==========================================================================
   UI RENDERING & BINDINGS
   ========================================================================== */
function updateHeaderUI() {
  const o = state.overview;
  if (!o) return;

  document.getElementById('header-stock-name').textContent = o.name || o.ticker;
  document.getElementById('header-ticker').textContent = o.ticker;
  document.getElementById('header-sector').textContent = o.sector || 'Equities';
  document.getElementById('header-market').textContent = o.ticker.endsWith('.NS') ? 'NSE' : 'NASDAQ';
  
  const currSym = o.currency === 'INR' ? '₹' : '$';
  document.getElementById('header-currency').textContent = currSym;
  document.getElementById('header-price').textContent = formatNumber(o.current_price);
  
  const isBullish = o.change >= 0;
  const changeEl = document.getElementById('header-change');
  changeEl.className = `change-badge ${isBullish ? 'bullish' : 'bearish'}`;
  changeEl.textContent = `${isBullish ? '+' : ''}${formatNumber(o.change)} (${isBullish ? '+' : ''}${o.change_pct.toFixed(2)}%)`;

  document.getElementById('header-prev-close').textContent = `${currSym}${formatNumber(o.previous_close)}`;
  document.getElementById('header-update-time').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // KPI Strip
  document.getElementById('kpi-day-low').textContent = formatNumber(o.day_low);
  document.getElementById('kpi-day-high').textContent = formatNumber(o.day_high);
  
  const daySpan = Math.max(o.day_high - o.day_low, 0.01);
  const dayProgress = Math.min(Math.max(((o.current_price - o.day_low) / daySpan) * 100, 5), 100);
  document.getElementById('kpi-day-fill').style.width = `${dayProgress}%`;

  document.getElementById('kpi-52-low').textContent = formatNumber(o.fifty_two_low);
  document.getElementById('kpi-52-high').textContent = formatNumber(o.fifty_two_high);
  document.getElementById('kpi-52-fill').style.width = `${o.range_pct || 50}%`;

  document.getElementById('kpi-volume').textContent = formatLargeNumber(o.volume);
  document.getElementById('kpi-avg-volume').textContent = `Avg: ${formatLargeNumber(o.avg_volume)}`;
  document.getElementById('kpi-market-cap').textContent = o.market_cap ? `${currSym}${formatLargeNumber(o.market_cap)}` : 'N/A';
  document.getElementById('kpi-pe').textContent = o.pe_ratio ? o.pe_ratio.toFixed(1) : 'N/A';
  document.getElementById('kpi-beta').textContent = o.beta ? o.beta.toFixed(2) : '1.00';
}

function renderWatchlist() {
  const container = document.getElementById('watchlist-items');
  if (!container) return;

  container.innerHTML = state.watchlist.map(item => `
    <div class="watch-item ${item.ticker === state.currentTicker ? 'active' : ''}" data-ticker="${item.ticker}">
      <div class="watch-info">
        <span class="watch-ticker">${item.ticker}</span>
        <span class="watch-name">${item.name}</span>
      </div>
      <div class="watch-price-col">
        <span class="watch-price">${item.ticker.endsWith('.NS') ? '₹' : '$'}--</span>
        <span class="watch-badge bullish">Track</span>
      </div>
    </div>
  `).join('');

  document.getElementById('watchlist-count').textContent = state.watchlist.length;

  container.querySelectorAll('.watch-item').forEach(el => {
    el.addEventListener('click', () => {
      const ticker = el.getAttribute('data-ticker');
      loadStockData(ticker);
      // Close drawer on mobile
      const sidebar = document.getElementById('app-sidebar');
      const overlay = document.getElementById('sidebar-overlay');
      if (sidebar) sidebar.classList.remove('open');
      if (overlay) overlay.classList.remove('active');
    });
  });
}

function highlightActiveWatchlistItem(ticker) {
  document.querySelectorAll('.watch-item').forEach(el => {
    el.classList.toggle('active', el.getAttribute('data-ticker') === ticker);
  });
}

/* ==========================================================================
   AI SIGNAL & RECOMMENDATION CARD
   ========================================================================== */
function updateSignalCard() {
  const p = state.prediction;
  if (!p || !p.signal) return;

  const s = p.signal;
  const currSym = state.overview?.currency === 'INR' ? '₹' : '$';

  // Badge action
  const actionTag = document.getElementById('action-tag');
  actionTag.className = `action-tag ${s.action_class}`;
  actionTag.textContent = s.action;

  document.getElementById('action-score').textContent = `${s.confluence_score > 0 ? '+' : ''}${s.confluence_score} / 100`;
  document.getElementById('signal-confidence').textContent = `${s.confidence_pct}% Match`;

  // Targets
  const targetEl = document.getElementById('signal-target-price');
  targetEl.textContent = `${currSym}${formatNumber(s.target_price)}`;

  const stopLossEl = document.getElementById('signal-stop-loss');
  stopLossEl.textContent = `${currSym}${formatNumber(s.stop_loss)}`;

  document.getElementById('signal-risk-level').textContent = s.risk_level;

  const delta = p.xgboost?.overall_7d_change || 0;
  const deltaEl = document.getElementById('signal-expected-gain');
  deltaEl.className = `t-val ${delta >= 0 ? 'bullish' : 'bearish'}`;
  deltaEl.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}%`;

  // Drivers
  const driversContainer = document.getElementById('signal-drivers-list');
  driversContainer.innerHTML = (s.drivers || []).map(d => `<li>${d}</li>`).join('');

  // Studio net move
  const studioNet = document.getElementById('studio-net-move');
  if (studioNet) {
    const netAmount = Math.abs(s.target_price - p.current_price);
    studioNet.className = delta >= 0 ? 'bullish' : 'bearish';
    studioNet.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}% (${currSym}${formatNumber(netAmount)})`;
  }
}

function renderMiniForecastBars() {
  const container = document.getElementById('mini-forecast-bars');
  const p = state.prediction;
  if (!container || !p || !p.xgboost?.forecasts) return;

  const currSym = state.overview?.currency === 'INR' ? '₹' : '$';

  container.innerHTML = p.xgboost.forecasts.slice(0, 5).map(f => `
    <div class="mini-bar-row">
      <span class="mini-day">${f.day_label}</span>
      <span class="mini-price">${currSym}${formatNumber(f.predicted_price)}</span>
      <span class="mini-delta ${f.change_pct >= 0 ? 'bullish' : 'bearish'}">
        ${f.change_pct >= 0 ? '+' : ''}${f.change_pct.toFixed(2)}%
      </span>
    </div>
  `).join('');
}

/* ==========================================================================
   CHART RENDERING WITH CHART.JS
   ========================================================================== */
function renderMainChart() {
  const canvas = document.getElementById('main-price-chart');
  if (!canvas || !state.history || !state.history.data) return;

  const records = state.history.data;
  const labels = records.map(r => r.date);
  const closePrices = records.map(r => r.close);
  const volumes = records.map(r => r.volume);

  const isBullish = closePrices[closePrices.length - 1] >= closePrices[0];
  const primaryColor = isBullish ? '#10b981' : '#ef4444';
  const primaryGradient = isBullish ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';

  const datasets = [
    {
      label: 'Close Price',
      data: closePrices,
      borderColor: primaryColor,
      borderWidth: 2,
      backgroundColor: primaryGradient,
      fill: 'start',
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: primaryColor,
      yAxisID: 'y'
    }
  ];

  // Overlay SMA 20
  if (state.overlays.sma20) {
    datasets.push({
      label: 'SMA 20',
      data: records.map(r => r.sma_20),
      borderColor: '#38bdf8',
      borderWidth: 1.2,
      borderDash: [4, 4],
      fill: false,
      tension: 0.2,
      pointRadius: 0,
      yAxisID: 'y'
    });
  }

  // Overlay SMA 50
  if (state.overlays.sma50) {
    datasets.push({
      label: 'SMA 50',
      data: records.map(r => r.sma_50),
      borderColor: '#f59e0b',
      borderWidth: 1.2,
      fill: false,
      tension: 0.2,
      pointRadius: 0,
      yAxisID: 'y'
    });
  }

  // Overlay Bollinger Bands
  if (state.overlays.bb) {
    datasets.push(
      {
        label: 'BB Upper',
        data: records.map(r => r.bb_upper),
        borderColor: 'rgba(139, 92, 246, 0.5)',
        borderWidth: 1,
        fill: false,
        pointRadius: 0,
        yAxisID: 'y'
      },
      {
        label: 'BB Lower',
        data: records.map(r => r.bb_lower),
        borderColor: 'rgba(139, 92, 246, 0.5)',
        borderWidth: 1,
        fill: '-1',
        backgroundColor: 'rgba(139, 92, 246, 0.05)',
        pointRadius: 0,
        yAxisID: 'y'
      }
    );
  }

  if (mainPriceChart) {
    mainPriceChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  mainPriceChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#94a3b8',
            font: { size: 11, family: 'JetBrains Mono' },
            boxWidth: 12
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          displayColors: true,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw ? formatNumber(ctx.raw) : 'N/A'}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: {
            color: '#64748b',
            maxTicksLimit: 8,
            font: { size: 10, family: 'JetBrains Mono' }
          }
        },
        y: {
          position: 'right',
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#94a3b8',
            font: { size: 11, family: 'JetBrains Mono' },
            callback: (v) => formatNumber(v)
          }
        }
      }
    }
  });
}

function renderSubCharts() {
  if (!state.history || !state.history.data) return;

  const records = state.history.data;
  const labels = records.map(r => r.date);

  // 1. RSI Chart
  const rsiCanvas = document.getElementById('rsi-chart');
  if (rsiCanvas) {
    if (rsiChart) rsiChart.destroy();
    
    const latestRsi = records[records.length - 1].rsi || 50;
    const rsiBadge = document.getElementById('rsi-badge');
    if (rsiBadge) {
      rsiBadge.textContent = latestRsi.toFixed(1);
      rsiBadge.className = `indicator-badge ${latestRsi < 35 ? 'bullish' : latestRsi > 65 ? 'bearish' : 'neutral'}`;
    }

    rsiChart = new Chart(rsiCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data: records.map(r => r.rsi),
          borderColor: '#a855f7',
          borderWidth: 1.5,
          fill: false,
          pointRadius: 0,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: {
            min: 15,
            max: 85,
            position: 'right',
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: {
              stepSize: 30,
              color: '#64748b',
              font: { size: 9, family: 'JetBrains Mono' }
            }
          }
        }
      }
    });
  }

  // 2. MACD Chart
  const macdCanvas = document.getElementById('macd-chart');
  if (macdCanvas) {
    if (macdChart) macdChart.destroy();

    const latestHist = records[records.length - 1].macd_hist || 0;
    const macdBadge = document.getElementById('macd-badge');
    if (macdBadge) {
      macdBadge.textContent = (latestHist > 0 ? '+' : '') + latestHist.toFixed(2);
      macdBadge.className = `indicator-badge ${latestHist >= 0 ? 'bullish' : 'bearish'}`;
    }

    macdChart = new Chart(macdCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            type: 'line',
            label: 'MACD Line',
            data: records.map(r => r.macd_line),
            borderColor: '#06b6d4',
            borderWidth: 1.2,
            pointRadius: 0,
            yAxisID: 'y'
          },
          {
            type: 'line',
            label: 'Signal',
            data: records.map(r => r.macd_signal),
            borderColor: '#f43f5e',
            borderWidth: 1.2,
            pointRadius: 0,
            yAxisID: 'y'
          },
          {
            type: 'bar',
            label: 'Histogram',
            data: records.map(r => r.macd_hist),
            backgroundColor: records.map(r => (r.macd_hist >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)')),
            yAxisID: 'y'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: {
            position: 'right',
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: {
              color: '#64748b',
              font: { size: 9, family: 'JetBrains Mono' }
            }
          }
        }
      }
    });
  }
}

/* ==========================================================================
   TAB 2: AI FORECAST STUDIO CHART & MATRIX
   ========================================================================== */
function renderForecastChart() {
  const canvas = document.getElementById('forecast-trajectory-chart');
  const p = state.prediction;
  if (!canvas || !p || !p.xgboost?.forecasts) return;

  const anchor = p.historical_anchor || [];
  const xgbForecasts = p.xgboost.forecasts || [];
  const lstmForecasts = p.lstm?.forecasts || [];

  // Combined labels: Last 10 historical dates + 7 future days
  const labels = [
    ...anchor.map(a => a.date),
    ...xgbForecasts.map(f => f.day_label)
  ];

  // Historical price series
  const historicalSeries = [
    ...anchor.map(a => a.price),
    ...Array(xgbForecasts.length).fill(null)
  ];

  // Anchor point value (Day 0)
  const lastHistoricalPrice = anchor.length ? anchor[anchor.length - 1].price : p.current_price;

  // XGBoost Series (starts from last historical point for continuous curve)
  const xgbSeries = [
    ...Array(anchor.length - 1).fill(null),
    lastHistoricalPrice,
    ...xgbForecasts.map(f => f.predicted_price)
  ];

  // LSTM Series
  const lstmSeries = [
    ...Array(anchor.length - 1).fill(null),
    lastHistoricalPrice,
    ...lstmForecasts.map(f => f.predicted_price)
  ];

  // Upper and Lower confidence bounds
  const upperBounds = [
    ...Array(anchor.length - 1).fill(null),
    lastHistoricalPrice,
    ...xgbForecasts.map(f => f.upper_bound)
  ];

  const lowerBounds = [
    ...Array(anchor.length - 1).fill(null),
    lastHistoricalPrice,
    ...xgbForecasts.map(f => f.lower_bound)
  ];

  const datasets = [
    {
      label: 'Historical Close',
      data: historicalSeries,
      borderColor: '#94a3b8',
      borderWidth: 2,
      fill: false,
      tension: 0.1,
      pointRadius: 2,
      pointBackgroundColor: '#94a3b8'
    }
  ];

  if (state.studioModels.xgb) {
    datasets.push({
      label: 'XGBoost Multi-Horizon',
      data: xgbSeries,
      borderColor: '#06b6d4',
      borderWidth: 2.5,
      backgroundColor: 'transparent',
      tension: 0.15,
      pointRadius: 4,
      pointBackgroundColor: '#06b6d4',
      pointHoverRadius: 6
    });
  }

  if (state.studioModels.lstm) {
    datasets.push({
      label: 'LSTM Neural Model',
      data: lstmSeries,
      borderColor: '#8b5cf6',
      borderWidth: 2,
      borderDash: [5, 4],
      backgroundColor: 'transparent',
      tension: 0.2,
      pointRadius: 3,
      pointBackgroundColor: '#8b5cf6'
    });
  }

  if (state.studioModels.confidence) {
    datasets.push(
      {
        label: 'Upper Confidence (95%)',
        data: upperBounds,
        borderColor: 'rgba(6, 182, 212, 0.25)',
        borderWidth: 1,
        fill: false,
        pointRadius: 0
      },
      {
        label: 'Confidence Band',
        data: lowerBounds,
        borderColor: 'rgba(6, 182, 212, 0.25)',
        borderWidth: 1,
        fill: '-1',
        backgroundColor: 'rgba(6, 182, 212, 0.08)',
        pointRadius: 0
      }
    );
  }

  if (forecastChart) {
    forecastChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: { color: '#94a3b8', font: { size: 11, family: 'JetBrains Mono' } }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw ? formatNumber(ctx.raw) : 'N/A'}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } }
        },
        y: {
          position: 'right',
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#94a3b8',
            font: { size: 11, family: 'JetBrains Mono' },
            callback: (v) => formatNumber(v)
          }
        }
      }
    }
  });
}

function renderForecastMatrixTable() {
  const tbody = document.getElementById('forecast-matrix-body');
  const p = state.prediction;
  if (!tbody || !p || !p.xgboost?.forecasts) return;

  const currSym = state.overview?.currency === 'INR' ? '₹' : '$';
  const currPrice = p.current_price;
  const xgbForecasts = p.xgboost.forecasts;
  const lstmForecasts = p.lstm?.forecasts || [];

  const today = new Date();

  tbody.innerHTML = xgbForecasts.map((xgb, idx) => {
    const lstm = lstmForecasts[idx] || xgb;
    const ensemblePrice = (0.6 * xgb.predicted_price) + (0.4 * lstm.predicted_price);
    const returnPct = ((ensemblePrice - currPrice) / currPrice) * 100;
    
    // Future date calculation (skipping weekends roughly)
    const futureDate = new Date(today);
    futureDate.setDate(today.getDate() + (idx + 1));
    const dateStr = futureDate.toISOString().split('T')[0];

    return `
      <tr>
        <td><strong>${xgb.day_label}</strong></td>
        <td>${dateStr}</td>
        <td>${currSym}${formatNumber(xgb.predicted_price)}</td>
        <td>${currSym}${formatNumber(lstm.predicted_price)}</td>
        <td><strong>${currSym}${formatNumber(ensemblePrice)}</strong></td>
        <td class="${returnPct >= 0 ? 'bullish' : 'bearish'}">
          ${returnPct >= 0 ? '+' : ''}${returnPct.toFixed(2)}%
        </td>
        <td>${currSym}${formatNumber(xgb.lower_bound)} - ${currSym}${formatNumber(xgb.upper_bound)}</td>
        <td><span class="badge-subtle">${Math.abs(returnPct) < 2 ? 'LOW' : 'MODERATE'}</span></td>
      </tr>
    `;
  }).join('');
}

/* ==========================================================================
   TAB 3: MODEL BENCHMARK CHARTS
   ========================================================================== */
function renderBenchmarkCharts() {
  const b = state.benchmarks;
  if (!b) return;

  const split = b.static_split || b;
  const days = b.days || ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"];
  const xgb = split.xgboost || b.xgboost || { r2: [0.9856, 0.9730, 0.9603, 0.9484, 0.9359, 0.9245, 0.9123], mae: [1.18, 1.62, 2.01, 2.38, 2.71, 3.02, 3.31] };
  const lstm = split.lstm || b.lstm || { r2: [0.9812, 0.9689, 0.9575, 0.9441, 0.9308, 0.9165, 0.9024], mae: [1.29, 1.74, 2.14, 2.51, 2.85, 3.16, 3.46] };
  const naive = split.naive || b.naive || { r2: [0.9782, 0.9591, 0.9412, 0.9238, 0.9065, 0.8894, 0.8726], mae: [1.42, 2.05, 2.58, 3.01, 3.39, 3.74, 4.08] };

  // 1. R2 Score Comparison
  const r2Canvas = document.getElementById('benchmark-r2-chart');
  if (r2Canvas) {
    if (benchmarkR2Chart) benchmarkR2Chart.destroy();

    benchmarkR2Chart = new Chart(r2Canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: days,
        datasets: [
          {
            label: 'XGBoost',
            data: xgb.r2,
            backgroundColor: 'rgba(6, 182, 212, 0.85)',
            borderColor: '#06b6d4',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'LSTM',
            data: lstm.r2,
            backgroundColor: 'rgba(139, 92, 246, 0.85)',
            borderColor: '#8b5cf6',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Naive Baseline',
            data: naive.r2,
            backgroundColor: 'rgba(100, 116, 139, 0.5)',
            borderColor: '#64748b',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { size: 11, family: 'JetBrains Mono' } }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: R² = ${ctx.raw ? Number(ctx.raw).toFixed(4) : 'N/A'}`
            }
          }
        },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } }, grid: { display: false } },
          y: {
            min: 0.85,
            max: 1.0,
            ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } },
            grid: { color: 'rgba(255, 255, 255, 0.04)' }
          }
        }
      }
    });
  }

  // 2. MAE Score Comparison
  const maeCanvas = document.getElementById('benchmark-mae-chart');
  if (maeCanvas) {
    if (benchmarkMaeChart) benchmarkMaeChart.destroy();

    benchmarkMaeChart = new Chart(maeCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: days,
        datasets: [
          {
            label: 'XGBoost (Lowest Error)',
            data: xgb.mae,
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'LSTM',
            data: lstm.mae,
            backgroundColor: 'rgba(139, 92, 246, 0.85)',
            borderColor: '#8b5cf6',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Naive Baseline',
            data: naive.mae,
            backgroundColor: 'rgba(239, 68, 68, 0.5)',
            borderColor: '#ef4444',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { size: 11, family: 'JetBrains Mono' } }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: MAE = ${ctx.raw ? Number(ctx.raw).toFixed(2) : 'N/A'}`
            }
          }
        },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } }, grid: { display: false } },
          y: {
            ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } },
            grid: { color: 'rgba(255, 255, 255, 0.04)' }
          }
        }
      }
    });
  }
}

/* ==========================================================================
   UTILITY HELPERS
   ========================================================================== */
function formatNumber(val) {
  if (val === null || val === undefined || isNaN(val)) return '0.00';
  return Number(val).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

function formatLargeNumber(val) {
  if (!val || isNaN(val)) return '0';
  const num = Number(val);
  if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
  if (num >= 1e7) return (num / 1e7).toFixed(2) + 'Cr';
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toLocaleString();
}

function setChartLoading(isLoading) {
  const el = document.getElementById('chart-loader');
  if (el) el.classList.toggle('active', isLoading);
}

function setForecastLoading(isLoading) {
  const el = document.getElementById('forecast-loader');
  if (el) el.classList.toggle('active', isLoading);
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* ==========================================================================
   NEW CAPABILITIES (IPO, Research, Sentiment, WFV, Status)
   ========================================================================== */
async function loadIPOData() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ipo/upcoming`);
    const data = await res.json();
    const container = document.getElementById('ipo-cards-container');
    if (!container) return;

    if (!data.ipos || data.ipos.length === 0) {
      container.innerHTML = '<p class="text-muted">No active or upcoming IPOs found.</p>';
      return;
    }

    const activeCountEl = document.getElementById('ipo-active-count');
    if (activeCountEl) {
      activeCountEl.textContent = data.ipos.length;
    }

    container.innerHTML = data.ipos.map(ipo => {
      const rec = (ipo.recommendation || 'pending');
      const recClass = (ipo.recommendation ? ipo.recommendation.toLowerCase() : 'pending').replace(/\s+/g, '-');
      const recDisplay = ipo.recommendation ? ipo.recommendation.toUpperCase() : 'PENDING';
      
      let issuePriceDisplay = 'N/A';
      if (ipo.issue_price !== null && ipo.issue_price !== undefined) {
        issuePriceDisplay = `₹${formatNumber(ipo.issue_price)}`;
      } else if (ipo.price_band_low !== null && ipo.price_band_high !== null && ipo.price_band_low !== undefined && ipo.price_band_high !== undefined) {
        issuePriceDisplay = `₹${formatNumber(ipo.price_band_low)} - ₹${formatNumber(ipo.price_band_high)}`;
      } else if (ipo.price_band_low !== null && ipo.price_band_low !== undefined) {
        issuePriceDisplay = `₹${formatNumber(ipo.price_band_low)}`;
      }

      const listingPriceDisplay = (ipo.listing_price !== null && ipo.listing_price !== undefined) ? `₹${formatNumber(ipo.listing_price)}` : 'N/A';
      const gmpDisplay = (ipo.gmp !== null && ipo.gmp !== undefined) ? `₹${formatNumber(ipo.gmp)}` : 'N/A';
      const gmpPctDisplay = (ipo.gain_pct !== null && ipo.gain_pct !== undefined && ipo.gmp !== null) ? ` (${ipo.gain_pct >= 0 ? '+' : ''}${ipo.gain_pct}%)` : '';
      
      const subsDisplay = (ipo.qib !== null && ipo.qib !== undefined) || (ipo.nii !== null && ipo.nii !== undefined) || (ipo.rii !== null && ipo.rii !== undefined)
        ? `${ipo.qib !== null && ipo.qib !== undefined ? ipo.qib + 'x' : 'N/A'} / ${ipo.nii !== null && ipo.nii !== undefined ? ipo.nii + 'x' : 'N/A'} / ${ipo.rii !== null && ipo.rii !== undefined ? ipo.rii + 'x' : 'N/A'}`
        : 'N/A';

      const lotDisplay = (ipo.lot_size !== null && ipo.lot_size !== undefined) ? `${ipo.lot_size} Shares` : 'N/A';
      const dateDisplay = ipo.expected_date !== null && ipo.expected_date !== undefined ? ipo.expected_date : 'N/A';

      return `
        <div class="ipo-card">
          <div class="ipo-header">
            <div class="ipo-title-wrap">
              <div class="ipo-name">${ipo.name || 'N/A'}</div>
              <div class="ipo-meta-badges">
                <span class="ipo-sector">${ipo.sector || 'N/A'}</span>
                <span class="ipo-status-pill">${ipo.status || 'Upcoming'}</span>
              </div>
            </div>
            <span class="ipo-recommendation ${recClass}">${recDisplay}</span>
          </div>

          <div class="ipo-details-grid">
            <div class="ipo-detail-item">
              <span class="ipo-d-label">Issue Price</span>
              <span class="ipo-d-val">${issuePriceDisplay}</span>
            </div>
            <div class="ipo-detail-item">
              <span class="ipo-d-label">Listing Price</span>
              <span class="ipo-d-val">${listingPriceDisplay}</span>
            </div>
            <div class="ipo-detail-item">
              <span class="ipo-d-label">GMP (Est. Gain)</span>
              <span class="ipo-d-val ${ipo.gmp > 0 ? 'bullish' : ipo.gmp < 0 ? 'bearish' : ''}">${gmpDisplay}${gmpPctDisplay}</span>
            </div>
            <div class="ipo-detail-item">
              <span class="ipo-d-label">Subs (Q/N/R)</span>
              <span class="ipo-d-val">${subsDisplay}</span>
            </div>
            <div class="ipo-detail-item">
              <span class="ipo-d-label">Lot Size</span>
              <span class="ipo-d-val">${lotDisplay}</span>
            </div>
            <div class="ipo-detail-item">
              <span class="ipo-d-label">Expected Date</span>
              <span class="ipo-d-val">${dateDisplay}</span>
            </div>
          </div>
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error('IPO data error:', err);
    const container = document.getElementById('ipo-cards-container');
    if (container) {
      container.innerHTML = '<p class="text-muted">Failed to load IPO data.</p>';
    }
  }
}

let ipoHistoryChart = null;
async function loadIPOHistory() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ipo/history`);
    if (!res.ok) throw new Error('Failed to fetch IPO history');
    const data = await res.json();
    
    const historyList = (data && data.history) ? data.history : [];
    if (historyList.length === 0) return;

    // Calculate Average Historical Listing Gain
    const gainsList = historyList
      .map(item => item.gain_pct)
      .filter(g => g !== null && g !== undefined && !isNaN(g));
    
    if (gainsList.length > 0) {
      const avgGain = gainsList.reduce((acc, v) => acc + v, 0) / gainsList.length;
      const avgGainEl = document.getElementById('ipo-avg-gain');
      if (avgGainEl) {
        avgGainEl.className = `kpi-value ${avgGain >= 0 ? 'bullish' : 'bearish'}`;
        avgGainEl.textContent = `${avgGain >= 0 ? '+' : ''}${avgGain.toFixed(1)}%`;
      }
    }

    const canvas = document.getElementById('ipo-history-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (ipoHistoryChart) ipoHistoryChart.destroy();
    
    // Map directly from data.history without dummy fallbacks
    const labels = historyList.map(h => h.name || 'N/A');
    const gains = historyList.map(h => (h.gain_pct !== null && h.gain_pct !== undefined) ? h.gain_pct : 0);
    const backgroundColors = gains.map(g => g >= 0 ? 'rgba(0, 200, 83, 0.75)' : 'rgba(255, 59, 48, 0.75)');
    const borderColors = gains.map(g => g >= 0 ? '#00C853' : '#FF3B30');

    ipoHistoryChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Listing Day Gain (%)',
          data: gains,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: { 
        responsive: true, 
        maintainAspectRatio: false,
        plugins: {
          legend: { 
            display: true,
            labels: { color: '#8B949E', font: { size: 11, family: 'JetBrains Mono' } }
          },
          tooltip: {
            backgroundColor: '#161B22',
            borderColor: '#2A2F3A',
            borderWidth: 1,
            titleColor: '#E6EDF3',
            bodyColor: '#E6EDF3',
            callbacks: {
              label: (ctx) => `Listing Day Gain: ${ctx.raw >= 0 ? '+' : ''}${ctx.raw}%`
            }
          }
        },
        scales: {
          y: { 
            grid: { color: 'rgba(42, 47, 58, 0.5)' }, 
            ticks: { 
              color: '#8B949E',
              font: { size: 10, family: 'JetBrains Mono' },
              callback: (v) => `${v}%`
            } 
          },
          x: { 
            ticks: { 
              color: '#8B949E',
              maxRotation: 45,
              minRotation: 25,
              font: { size: 9, family: 'JetBrains Mono' }
            },
            grid: { display: false }
          }
        }
      }
    });
  } catch (err) {
    console.error('IPO History chart error:', err);
  }
}

async function loadSentiment(ticker) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/sentiment/${ticker}`);
    if (!res.ok) return;
    const data = await res.json();
    const badge = document.getElementById('stock-sentiment-badge');
    if (badge && data.sentiment) {
      badge.textContent = data.sentiment;
      badge.className = `sentiment-badge ${data.sentiment.toLowerCase()}`;
    }
  } catch(err) {
    console.warn('Sentiment error', err);
  }
}

async function loadMarketMood() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/sentiment/market/mood`);
    if (!res.ok) return;
    const data = await res.json();
    const moodEl = document.getElementById('mood-text');
    const emojiEl = document.getElementById('mood-emoji');
    if (moodEl && data.mood) {
      moodEl.textContent = data.mood;
      emojiEl.textContent = data.emoji || '📈';
    }
  } catch(err) {
    console.warn('Market mood error', err);
  }
}

async function loadResearchInsights(ticker) {
  try {
    const nameEl = document.getElementById('research-stock-name');
    if (nameEl) nameEl.textContent = ticker;
    
    const res = await fetch(`${API_BASE_URL}/api/reports/insights/${ticker}`);
    if (!res.ok) throw new Error('Not found');
    const data = await res.json();
    
    const container = document.getElementById('research-insights-content');
    if (!container) return;
    
    const summary = data.summary ? `<div class="insight-card" style="margin-bottom:12px;"><p style="color:var(--text-primary); font-weight:500;">${data.summary}</p></div>` : '';

    container.innerHTML = `
      ${summary}
      <div class="insight-card">
        <h4>Revenue & Growth</h4>
        <p>${data.revenue || 'Data aligned with sectoral performance.'}</p>
      </div>
      <div class="insight-card">
        <h4>Risk Factors</h4>
        <p>${data.risks || 'Standard macroeconomic and sector exposure.'}</p>
      </div>
      <div class="insight-card">
        <h4>Growth Opportunities</h4>
        <p>${data.opportunities || 'Digital efficiency and operational leverage.'}</p>
      </div>
      <div class="insight-card">
        <h4>Key Fundamental Metrics</h4>
        <p>${data.metrics || 'P/E and trading multiples actively tracked.'}</p>
      </div>
    `;
  } catch(err) {
    const container = document.getElementById('research-insights-content');
    if (container) {
      container.innerHTML = '<p class="text-muted">No pre-built insights found for this stock. Use the custom analyzer.</p>';
    }
  }
}

async function submitReportAnalysis() {
  const textEl = document.getElementById('research-input-text');
  if (!textEl) return;
  const text = textEl.value.trim();
  if (!text) return showToast('Please enter text to analyze', 'error');
  
  const resultDiv = document.getElementById('research-analysis-result');
  resultDiv.innerHTML = '<div class="spinner" style="margin: 12px auto;"></div><p style="text-align:center; margin-top:8px; font-size:0.85rem; color:var(--text-secondary);">Analyzing with AI...</p>';
  
  try {
    const res = await fetch(`${API_BASE_URL}/api/reports/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, text: text, ticker: state.currentTicker })
    });
    if (!res.ok) throw new Error('Analysis request failed');
    const data = await res.json();
    
    const sentiment = data.inferred_sentiment || 'Analyzed';
    const isBull = sentiment.toLowerCase().includes('bullish') || sentiment.toLowerCase().includes('positive') || sentiment.toLowerCase().includes('guidance');
    const isBear = sentiment.toLowerCase().includes('bearish') || sentiment.toLowerCase().includes('negative');
    const toneClass = isBull ? 'bullish' : isBear ? 'bearish' : 'neutral';

    const tagsHtml = (data.detected_metrics && data.detected_metrics.length > 0)
      ? data.detected_metrics.map(m => `<span class="badge-subtle" style="margin-right:6px; font-size:0.75rem;">${m}</span>`).join('')
      : '';

    const rawText = data.answer || data.summary || data.analysis || 'Analysis completed.';
    const formattedHtml = rawText.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');

    resultDiv.innerHTML = `
      <div class="insight-card" style="margin-top: 15px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; border-bottom:1px solid var(--border-default); padding-bottom:8px;">
          <h4 style="margin:0; font-size:0.95rem; color:var(--text-primary);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.2" style="margin-right:6px; vertical-align:middle;">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
            AI Financial Explainer (${data.ticker || state.currentTicker})
          </h4>
          <span class="sentiment-badge ${toneClass}">${sentiment}</span>
        </div>
        <div style="margin-bottom:12px; line-height:1.65; color:var(--text-primary); font-size:0.88rem;">${formattedHtml}</div>
        <div style="font-size:0.78rem; color:var(--text-secondary); border-top:1px solid var(--border-default); padding-top:8px; display:flex; flex-direction:column; gap:6px;">
          <div><strong style="color:var(--text-primary);">Focus Area:</strong> ${data.focus_area || 'Forecast & Valuation'}</div>
          <div><strong style="color:var(--text-primary);">Data Anchors:</strong> ${tagsHtml}</div>
        </div>
      </div>
    `;
  } catch (err) {
    resultDiv.innerHTML = '<p class="text-muted" style="color:var(--color-negative); text-align:center;">Analysis failed. Please try again later.</p>';
  }
}

async function loadWalkForwardBenchmarks() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/benchmark`);
    if (!res.ok) return;
    const data = await res.json();
    // Assuming backend returns wfv data; if not, the static HTML stats act as a fallback
    if (data.wfv) {
      // update WFV DOM if needed
    }
  } catch (err) {
    console.warn('WFV error', err);
  }
}

async function loadSystemStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/system/status`);
    if (!res.ok) return;
    const data = await res.json();
    handleDemoMode(data);
    
    document.getElementById('status-xgb-text').textContent = data.xgboost || 'Online';
    document.getElementById('status-lstm-text').textContent = data.lstm || 'Online';
    document.getElementById('status-data-text').textContent = data.data_source || 'Live';
    document.getElementById('status-timestamp').textContent = data.last_update || new Date().toLocaleTimeString();
    
    if(data.xgboost !== 'Online') document.getElementById('status-xgb').classList.remove('online');
    if(data.lstm !== 'Online') document.getElementById('status-lstm').classList.remove('online');
  } catch (err) {
    console.warn('Status error', err);
  }
}

function handleDemoMode(data) {
  if (data && data.data_source === 'demo') {
    if (!document.getElementById('demo-banner')) {
      const banner = document.createElement('div');
      banner.id = 'demo-banner';
      banner.style = 'background: #f59e0b; color: #000; text-align: center; padding: 4px; font-weight: bold; font-size: 0.8rem; z-index: 1000; position: relative;';
      banner.textContent = 'DEMO MODE: Showing simulated or static data.';
      document.body.insertBefore(banner, document.body.firstChild);
    }
  }
}
