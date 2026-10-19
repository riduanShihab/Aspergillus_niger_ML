// Aspergillus niger ML Predictor Client Application

const state = {
    strain: "4",
    substrate: "orange peel",
    ph: 5.5,
    time_days: 5,
    temperature: 28.0,
    history: [],
    presets: [],
    lastResponse: null
};

const substrateIcons = {
    "orange peel": "🍊",
    "pectin": "🧪",
    "banana peel": "🍌",
    "wheat bran": "🌾",
    "rice bran": "🍚",
    "potato peel": "🥔"
};

// Initial DOM Load
document.addEventListener("DOMContentLoaded", async () => {
    await initSubstrates();
    await initPresets();
    setupCanvas();
    // Auto-run initial prediction
    handlePredict();
});

// Initialize Substrate Tiles
async function initSubstrates() {
    const substrates = ["orange peel", "pectin", "banana peel", "wheat bran", "rice bran", "potato peel"];
    const grid = document.getElementById("substrateGrid");
    grid.innerHTML = "";

    substrates.forEach(sub => {
        const tile = document.createElement("div");
        tile.className = `substrate-tile ${sub === state.substrate ? "active" : ""}`;
        tile.onclick = () => selectSubstrate(sub);
        tile.id = `sub-tile-${sub.replace(/\s+/g, "-")}`;
        
        const icon = substrateIcons[sub] || "🌱";
        tile.innerHTML = `
            <span class="sub-icon">${icon}</span>
            <span class="sub-title">${sub}</span>
        `;
        grid.appendChild(tile);
    });
}

// Fetch Presets from Backend
async function initPresets() {
    try {
        const res = await fetch("/api/presets");
        if (!res.ok) return;
        state.presets = await res.json();
        
        const container = document.getElementById("presetsList");
        container.innerHTML = "";
        
        state.presets.forEach(p => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "preset-chip";
            btn.onclick = () => loadPreset(p);
            btn.innerHTML = `
                <span>${p.name}</span>
                <span class="chip-tag">${p.tag}</span>
            `;
            container.appendChild(btn);
        });
    } catch (err) {
        console.warn("Could not load presets from server", err);
    }
}

// Load a selected Preset
function loadPreset(preset) {
    const p = preset.params;
    selectStrain(p.strain);
    selectSubstrate(p.substrate);
    
    document.getElementById("phSlider").value = p.ph;
    updatePh(p.ph);
    
    document.getElementById("timeSlider").value = p.time_days;
    updateTime(p.time_days);
    
    document.getElementById("tempSlider").value = p.temperature;
    updateTemp(p.temperature);
    
    handlePredict();
}

// Strain selection
function selectStrain(val) {
    state.strain = String(val);
    document.getElementById("strainInput").value = state.strain;
    
    document.querySelectorAll(".strain-selector .pill-btn").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-value") === state.strain);
    });
}

// Substrate selection
function selectSubstrate(sub) {
    state.substrate = sub;
    document.getElementById("substrateInput").value = state.substrate;
    
    document.querySelectorAll(".substrate-tile").forEach(tile => {
        tile.classList.toggle("active", tile.id === `sub-tile-${sub.replace(/\s+/g, "-")}`);
    });
}

// Numerical Sliders Live Updates
function updatePh(val) {
    state.ph = parseFloat(val);
    document.getElementById("phVal").innerText = Number(val).toFixed(1);
}

function updateTime(val) {
    state.time_days = parseInt(val, 10);
    document.getElementById("timeVal").innerText = `${val} days`;
}

function updateTemp(val) {
    state.temperature = parseFloat(val);
    const readout = document.getElementById("tempVal");
    readout.innerText = `${Number(val).toFixed(1)} °C`;
    
    if (state.temperature >= 65) {
        readout.style.color = "#f87171";
        readout.style.borderColor = "rgba(239, 68, 68, 0.4)";
        readout.style.background = "rgba(239, 68, 68, 0.15)";
    } else {
        readout.style.color = "#38bdf8";
        readout.style.borderColor = "rgba(6, 182, 212, 0.3)";
        readout.style.background = "rgba(6, 182, 212, 0.12)";
    }
}

// Main Predict Handler
async function handlePredict() {
    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.classList.add("loading");

    const payload = {
        strain: state.strain,
        ph: state.ph,
        substrate: state.substrate,
        time_days: state.time_days,
        temperature: state.temperature
    };

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Prediction request failed");
        }

        const data = await response.json();
        state.lastResponse = data;
        renderResults(data);
        addHistoryRow(payload, data);
    } catch (err) {
        alert("Prediction Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.classList.remove("loading");
    }
}

// Render Results & Metrics
function renderResults(data) {
    // 1. Number Counter Animation
    const targetVal = data.predicted_activity;
    animateNumber(document.getElementById("predActivityNumber"), targetVal, 600);

    // 2. Status Badge
    const statusBadge = document.getElementById("statusBadge");
    statusBadge.innerText = data.status;
    statusBadge.className = "status-badge";
    if (data.thermal_alert) {
        statusBadge.classList.add("status-danger");
    } else if (data.status.includes("Induction") || data.status.includes("Optimal")) {
        statusBadge.classList.add("status-optimal");
    } else {
        statusBadge.classList.add("status-warning");
    }

    // 3. 95% Confidence Interval Bar
    document.getElementById("ciBoundsText").innerText = `[${data.ci_95_lower.toFixed(1)} – ${data.ci_95_upper.toFixed(1)} U/mL]`;
    document.getElementById("stdErrText").innerText = `±${data.std_error.toFixed(1)} U/mL`;

    // Visual scale based on max expected activity (~1600 U/mL)
    const maxScale = 1600;
    const leftPercent = Math.max(0, Math.min(100, (data.ci_95_lower / maxScale) * 100));
    const rightPercent = Math.max(0, Math.min(100, (data.ci_95_upper / maxScale) * 100));
    const meanPercent = Math.max(0, Math.min(100, (data.predicted_activity / maxScale) * 100));

    const ciBar = document.getElementById("ciBar");
    ciBar.style.left = `${leftPercent}%`;
    ciBar.style.width = `${Math.max(4, rightPercent - leftPercent)}%`;

    const ciMarker = document.getElementById("ciMarker");
    ciMarker.style.left = `${meanPercent}%`;

    // 4. Biotech Insights Callout
    const alertBox = document.getElementById("biotechAlert");
    const alertIcon = document.getElementById("alertIcon");
    const alertTitle = document.getElementById("alertTitle");
    const alertDesc = document.getElementById("alertDescription");

    if (data.thermal_alert) {
        alertBox.style.borderColor = "rgba(239, 68, 68, 0.4)";
        alertBox.style.background = "rgba(239, 68, 68, 0.1)";
        alertIcon.innerText = "⚠️";
        alertTitle.style.color = "#f87171";
        alertTitle.innerText = "Thermal Inactivation Warning (≥65°C)";
    } else if (data.status.includes("Pectinolytic")) {
        alertBox.style.borderColor = "rgba(16, 185, 129, 0.4)";
        alertBox.style.background = "rgba(16, 185, 129, 0.1)";
        alertIcon.innerText = "🚀";
        alertTitle.style.color = "#34d399";
        alertTitle.innerText = "Optimal Pectinase Induction Region";
    } else {
        alertBox.style.borderColor = "rgba(6, 182, 212, 0.25)";
        alertBox.style.background = "rgba(6, 182, 212, 0.08)";
        alertIcon.innerText = "🔬";
        alertTitle.style.color = "#38bdf8";
        alertTitle.innerText = "Bioprocess Analysis";
    }
    alertDesc.innerText = data.bioprocess_notes;

    // 5. Draw Temperature Response Canvas Chart
    drawTempChart(data.temperature_curve, state.temperature, data.predicted_activity);
}

// Animate numbers smoothly
function animateNumber(element, finalValue, duration = 500) {
    const startVal = parseFloat(element.innerText) || 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = startVal + (finalValue - startVal) * easeOut;
        element.innerText = current.toFixed(1);

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.innerText = finalValue.toFixed(2);
        }
    }
    requestAnimationFrame(update);
}

// Canvas-Based Temperature Sensitivity Profile Chart
function setupCanvas() {
    const canvas = document.getElementById("tempChart");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
}

function drawTempChart(curveData, currentTemp, currentPred) {
    if (!curveData || curveData.length === 0) return;

    const canvas = document.getElementById("tempChart");
    const ctx = canvas.getContext("2d");
    const width = canvas.getBoundingClientRect().width;
    const height = canvas.getBoundingClientRect().height;

    ctx.clearRect(0, 0, width, height);

    const padding = { top: 20, right: 30, bottom: 35, left: 55 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Find min/max ranges
    const minTemp = 20;
    const maxTemp = 75;
    let maxAct = Math.max(1200, ...curveData.map(d => d.ci_upper));
    maxAct = Math.ceil(maxAct / 200) * 200;

    const getX = t => padding.left + ((t - minTemp) / (maxTemp - minTemp)) * chartW;
    const getY = y => padding.top + chartH - (Math.max(0, y) / maxAct) * chartH;

    // 1. Grid Lines & Axis Labels
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "10px JetBrains Mono";
    ctx.textAlign = "right";

    // Horizontal Y Grid
    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
        const val = (maxAct / ySteps) * i;
        const yPos = getY(val);
        ctx.beginPath();
        ctx.moveTo(padding.left, yPos);
        ctx.lineTo(width - padding.right, yPos);
        ctx.stroke();
        ctx.fillText(Math.round(val), padding.left - 8, yPos + 3);
    }

    // Vertical X Grid
    ctx.textAlign = "center";
    [25, 35, 45, 55, 65, 75].forEach(t => {
        const xPos = getX(t);
        ctx.beginPath();
        ctx.moveTo(xPos, padding.top);
        ctx.lineTo(xPos, padding.top + chartH);
        ctx.stroke();
        ctx.fillText(`${t}°C`, xPos, height - padding.bottom + 18);
    });

    // 2. Denaturation Zone Shade (>= 65°C)
    const xDenat = getX(65);
    ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
    ctx.fillRect(xDenat, padding.top, (width - padding.right) - xDenat, chartH);

    // 3. Shaded 95% Confidence Interval Ribbon
    ctx.beginPath();
    curveData.forEach((pt, i) => {
        const px = getX(pt.temperature);
        const py = getY(pt.ci_upper);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    });
    for (let i = curveData.length - 1; i >= 0; i--) {
        const pt = curveData[i];
        const px = getX(pt.temperature);
        const py = getY(pt.ci_lower);
        ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(6, 182, 212, 0.12)";
    ctx.fill();

    // 4. Mean Prediction Curve Line
    ctx.beginPath();
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2.5;
    curveData.forEach((pt, i) => {
        const px = getX(pt.temperature);
        const py = getY(pt.predicted_activity);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // 5. Current Selected Temperature Marker Line & Point
    const curX = getX(currentTemp);
    const curY = getY(currentPred);

    // Dotted vertical line
    ctx.beginPath();
    ctx.strokeStyle = "rgba(239, 68, 68, 0.7)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.moveTo(curX, padding.top);
    ctx.lineTo(curX, padding.top + chartH);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // Outer glow ring
    ctx.beginPath();
    ctx.arc(curX, curY, 8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
    ctx.fill();

    // Inner marker dot
    ctx.beginPath();
    ctx.arc(curX, curY, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#ef4444";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
}

// Experiment History Log
function addHistoryRow(req, res) {
    state.history.unshift({
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        strain: req.strain,
        substrate: req.substrate,
        ph: req.ph.toFixed(1),
        duration: `${req.time_days}d`,
        temp: `${req.temperature.toFixed(1)}°C`,
        activity: res.predicted_activity.toFixed(2),
        interval: `[${res.ci_95_lower.toFixed(1)} - ${res.ci_95_upper.toFixed(1)}]`,
        status: res.status
    });

    if (state.history.length > 20) state.history.pop();
    renderHistoryTable();
}

function renderHistoryTable() {
    const tbody = document.getElementById("historyBody");
    const countBadge = document.getElementById("historyCount");
    countBadge.innerText = `${state.history.length} runs recorded`;

    if (state.history.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="9">No predictions run yet.</td></tr>`;
        return;
    }

    tbody.innerHTML = state.history.map(row => `
        <tr>
            <td>${row.time}</td>
            <td>Strain ${row.strain}</td>
            <td style="text-transform: capitalize;">${row.substrate}</td>
            <td>${row.ph}</td>
            <td>${row.duration}</td>
            <td>${row.temp}</td>
            <td><strong>${row.activity}</strong></td>
            <td style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-dim);">${row.interval}</td>
            <td><span class="badge" style="background: rgba(255,255,255,0.06); font-size: 0.72rem;">${row.status}</span></td>
        </tr>
    `).join("");
}

function clearHistory() {
    state.history = [];
    renderHistoryTable();
}

function exportHistoryCSV() {
    if (state.history.length === 0) {
        alert("No experimental history to export yet.");
        return;
    }

    const headers = ["Timestamp", "Strain", "Substrate", "pH", "Duration", "Temperature", "Predicted_Activity_U_per_mL", "CI_95_Interval", "Status"];
    const rows = state.history.map(r => [
        `"${r.time}"`,
        `"${r.strain}"`,
        `"${r.substrate}"`,
        r.ph,
        `"${r.duration}"`,
        `"${r.temp}"`,
        r.activity,
        `"${r.interval}"`,
        `"${r.status}"`
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `aspergillus_niger_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
window.addEventListener("resize", () => {
    setupCanvas();
    if (state.lastResponse) {
        drawTempChart(state.lastResponse.temperature_curve, state.temperature, state.lastResponse.predicted_activity);
    }
});
