/* ─────────────────────────────────────────────────────────────────────────
   Digit Oracle — script.js
   Handles: canvas drawing, 28×28 downsampling, API call, result rendering
───────────────────────────────────────────────────────────────────────── */

const API_BASE = "http://localhost:5000";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const drawCanvas    = document.getElementById("drawCanvas");
const previewCanvas = document.getElementById("previewCanvas");
const ctx           = drawCanvas.getContext("2d");
const prevCtx       = previewCanvas.getContext("2d");
const clearBtn      = document.getElementById("clearBtn");
const predictBtn    = document.getElementById("predictBtn");
const resultDisplay = document.getElementById("resultDisplay");
const probSection   = document.getElementById("probSection");
const probBars      = document.getElementById("probBars");
const loadingOv     = document.getElementById("loadingOverlay");
const canvasHint    = document.getElementById("canvasHint");
const inferTime     = document.getElementById("inferenceTime");
const serverStatus  = document.getElementById("serverStatus");
const statusText    = document.getElementById("statusText");

// ── State ─────────────────────────────────────────────────────────────────────
let drawing   = false;
let brushSize = 22;
let hasDrawn  = false;

// Bar colours keyed by digit class
const CLASS_COLORS = [
  "#ff6b6b","#4ecdc4","#45b7d1","#f9ca24",
  "#6c5ce7","#fd79a8","#00cec9","#e17055",
  "#a29bfe","#55efc4"
];

// ── Canvas setup ──────────────────────────────────────────────────────────────
ctx.fillStyle = "#000";
ctx.fillRect(0, 0, 280, 280);
ctx.lineCap    = "round";
ctx.lineJoin   = "round";
ctx.strokeStyle = "#fff";
ctx.lineWidth   = brushSize;

// ── Drawing events ────────────────────────────────────────────────────────────
function getPos(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top)  * scaleY
  };
}

drawCanvas.addEventListener("mousedown",  startDraw);
drawCanvas.addEventListener("mousemove",  draw);
drawCanvas.addEventListener("mouseup",    stopDraw);
drawCanvas.addEventListener("mouseleave", stopDraw);
drawCanvas.addEventListener("touchstart", e => { e.preventDefault(); startDraw(e); }, { passive: false });
drawCanvas.addEventListener("touchmove",  e => { e.preventDefault(); draw(e); },      { passive: false });
drawCanvas.addEventListener("touchend",   stopDraw);

function startDraw(e) {
  drawing = true;
  if (!hasDrawn) {
    hasDrawn = true;
    canvasHint.classList.add("hidden");
  }
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  updatePreview();
}

function draw(e) {
  if (!drawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  updatePreview();
}

function stopDraw() {
  drawing = false;
  ctx.beginPath();
}

// ── Brush size buttons ────────────────────────────────────────────────────────
document.querySelectorAll(".brush-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".brush-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    brushSize = parseInt(btn.dataset.size);
    ctx.lineWidth = brushSize;
  });
});

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 280, 280);
  hasDrawn = false;
  canvasHint.classList.remove("hidden");
  resultDisplay.innerHTML = `
    <div class="result-idle">
      <div class="idle-digit">?</div>
      <p class="idle-msg">Draw a digit and hit PREDICT</p>
    </div>`;
  probSection.style.display = "none";
  updatePreview();
});

// ── Preview (28×28 downsampling) ──────────────────────────────────────────────
function updatePreview() {
  // Draw downsampled version for display
  prevCtx.imageSmoothingEnabled = true;
  prevCtx.imageSmoothingQuality = "high";
  prevCtx.drawImage(drawCanvas, 0, 0, 140, 140);
}

// Initialize preview
updatePreview();

// ── Get pixel data for model ──────────────────────────────────────────────────
function getPixels28x28() {
  // Downsample to 28×28 offscreen
  const off = document.createElement("canvas");
  off.width = off.height = 28;
  const offCtx = off.getContext("2d");
  offCtx.imageSmoothingEnabled = true;
  offCtx.imageSmoothingQuality = "high";
  offCtx.drawImage(drawCanvas, 0, 0, 28, 28);

  const imgData = offCtx.getImageData(0, 0, 28, 28);
  const pixels  = new Array(784);

  for (let i = 0; i < 784; i++) {
    // Grayscale from RGBA: use luminance of the pixel (canvas is BW so R=G=B)
    // MNIST uses white digit on black background
    const idx = i * 4;
    pixels[i] = imgData.data[idx]; // red channel (0–255), black bg = 0
  }
  return pixels;
}

// ── Predict ───────────────────────────────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
  if (!hasDrawn) {
    shakeCanvas();
    return;
  }

  const pixels = getPixels28x28();

  loadingOv.style.display = "flex";
  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels })
    });

    const elapsed = Math.round(performance.now() - t0);
    inferTime.textContent = elapsed;

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const data = await res.json();
    renderResult(data.prediction, data.probabilities, data.confidence);
  } catch (err) {
    renderError(err.message);
  } finally {
    loadingOv.style.display = "none";
  }
});

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(pred, probs, confidence) {
  const pct = Math.round(confidence * 100);

  resultDisplay.innerHTML = `
    <div class="pred-inner">
      <div class="pred-digit class-${pred}">${pred}</div>
      <div class="pred-confidence">
        CONFIDENCE <span class="pred-conf-val">${pct}%</span>
      </div>
    </div>`;

  // Build probability bars
  probBars.innerHTML = "";
  const sorted = probs
    .map((p, i) => ({ cls: i, prob: p }))
    .sort((a, b) => b.prob - a.prob);

  sorted.forEach(({ cls, prob }) => {
    const pct = (prob * 100).toFixed(1);
    const w   = Math.round(prob * 100);
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <span class="prob-bar-label">${cls}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:0%; background:${CLASS_COLORS[cls]}"></div>
      </div>
      <span class="prob-bar-pct">${pct}%</span>`;
    probBars.appendChild(row);

    // Animate bar fill
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector(".prob-bar-fill").style.width = `${w}%`;
      });
    });
  });

  probSection.style.display = "flex";
}

// ── Render error ──────────────────────────────────────────────────────────────
function renderError(msg) {
  resultDisplay.innerHTML = `
    <div class="error-msg">
      ⚠ ${msg}<br/>
      <small style="color:var(--text-dim);margin-top:8px;display:block">
        Make sure the Flask server is running on port 5000
      </small>
    </div>`;
}

// ── Shake animation for empty canvas ─────────────────────────────────────────
function shakeCanvas() {
  const w = drawCanvas.parentElement;
  w.style.transition = "transform 0.08s";
  let i = 0;
  const offsets = [6, -6, 5, -5, 3, -3, 0];
  const shake = () => {
    if (i >= offsets.length) { w.style.transform = ""; return; }
    w.style.transform = `translateX(${offsets[i++]}px)`;
    setTimeout(shake, 60);
  };
  shake();
}

// ── Server health check ───────────────────────────────────────────────────────
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      setStatus("connected", "Server connected — ready to predict");
    } else {
      setStatus("error", "Server returned error");
    }
  } catch {
    setStatus("error", "Server offline — run: python app.py");
  }
}

function setStatus(state, msg) {
  const dot = serverStatus.querySelector(".dot");
  dot.className = "dot pulse " + (state === "connected" ? "dot--green" : "dot--amber");
  statusText.textContent = msg;
}

// Check on load and every 5 seconds
checkServer();
setInterval(checkServer, 5000);

// ── Keyboard shortcut ─────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  if (e.key === "Enter")           predictBtn.click();
  if (e.key === "Escape" || e.key === "Delete") clearBtn.click();
});
