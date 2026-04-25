/**
 * app.js
 * ------
 * Phase 4 | Executed: Local (browser, served by nginx)
 *
 * Handles:
 * - Image upload via file picker and drag-and-drop
 * - Calls POST /predict on FastAPI backend
 * - Displays prediction result with color coding
 * - Feedback submission via POST /feedback
 * - System status check via GET /health and GET /ready
 */

const API_BASE = "http://localhost:8000";

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile       = null;
let currentPrediction = null;   // stores last prediction for feedback

// ── DOM references ─────────────────────────────────────────────────────────
const fileInput      = document.getElementById("file-input");
const dropzone       = document.getElementById("dropzone");
const dropzoneInner  = document.getElementById("dropzone-inner");
const previewImg     = document.getElementById("preview-img");
const btnAnalyse     = document.getElementById("btn-analyse");
const resultCard     = document.getElementById("result-card");
const resultHeader   = document.getElementById("result-header");
const resultIcon     = document.getElementById("result-icon");
const resultLabel    = document.getElementById("result-label");
const spinner        = document.getElementById("spinner");
const errorBanner    = document.getElementById("error-banner");
const errorMessage   = document.getElementById("error-message");
const feedbackStatus = document.getElementById("feedback-status");


// ── File selection ─────────────────────────────────────────────────────────

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
});

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
});


function handleFileSelected(file) {
    // Validate type
    if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
        showError("Only JPEG and PNG images are supported.");
        return;
    }

    currentFile = file;
    hideError();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.classList.remove("hidden");
        dropzoneInner.classList.add("hidden");
    };
    reader.readAsDataURL(file);

    // Enable analyse button
    btnAnalyse.disabled = false;
}


// ── Analyse ────────────────────────────────────────────────────────────────

btnAnalyse.addEventListener("click", async () => {
    if (!currentFile) return;
    await runPrediction(currentFile);
});


async function runPrediction(file) {
    showSpinner();
    hideError();
    resultCard.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        currentPrediction = { ...data };  // image_id comes from server response
        displayResult(data);

    } catch (err) {
        showError(`Prediction failed: ${err.message}`);
    } finally {
        hideSpinner();
    }
}


function displayResult(data) {
    const isMalignant = data.label === "malignant";

    // ── Result header ───────────────────────────────────────────────────
    resultHeader.className = `result-header ${data.label}`;
    resultIcon.textContent  = isMalignant ? "⚠️" : "✅";
    resultLabel.textContent = isMalignant ? "Malignant Detected" : "Benign — Low Risk";

    // ── Details ─────────────────────────────────────────────────────────
    document.getElementById("detail-confidence").textContent =
        `${(data.confidence * 100).toFixed(1)}%`;

    document.getElementById("detail-prob").textContent =
        `${(data.malignant_prob * 100).toFixed(1)}%`;

    document.getElementById("detail-threshold").textContent =
        data.threshold_used.toFixed(2);

    // ── Recommendation ──────────────────────────────────────────────────
    document.getElementById("recommendation").textContent =
        data.recommendation;

    // ── Reset feedback state ─────────────────────────────────────────────
    feedbackStatus.classList.add("hidden");
    feedbackStatus.textContent = "";

    // ── Show result card ─────────────────────────────────────────────────
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth" });
}


// ── Feedback ───────────────────────────────────────────────────────────────

async function submitFeedback(trueLabel) {
    if (!currentPrediction) return;

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_id:       currentPrediction.image_id,
                predicted_label: currentPrediction.label,
                true_label:     trueLabel
            })
        });

        if (!response.ok) throw new Error("Feedback submission failed.");

        feedbackStatus.textContent = "✅ Feedback recorded. Thank you!";
        feedbackStatus.classList.remove("hidden");

        // Disable feedback buttons after submission
        document.querySelectorAll(".btn-feedback").forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = "0.5";
        });

    } catch (err) {
        feedbackStatus.textContent = `❌ ${err.message}`;
        feedbackStatus.classList.remove("hidden");
    }
}


// ── Reset ──────────────────────────────────────────────────────────────────

function resetUI() {
    currentFile       = null;
    currentPrediction = null;

    fileInput.value        = "";
    previewImg.src         = "";
    previewImg.classList.add("hidden");
    dropzoneInner.classList.remove("hidden");
    btnAnalyse.disabled    = true;
    resultCard.classList.add("hidden");
    hideError();

    // Re-enable feedback buttons
    document.querySelectorAll(".btn-feedback").forEach(btn => {
        btn.disabled    = false;
        btn.style.opacity = "1";
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
}


// ── System status check ────────────────────────────────────────────────────

async function checkSystemStatus() {
    const dotApi   = document.getElementById("dot-api");
    const dotModel = document.getElementById("dot-model");
    const modelVersionText = document.getElementById("model-version-text");

    // Check /health
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            dotApi.className = "status-dot green";
        } else {
            dotApi.className = "status-dot red";
        }
    } catch {
        dotApi.className = "status-dot red";
    }

    // Check /ready
    try {
        const res = await fetch(`${API_BASE}/ready`);
        if (res.ok) {
            const data = await res.json();
            if (data.model_loaded) {
                dotModel.className = "status-dot green";
                modelVersionText.textContent =
                    `Model: ${data.model_name || "unknown"} | Version: ${data.model_version || "?"}`;
            } else {
                dotModel.className = "status-dot red";
                modelVersionText.textContent = "Model not loaded";
            }
        } else {
            dotModel.className = "status-dot red";
        }
    } catch {
        dotModel.className = "status-dot red";
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function showSpinner() { spinner.classList.remove("hidden"); }
function hideSpinner() { spinner.classList.add("hidden"); }

function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove("hidden");
}
function hideError() { errorBanner.classList.add("hidden"); }


// ── Init ───────────────────────────────────────────────────────────────────

// Check status on load and every 30 seconds
checkSystemStatus();
setInterval(checkSystemStatus, 30000);