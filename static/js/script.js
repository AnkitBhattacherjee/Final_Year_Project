/**
 * Frontend JavaScript Controller for AI Data Analyst Agent.
 * Manages view routing, AJAX API requests, dynamic UI renders, and Plotly graphics.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Navigation & View Routing Setup
    // -------------------------------------------------------------------------
    const navButtons = document.querySelectorAll(".nav-btn");
    const viewSections = document.querySelectorAll(".view-section");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetView = btn.getAttribute("data-view");
            
            navButtons.forEach(b => b.classList.remove("active"));
            viewSections.forEach(s => s.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(targetView).classList.add("active");

            // Lazy load view content if dataset is loaded
            if (targetView === "profile-view") fetchDatasetProfile();
            if (targetView === "quality-view") fetchQualityReport();

            // FIX: Force Plotly chart to recalculate width when tab becomes visible
            if (targetView === "visualizations-view") {
                setTimeout(() => {
                    const chartContainer = document.getElementById("plotly-chart-container");
                    if (chartContainer && chartContainer.data) {
                        Plotly.Plots.resize(chartContainer);
                    }
                }, 100);
            }
        });
    });

    // -------------------------------------------------------------------------
    // 2. File Upload Handling
    // -------------------------------------------------------------------------
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const filenameDisplay = document.getElementById("upload-filename-display");
    const loader = document.getElementById("global-loader");

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            filenameDisplay.textContent = `Selected: ${fileInput.files[0].name}`;
        }
    });

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) {
            alert("Please select a file to upload first.");
            return;
        }

        const formData = new FormData();
        formData.append("dataset", fileInput.files[0]);

        showLoader("Uploading & analyzing dataset...");

        try {
            const res = await fetch("/upload", { method: "POST", body: formData });
            const text = await res.text();
            let data = null;
            try {
                data = JSON.parse(text);
            } catch (jsonErr) {
                console.error("Non-JSON server response:", text);
                alert(`Upload failed (Status ${res.status}): ${text.slice(0, 200) || res.statusText || "Server error"}`);
                return;
            }

            if (res.ok && data) {
                // Update Badge Header
                document.getElementById("badge-filename").textContent = data.filename;
                document.getElementById("active-file-badge").classList.remove("hidden");
                
                // Render Dashboard
                renderDashboardMetrics(data.metrics);
                await fetchDatasetInfo();
                await fetchSuggestions();
            } else {
                alert((data && data.error) || `Upload failed with status ${res.status}.`);
            }
        } catch (err) {
            console.error("Upload fetch error:", err);
            alert("Upload network error: " + (err.message || "Please check server connection."));
        } finally {
            hideLoader();
        }
    });

    // -------------------------------------------------------------------------
    // 3. Dataset Info & Select Box Population
    // -------------------------------------------------------------------------
    async function fetchDatasetInfo() {
        try {
            const res = await fetch("/dataset");
            const data = await res.json();

            if (res.ok) {
                document.getElementById("empty-dashboard-state").classList.add("hidden");
                document.getElementById("dashboard-content").classList.remove("hidden");

                renderPreviewTable(data.preview);
                populateAxisDropdowns(data.columns, data.numeric_columns);
            }
        } catch (err) {
            console.error("Error fetching dataset info:", err);
        }
    }

    function renderDashboardMetrics(m) {
        document.getElementById("m-rows").textContent = m.total_rows.toLocaleString();
        document.getElementById("m-cols").textContent = m.total_columns;
        document.getElementById("m-numeric").textContent = m.numeric_columns_count;
        document.getElementById("m-missing").textContent = m.missing_cells.toLocaleString();
        document.getElementById("m-duplicates").textContent = m.duplicate_rows;
    }

    function renderPreviewTable(records) {
        if (!records || !records.length) return;
        
        const table = document.getElementById("preview-table");
        const thead = table.querySelector("thead");
        const tbody = table.querySelector("tbody");

        const headers = Object.keys(records[0]);
        thead.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr>`;
        
        tbody.innerHTML = records.map(row => `
            <tr>${headers.map(h => `<td>${row[h] !== null ? row[h] : ""}</td>`).join("")}</tr>
        `).join("");
    }

    function populateAxisDropdowns(allCols, numCols) {
        const xAxisSelect = document.getElementById("x-axis-select");
        const yAxisSelect = document.getElementById("y-axis-select");

        xAxisSelect.innerHTML = allCols.map(c => `<option value="${c}">${c}</option>`).join("");
        yAxisSelect.innerHTML = numCols.map(c => `<option value="${c}">${c}</option>`).join("");
    }

    // -------------------------------------------------------------------------
    // 4. AI Analyst Chat Operations
    // -------------------------------------------------------------------------
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const clearChatBtn = document.getElementById("clear-chat-btn");

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;

        appendChatMessage("user", question);
        chatInput.value = "";

        showLoader("AI Agent thinking...");

        try {
            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });
            const data = await res.json();

            if (res.ok) {
                appendChatMessage("assistant", data.answer);
                if (data.chart) {
                    appendChatChart(data.chart);
                }
            } else {
                appendChatMessage("assistant", `⚠️ Error: ${data.error}`);
            }
        } catch (err) {
            appendChatMessage("assistant", "⚠️ Server connection failure.");
        } finally {
            hideLoader();
        }
    });

    /**
     * Converts a subset of Markdown to clean HTML for chat display.
     * Handles: **bold**, *italic*, `code`, bullet lists (- / *), and newlines.
     */
    function formatMarkdown(text) {
        if (!text) return "";

        // Escape raw HTML entities to prevent XSS before re-injecting our own
        let safe = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Convert bullet list lines (- item or * item at start of line)
        // Group consecutive bullet lines into a <ul>
        safe = safe.replace(/((?:^[ \t]*[-*][ \t]+.+\n?)+)/gm, (block) => {
            const items = block.split("\n")
                .filter(l => l.trim())
                .map(l => `<li>${l.replace(/^[ \t]*[-*][ \t]+/, "")}</li>`)
                .join("");
            return `<ul>${items}</ul>`;
        });

        // Inline: **bold**
        safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        // Inline: *italic* (but not already-processed **)
        safe = safe.replace(/\*([^*\n]+?)\*/g, "<em>$1</em>");
        // Inline: `code`
        safe = safe.replace(/`([^`]+?)`/g, "<code>$1</code>");

        // Convert remaining newlines to <br> (skip lines already inside list blocks)
        safe = safe.replace(/\n/g, "<br>");

        return safe;
    }

    function appendChatMessage(role, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}-message`;
        const formattedText = role === "assistant" ? formatMarkdown(text) : text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        msgDiv.innerHTML = `
            <div class="avatar">${role === "user" ? "👤" : "🤖"}</div>
            <div class="message-content">${formattedText}</div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendChatChart(chartJson) {
        const chartId = `chat-chart-${Date.now()}`;
        const msgDiv = document.createElement("div");
        msgDiv.className = "message assistant-message";
        msgDiv.innerHTML = `
            <div class="avatar">📊</div>
            <div class="message-content" style="width: 100%;"><div id="${chartId}" style="height: 300px;"></div></div>
        `;
        chatMessages.appendChild(msgDiv);
        Plotly.newPlot(chartId, chartJson.data, chartJson.layout, { responsive: true });
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    clearChatBtn.addEventListener("click", () => {
        chatMessages.innerHTML = `
            <div class="message assistant-message">
                <div class="avatar">🤖</div>
                <div class="message-content"><p>Chat history cleared. How else can I assist?</p></div>
            </div>
        `;
    });

    async function fetchSuggestions() {
        try {
            const res = await fetch("/suggestions");
            const data = await res.json();
            const container = document.getElementById("suggestions-container");

            if (data.suggestions && data.suggestions.length) {
                container.innerHTML = data.suggestions.map(s => `
                    <div class="chip" data-q="${s.question}">${s.question}</div>
                `).join("");

                container.querySelectorAll(".chip").forEach(c => {
                    c.addEventListener("click", () => {
                        chatInput.value = c.getAttribute("data-q");
                        chatForm.dispatchEvent(new Event("submit"));
                    });
                });
            }
        } catch (err) {
            console.error("Suggestions error:", err);
        }
    }

    // -------------------------------------------------------------------------
    // 5. Insights View
    // -------------------------------------------------------------------------
    const generateInsightsBtn = document.getElementById("generate-insights-btn");
    generateInsightsBtn.addEventListener("click", fetchInsights);

    async function fetchInsights() {
        showLoader("Generating automated AI insights...");
        try {
            const res = await fetch("/insights");
            const data = await res.json();
            const container = document.getElementById("insights-container");

            if (res.ok && data.insights && data.insights.length > 0) {
                container.innerHTML = data.insights.map(i => `
                    <div class="insight-card">
                        <div class="insight-header">
                            <h4 class="insight-title">${i.title}</h4>
                            <span class="badge ${i.level === 'Good' ? 'badge-good' : 'badge-warning'}">${i.level}</span>
                        </div>
                        <p style="color: #64748B; font-size: 0.875rem; margin-top: 0.5rem;">${i.explanation}</p>
                        <div style="margin-top: 0.75rem; font-weight: 600; font-size: 0.85rem; color: #2563EB;">
                            Indicator: ${i.metric}
                        </div>
                    </div>
                `).join("");
            } else {
                container.innerHTML = `
                    <div class="empty-state">
                        <p style="color: #EF4444;">${data.error || "Please upload a dataset first to generate AI insights."}</p>
                    </div>
                `;
            }
        } catch (err) {
            console.error("Insights Generation Error:", err);
            alert("Failed to reach server endpoint for insights.");
        } finally {
            hideLoader();
        }
    }

    // -------------------------------------------------------------------------
    // 6. Custom Visualization Render
    // -------------------------------------------------------------------------
    const renderChartBtn = document.getElementById("render-chart-btn");
    renderChartBtn.addEventListener("click", async () => {
        const chartType = document.getElementById("chart-type-select").value;
        const xCol = document.getElementById("x-axis-select").value;
        const yCol = document.getElementById("y-axis-select").value;

        showLoader("Building chart visual...");
        try {
            const res = await fetch("/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ chart_type: chartType, x_col: xCol, y_col: yCol })
            });
            const data = await res.json();

            if (res.ok) {
                const box = document.getElementById("plotly-chart-container");
                box.innerHTML = "";
                
                // Set explicit layout margins and dimensions
                data.chart.layout.autosize = true;
                data.chart.layout.width = null;
                data.chart.layout.height = 460;

                Plotly.newPlot("plotly-chart-container", data.chart.data, data.chart.layout, { 
                    responsive: true,
                    useResizeHandler: true 
                }).then(() => {
                    Plotly.Plots.resize(document.getElementById("plotly-chart-container"));
                });
            } else {
                alert(data.error);
            }
        } catch (err) {
            alert("Chart rendering failed.");
        } finally {
            hideLoader();
        }
    });

    // -------------------------------------------------------------------------
    // 7. Profile & Quality API Handlers
    // -------------------------------------------------------------------------
    async function fetchDatasetProfile() {
        try {
            const res = await fetch("/profile");
            const data = await res.json();
            if (res.ok) {
                const tbody = document.getElementById("profile-table").querySelector("tbody");
                tbody.innerHTML = data.profiles.map(p => `
                    <tr>
                        <td><strong>${p.column_name}</strong></td>
                        <td><code>${p.data_type}</code></td>
                        <td>${p.unique_values}</td>
                        <td>${p.missing_values}</td>
                        <td>${p.missing_percentage}%</td>
                        <td>${p.mean}</td>
                        <td>${p.median}</td>
                        <td>${p.min}</td>
                        <td>${p.max}</td>
                    </tr>
                `).join("");
            }
        } catch (e) {}
    }

    async function fetchQualityReport() {
        try {
            const res = await fetch("/quality");
            const data = await res.json();
            if (res.ok) {
                const q = data.quality;
                const container = document.getElementById("quality-dashboard-content");
                container.innerHTML = `
                    <div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 1.5rem;">
                        <div style="font-size: 2.5rem; font-weight: 700; color: ${q.score > 80 ? '#22C55E' : '#F59E0B'}">
                            ${q.score}/100
                        </div>
                        <div>
                            <h4>Overall Health Status: ${q.status}</h4>
                            <p style="color: #64748B;">${q.summary_message}</p>
                        </div>
                    </div>
                    <h4>Actionable Quality Recommendations</h4>
                    <ul style="margin-top: 0.5rem; padding-left: 1.25rem; color: #334155;">
                        ${q.recommendations.map(r => `<li style="margin-bottom: 0.35rem;">${r}</li>`).join("")}
                    </ul>
                `;
            }
        } catch (e) {}
    }

    // Loader Utilities
    function showLoader(text) {
        document.getElementById("loader-text").textContent = text || "Processing...";
        loader.classList.remove("hidden");
    }

    function hideLoader() {
        loader.classList.add("hidden");
    }

    // -------------------------------------------------------------------------
    // 8. Voice Transcription — Gemini Multilingual Speech-to-Text
    // Supports: English | বাংলা (Bengali) | हिन्दी (Hindi)
    // -------------------------------------------------------------------------

    const micBtn            = document.getElementById("mic-btn");
    const stopRecordingBtn  = document.getElementById("stop-recording-btn");
    const voiceRecordingBar = document.getElementById("voice-recording-bar");
    const voiceTimerEl      = document.getElementById("voice-timer");

    let mediaRecorder   = null;
    let audioChunks     = [];
    let recordingTimer  = null;
    let recordingSeconds = 0;

    // ── Toast helper ──────────────────────────────────────────────────────────
    function showVoiceToast(message, durationMs = 2800) {
        let toast = document.getElementById("voice-toast-el");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "voice-toast-el";
            toast.className = "voice-toast";
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.classList.add("show");
        clearTimeout(toast._hideTimer);
        toast._hideTimer = setTimeout(() => toast.classList.remove("show"), durationMs);
    }

    // ── Mic button: start recording ───────────────────────────────────────────
    micBtn.addEventListener("click", async () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            // Clicking mic again while recording stops it (same as Stop button)
            stopAndTranscribe();
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showVoiceToast("🎤 Microphone not supported in this browser.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            startRecording(stream);
        } catch (err) {
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                showVoiceToast("🔒 Microphone permission denied. Please allow mic access.");
            } else {
                showVoiceToast("⚠️ Could not access microphone: " + err.message);
            }
        }
    });

    stopRecordingBtn.addEventListener("click", () => stopAndTranscribe());

    // ── Start recording ───────────────────────────────────────────────────────
    function startRecording(stream) {
        audioChunks     = [];
        recordingSeconds = 0;

        // Pick the best supported MIME type
        const preferredTypes = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/mp4",
        ];
        const mimeType = preferredTypes.find(t => MediaRecorder.isTypeSupported(t)) || "";

        mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        mediaRecorder.addEventListener("dataavailable", e => {
            if (e.data && e.data.size > 0) audioChunks.push(e.data);
        });
        mediaRecorder.addEventListener("stop", () => {
            // Stop all microphone tracks to release the device
            stream.getTracks().forEach(t => t.stop());
            sendAudioToGemini(mimeType || "audio/webm");
        });

        mediaRecorder.start(250);  // collect chunks every 250ms

        // Update UI
        micBtn.classList.add("recording");
        voiceRecordingBar.classList.remove("hidden");
        voiceTimerEl.textContent = "0s";
        showVoiceToast("🎙️ Recording… speak your question");

        // Live timer
        recordingTimer = setInterval(() => {
            recordingSeconds++;
            voiceTimerEl.textContent = recordingSeconds + "s";

            // Auto-stop after 60 seconds to avoid huge payloads
            if (recordingSeconds >= 60) {
                showVoiceToast("⏱️ Max recording time reached. Transcribing…");
                stopAndTranscribe();
            }
        }, 1000);
    }

    // ── Stop recording ────────────────────────────────────────────────────────
    function stopAndTranscribe() {
        if (!mediaRecorder || mediaRecorder.state !== "recording") return;
        clearInterval(recordingTimer);
        mediaRecorder.stop();

        // Transition to "transcribing" state
        micBtn.classList.remove("recording");
        micBtn.classList.add("transcribing");
        voiceRecordingBar.classList.add("hidden");
        showVoiceToast("⏳ Transcribing with Gemini…");
    }

    // ── Send audio to /transcribe ─────────────────────────────────────────────
    async function sendAudioToGemini(mimeType) {
        if (!audioChunks.length) {
            resetMicState();
            showVoiceToast("❌ No audio captured. Please try again.");
            return;
        }

        const audioBlob = new Blob(audioChunks, { type: mimeType });
        const formData  = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        try {
            const res  = await fetch("/transcribe", { method: "POST", body: formData });
            const data = await res.json();

            if (res.ok && data.text) {
                chatInput.value = data.text;
                chatInput.focus();
                showVoiceToast("✅ Transcribed! Press Send or edit your question.");

                // Highlight the input briefly to show it was populated
                chatInput.style.borderColor = "#22C55E";
                chatInput.style.boxShadow   = "0 0 0 3px rgba(34,197,94,0.2)";
                setTimeout(() => {
                    chatInput.style.borderColor = "";
                    chatInput.style.boxShadow   = "";
                }, 2000);
            } else {
                const errMsg = data.error || "Transcription failed. Please try again.";
                showVoiceToast("❌ " + errMsg);
                console.error("Transcription error:", errMsg);
            }
        } catch (err) {
            showVoiceToast("❌ Could not reach the transcription server.");
            console.error("Transcription fetch error:", err);
        } finally {
            resetMicState();
        }
    }

    // ── Reset mic to idle state ───────────────────────────────────────────────
    function resetMicState() {
        micBtn.classList.remove("recording", "transcribing");
        voiceRecordingBar.classList.add("hidden");
        clearInterval(recordingTimer);
        mediaRecorder   = null;
        audioChunks     = [];
        recordingSeconds = 0;
        voiceTimerEl.textContent = "0s";
    }
});